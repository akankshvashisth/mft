from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from src.common.constants import DAYS_PER_YEAR
from src.quant.risk_measures import contracts_for_tail_budget, tail_loss_per_lot, tail_metrics
from src.quant.stress import shock_grid
from src.scanner.combos import CandidateCombo, CandidateLeg

# price_fn(S, K, t, r, sigma, flag, q) -> Black-Scholes-Merton prices; flag holds "c"/"p"
PriceFn = Callable[..., np.ndarray]

SPOT_SHOCKS = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)
VOL_SHOCKS = (-0.05, 0.0, 0.05, 0.10, 0.15)
SPOT_VOL_CORRELATION = -0.65


def fast_vollib_price(S, K, t, r, sigma, flag, q) -> np.ndarray:
    """fast_vollib Black-Scholes with a continuous dividend yield: spot discounted by exp(-q t), true rate r."""
    from fast_vollib import fast_black_scholes  # Lazy: tests inject src.quant.black_scholes.bsm_prices instead

    spot_ex_div = np.asarray(S, dtype=np.float64) * np.exp(-np.asarray(q, dtype=np.float64) * np.asarray(t, dtype=np.float64))
    return np.asarray(
        fast_black_scholes(S=spot_ex_div, K=K, t=t, r=r, sigma=sigma, flag=flag, return_as="numpy"),
        dtype=np.float64,
    )


def friction_usd(legs: Sequence[CandidateLeg], slippage_fraction: float, commission_per_leg: float) -> float:
    """Per 1-lot cost of crossing slippage_fraction of each leg's spread plus per-contract commissions."""
    half_spread_usd = sum(slippage_fraction * max(leg.ask - leg.bid, 0.01) * leg.qty * 100.0 for leg in legs)
    comm_usd = sum(commission_per_leg * leg.qty for leg in legs)
    return half_spread_usd + comm_usd


def defined_risk_margin_usd(combo: CandidateCombo, grid: list[list[float]]) -> float:
    """Width minus credit for defined-risk spreads, else the worst stress loss (floors $100 / $500)."""
    if combo.max_width > 0.0:
        return max((combo.max_width - combo.net_credit) * 100.0, 100.0)
    return max(abs(min(min(r) for r in grid)), 500.0)


@dataclass
class RiskAssessment:
    combo_name: str
    strategy_type: str
    horizon_days: float
    realized_vol_p: float
    entry_credit_usd: float          # Per 1-lot ($100 multiplier)
    friction_cost_usd: float         # Half-spread + IBKR commission per 1-lot
    net_ev_usd: float                # P-measure Net EV per 1-lot
    prob_profit: float               # P(Net P&L > 0)
    var_95_usd: float                # 5th percentile P&L per 1-lot
    es_95_usd: float                 # Mean P&L below 5th percentile per 1-lot
    var_99_usd: float                # 1st percentile P&L per 1-lot
    es_99_usd: float                 # Mean P&L below 1st percentile per 1-lot
    max_margin_est_usd: float        # Worst defined-risk or stress margin per 1-lot
    ev_to_es99_ratio: float          # Net EV / |ES_99|
    recommended_contracts: int       # Sized so Q * |ES_99| <= max_risk_per_trade_usd
    approved: bool                   # True if net_ev_usd > 0 and recommended_contracts >= 1
    spot_shocks_pct: list[float]
    vol_shocks_pts: list[float]
    scenario_grid_usd: list[list[float]]  # Shape (len(spot_shocks), len(vol_shocks)) per 1-lot

    def to_dict(self) -> dict:
        return asdict(self)


class ScenarioPricer:
    """Evaluates candidate option combos via P-measure Monte Carlo & 2D Spot x Vol grids.

    Every evaluation draws its scenarios from a fresh default_rng(seed) (common random numbers), so
    candidates are compared on identical paths. price_fn defaults to fast_vollib; bsm_prices (QuantLib) is a
    drop-in alternative that the tests inject.
    """

    def __init__(
        self,
        n_scenarios: int = 10_000,
        commission_per_leg: float = 0.65,
        slippage_fraction: float = 0.50,
        seed: int = 42,
        price_fn: PriceFn | None = None,
    ) -> None:
        self.n_scenarios = n_scenarios
        self.commission_per_leg = commission_per_leg
        self.slippage_fraction = slippage_fraction
        self.seed = seed
        self.price_fn: PriceFn = price_fn or fast_vollib_price

    def _reprice_combo_vectors(
        self,
        combo: CandidateCombo,
        spot_vec: np.ndarray,
        vol_shift_vec: np.ndarray,
        rem_ttm_years: float,
    ) -> np.ndarray:
        """Vectorized valuation across N scenarios for all legs in combo: P&L per 1-lot vs the legs' mids."""
        n = len(spot_vec)
        total_val_per_share = np.zeros(n, dtype=np.float64)
        T = np.full(n, max(rem_ttm_years, 1e-6), dtype=np.float64)
        r = np.full(n, combo.risk_free_rate, dtype=np.float64)
        q = np.full(n, combo.dividend_yield, dtype=np.float64)

        for leg in combo.legs:
            K = np.full(n, leg.strike, dtype=np.float64)
            flag = np.full(n, "c" if leg.right == "C" else "p")
            sigma = np.clip(leg.svi_iv + vol_shift_vec, 0.03, 3.0)
            leg_prices = np.asarray(self.price_fn(spot_vec, K, T, r, sigma, flag, q), dtype=np.float64)
            total_val_per_share += leg.side * leg.qty * (leg_prices - leg.mid)

        return total_val_per_share * 100.0

    def friction_usd(self, combo: CandidateCombo) -> float:
        return friction_usd(combo.legs, self.slippage_fraction, self.commission_per_leg)

    def evaluate_combo(
        self,
        combo: CandidateCombo,
        realized_vol_p: float = 0.165,
        horizon_days: float = 7.0,
        sleeve_equity_usd: float = 100_000.0,
        max_es_risk_pct: float = 0.01,
    ) -> RiskAssessment:
        """Runs P-measure Monte Carlo EV/VaR/ES + deterministic 2D Spot x Vol stress grid."""
        held_days = min(horizon_days, combo.ttm_days)
        dt_years = held_days / DAYS_PER_YEAR
        rem_ttm_years = max((combo.ttm_days - held_days) / DAYS_PER_YEAR, 1e-6)

        friction = self.friction_usd(combo)
        entry_credit_usd = combo.net_credit * 100.0

        # P-measure correlated Spot & Vol draws (leverage effect), identical for every combo
        rng = np.random.default_rng(self.seed)
        rho_sv = SPOT_VOL_CORRELATION
        draws = rng.multivariate_normal(
            [0.0, 0.0], [[1.0, rho_sv], [rho_sv, 1.0]], size=self.n_scenarios, method="cholesky"
        )
        z_spot, z_vol = draws[:, 0], draws[:, 1]

        drift = (combo.risk_free_rate - combo.dividend_yield - 0.5 * realized_vol_p ** 2) * dt_years
        spot_sim = combo.spot * np.exp(drift + realized_vol_p * np.sqrt(dt_years) * z_spot)
        vol_shift_sim = 0.35 * realized_vol_p * np.sqrt(dt_years) * z_vol

        net_pnl = self._reprice_combo_vectors(combo, spot_sim, vol_shift_sim, rem_ttm_years) - friction
        tail = tail_metrics(net_pnl)

        # 2D Spot x Vol Scenario P&L Grid at horizon_days
        grid = shock_grid(
            SPOT_SHOCKS,
            VOL_SHOCKS,
            lambda ds, dv: self._reprice_combo_vectors(combo, combo.spot * (1.0 + ds), dv, rem_ttm_years) - friction,
        )
        margin_est = defined_risk_margin_usd(combo, grid)

        max_risk_budget = sleeve_equity_usd * max_es_risk_pct
        rec_qty = contracts_for_tail_budget(tail.mean, tail.es_99, max_risk_budget)
        approved = (tail.mean > 0.0) and (rec_qty >= 1)

        return RiskAssessment(
            combo_name=combo.name,
            strategy_type=combo.strategy_type,
            horizon_days=round(horizon_days, 2),
            realized_vol_p=round(realized_vol_p, 4),
            entry_credit_usd=round(entry_credit_usd, 2),
            friction_cost_usd=round(friction, 2),
            net_ev_usd=round(tail.mean, 2),
            prob_profit=round(tail.prob_profit, 4),
            var_95_usd=round(tail.var_95, 2),
            es_95_usd=round(tail.es_95, 2),
            var_99_usd=round(tail.var_99, 2),
            es_99_usd=round(tail.es_99, 2),
            max_margin_est_usd=round(margin_est, 2),
            ev_to_es99_ratio=round(tail.mean / tail_loss_per_lot(tail.es_99), 4),
            recommended_contracts=rec_qty,
            approved=approved,
            spot_shocks_pct=[round(s * 100.0, 1) for s in SPOT_SHOCKS],
            vol_shocks_pts=[round(v * 100.0, 1) for v in VOL_SHOCKS],
            scenario_grid_usd=grid,
        )
