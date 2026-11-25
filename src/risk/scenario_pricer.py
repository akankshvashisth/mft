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
STUDENT_T_DOF = 5.0
TAIL_CORR_DOWN = 0.85
TAIL_CORR_UP = 0.40
BETA_LEV = 0.7
VOL_OF_VOL_FRAC = 0.35
SKEW_SLOPE = 0.25  # Sticky-delta skew slope: ∆σ per unit ∆ln(K/F)
MIN_VOL = 0.03


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


def round_trip_friction_usd(
    legs: Sequence[CandidateLeg],
    slippage_fraction: float,
    commission_per_leg: float,
    exit_slippage_fraction: float | None = None,
) -> float:
    """Round-trip friction: entry + exit costs.

    When ``exit_slippage_fraction`` is provided (same units as ``slippage_fraction``),
    exit spread crossing is deducted separately from entry crossing, enabling asymmetric
    friction when the spread is wider at exit (e.g. during vol spikes).
    """
    exit_frac = exit_slippage_fraction if exit_slippage_fraction is not None else slippage_fraction
    entry_friction = friction_usd(legs, slippage_fraction, commission_per_leg)
    exit_friction = friction_usd(legs, exit_frac, commission_per_leg)
    return entry_friction + exit_friction


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

    Every evaluation draws its scenarios from a deterministic local ``default_rng(seed)`` (Common Random
    Numbers), so candidates are compared on identical paths regardless of evaluation order.
    price_fn defaults to fast_vollib; bsm_prices (QuantLib) is a drop-in alternative that the tests inject.

    Uses Student-t (:math:`\nu=5`) spot shocks with tail correlation (0.85 down / 0.40 up),
    leverage-correlated IV shocks (β=0.7), and sticky-delta skew slide.  Friction deducts entry
    **and** exit costs when ``horizon_days < combo.ttm_days``.
    """

    def __init__(
        self,
        n_scenarios: int = 10_000,
        commission_per_leg: float = 0.65,
        slippage_fraction: float = 0.50,
        exit_slippage_fraction: float | None = None,
        seed: int = 42,
        price_fn: PriceFn | None = None,
    ) -> None:
        self.n_scenarios = n_scenarios
        self.commission_per_leg = commission_per_leg
        self.slippage_fraction = slippage_fraction
        self.exit_slippage_fraction = exit_slippage_fraction
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
        """Runs P-measure Monte Carlo EV/VaR/ES + deterministic 2D Spot x Vol stress grid.

        Shock model (Student-t ν=5 with tail correlation, leverage vol shock, sticky-delta skew):
        - Spot returns are Student-t(5) scaled to unit variance.
        - Tail correlation: 0.85 on down moves, 0.40 on up moves.
        - IV shock = -β * r_S + η * sqrt(dt) * Z_vol + skew_slope * ∆ln(K/F).
        - Common Random Numbers via deterministic ``default_rng(self.seed)``.
        - Friction: entry-only when held to expiry, else round-trip (entry + exit).
        """
        held_days = min(horizon_days, combo.ttm_days)
        dt_years = held_days / DAYS_PER_YEAR
        rem_ttm_years = max((combo.ttm_days - held_days) / DAYS_PER_YEAR, 1e-6)

        # Round-trip friction when exiting before expiry
        is_round_trip = held_days < combo.ttm_days
        if is_round_trip:
            friction = round_trip_friction_usd(
                combo.legs, self.slippage_fraction, self.commission_per_leg, self.exit_slippage_fraction
            )
        else:
            friction = self.friction_usd(combo)
        entry_credit_usd = combo.net_credit * 100.0

        # Common Random Numbers: deterministic seed per call
        rng = np.random.default_rng(self.seed)
        scale_t = np.sqrt((STUDENT_T_DOF - 2.0) / STUDENT_T_DOF)  # unit-variance scaling

        # Student-t 5 spot draws
        z_spot_raw = rng.standard_t(df=STUDENT_T_DOF, size=self.n_scenarios) / scale_t

        # Tail correlation: mix market and idiosyncratic components
        z_market = z_spot_raw.copy()
        z_idio = rng.standard_t(df=STUDENT_T_DOF, size=self.n_scenarios) / scale_t
        rho = np.where(z_market < 0.0, TAIL_CORR_DOWN, TAIL_CORR_UP)
        z_spot = rho * z_market + np.sqrt(1.0 - rho ** 2) * z_idio

        # Leverage-correlated vol shock: -β * r_S + η * sqrt(dt) * Z_vol + skew_adjustment
        z_vol_raw = rng.standard_normal(size=self.n_scenarios)
        r_S = realized_vol_p * np.sqrt(dt_years) * z_spot  # log-return
        vol_lev = -BETA_LEV * r_S / max(dt_years, 1e-6)  # vol pts / year
        vol_of_vol = VOL_OF_VOL_FRAC * realized_vol_p * np.sqrt(dt_years) * z_vol_raw

        # Sticky-delta skew slide: approximate ∆ln(K/F) per leg
        skew_shift = 0.0
        for leg in combo.legs:
            k = np.log(leg.strike / combo.spot)
            skew_shift += abs(leg.side) * SKEW_SLOPE * k * z_spot
        skew_adjust = skew_shift / max(len(combo.legs), 1)

        vol_shift_sim = vol_lev * dt_years + vol_of_vol + skew_adjust * realized_vol_p * np.sqrt(dt_years)

        # Exact BSM carry: drift uses risk-free rate - dividend yield
        drift = (combo.risk_free_rate - combo.dividend_yield - 0.5 * realized_vol_p ** 2) * dt_years
        spot_sim = combo.spot * np.exp(drift + realized_vol_p * np.sqrt(dt_years) * z_spot)

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
