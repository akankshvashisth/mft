from dataclasses import asdict, dataclass
import numpy as np
from fast_vollib import fast_black_scholes
from src.scanner.surface_scanner import CandidateCombo


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
    """Evaluates candidate option combos via P-measure Monte Carlo & 2D Spot x Vol grids."""

    def __init__(
        self,
        n_scenarios: int = 10_000,
        commission_per_leg: float = 0.65,
        slippage_fraction: float = 0.50,
        seed: int = 42,
    ) -> None:
        self.n_scenarios = n_scenarios
        self.commission_per_leg = commission_per_leg
        self.slippage_fraction = slippage_fraction
        self.rng = np.random.default_rng(seed)

    def _reprice_combo_vectors(
        self,
        combo: CandidateCombo,
        spot_vec: np.ndarray,
        vol_shift_vec: np.ndarray,
        rem_ttm_years: float,
    ) -> np.ndarray:
        """Vectorized fast-vollib valuation across N scenarios for all legs in combo."""
        n = len(spot_vec)
        total_val_per_share = np.zeros(n, dtype=np.float64)
        r_carry = combo.risk_free_rate - combo.dividend_yield

        for leg in combo.legs:
            K = np.full(n, leg.strike, dtype=np.float64)
            T = np.full(n, max(rem_ttm_years, 1e-6), dtype=np.float64)
            r = np.full(n, r_carry, dtype=np.float64)
            flag = np.full(n, "c" if leg.right == "C" else "p")
            sigma = np.clip(leg.svi_iv + vol_shift_vec, 0.03, 3.0)

            leg_prices = np.asarray(
                fast_black_scholes(
                    S=spot_vec, K=K, t=T, r=r, sigma=sigma, flag=flag, return_as="numpy"
                ),
                dtype=np.float64,
            )
            leg_pnl_per_share = leg.side * leg.qty * (leg_prices - leg.mid)
            total_val_per_share += leg_pnl_per_share

        return total_val_per_share * 100.0

    def evaluate_combo(
        self,
        combo: CandidateCombo,
        realized_vol_p: float = 0.165,
        horizon_days: float = 7.0,
        sleeve_equity_usd: float = 100_000.0,
        max_es_risk_pct: float = 0.01,
    ) -> RiskAssessment:
        """Runs P-measure Monte Carlo EV/VaR/ES + deterministic 2D Spot x Vol stress grid."""
        dt_years = min(horizon_days, combo.ttm_days) / 365.25
        rem_ttm_years = max((combo.ttm_days - min(horizon_days, combo.ttm_days)) / 365.25, 1e-6)

        half_spread_usd = sum(
            self.slippage_fraction * max(leg.ask - leg.bid, 0.01) * leg.qty * 100.0
            for leg in combo.legs
        )
        comm_usd = sum(self.commission_per_leg * leg.qty for leg in combo.legs)
        friction_usd = half_spread_usd + comm_usd
        entry_credit_usd = combo.net_credit * 100.0

        # P-measure correlated Spot & Vol draws (leverage effect rho = -0.65)
        z1 = self.rng.standard_normal(self.n_scenarios)
        z2 = self.rng.standard_normal(self.n_scenarios)
        rho_sv = -0.65
        z_vol = rho_sv * z1 + np.sqrt(1.0 - rho_sv * rho_sv) * z2

        drift = (combo.risk_free_rate - combo.dividend_yield - 0.5 * realized_vol_p ** 2) * dt_years
        spot_sim = combo.spot * np.exp(drift + realized_vol_p * np.sqrt(dt_years) * z1)
        vol_shift_sim = 0.35 * realized_vol_p * np.sqrt(dt_years) * z_vol

        gross_pnl = self._reprice_combo_vectors(combo, spot_sim, vol_shift_sim, rem_ttm_years)
        net_pnl = gross_pnl - friction_usd

        net_ev = float(np.mean(net_pnl))
        prob_profit = float(np.mean(net_pnl > 0.0))
        var_95 = float(np.percentile(net_pnl, 5.0))
        es_95 = float(np.mean(net_pnl[net_pnl <= var_95]))
        var_99 = float(np.percentile(net_pnl, 1.0))
        es_99 = float(np.mean(net_pnl[net_pnl <= var_99]))

        # 2D Spot x Vol Scenario P&L Grid at horizon_days
        spot_shocks = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
        vol_shocks = [-0.05, 0.0, 0.05, 0.10, 0.15]
        grid: list[list[float]] = []
        for ds in spot_shocks:
            s_vec = np.full(len(vol_shocks), combo.spot * (1.0 + ds), dtype=np.float64)
            v_vec = np.asarray(vol_shocks, dtype=np.float64)
            row_pnl = self._reprice_combo_vectors(combo, s_vec, v_vec, rem_ttm_years) - friction_usd
            grid.append([round(float(x), 2) for x in row_pnl])

        if combo.max_width > 0.0:
            margin_est = max((combo.max_width - combo.net_credit) * 100.0, 100.0)
        else:
            margin_est = max(abs(min(min(r) for r in grid)), 500.0)

        max_risk_budget = sleeve_equity_usd * max_es_risk_pct
        tail_loss_per_lot = abs(min(es_99, -1.0))
        rec_qty = int(np.floor(max_risk_budget / tail_loss_per_lot)) if net_ev > 0.0 else 0
        approved = (net_ev > 0.0) and (rec_qty >= 1)

        return RiskAssessment(
            combo_name=combo.name,
            strategy_type=combo.strategy_type,
            horizon_days=round(horizon_days, 2),
            realized_vol_p=round(realized_vol_p, 4),
            entry_credit_usd=round(entry_credit_usd, 2),
            friction_cost_usd=round(friction_usd, 2),
            net_ev_usd=round(net_ev, 2),
            prob_profit=round(prob_profit, 4),
            var_95_usd=round(var_95, 2),
            es_95_usd=round(es_95, 2),
            var_99_usd=round(var_99, 2),
            es_99_usd=round(es_99, 2),
            max_margin_est_usd=round(margin_est, 2),
            ev_to_es99_ratio=round(net_ev / tail_loss_per_lot, 4),
            recommended_contracts=rec_qty,
            approved=approved,
            spot_shocks_pct=[round(s * 100.0, 1) for s in spot_shocks],
            vol_shocks_pts=[round(v * 100.0, 1) for v in vol_shocks],
            scenario_grid_usd=grid,
        )
