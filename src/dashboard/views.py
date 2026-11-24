"""Pure builders for the dashboard's banners, tables and derived series."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.common.market import symbol_of
from src.common.numeric import is_number
from src.common.timeutil import days_to_years
from src.dashboard.formatting import fmt, pct_or_none
from src.dashboard.state import age_seconds
from src.quant.density import breeden_litzenberger_density
from src.quant.surface_metrics import BIAS_BULLISH_RR, BIAS_DEFENSIVE, BIAS_SELECTIVE, BIAS_SHORT_VRP
from src.quant.svi import roger_lee_utilization, svi_implied_vol

WAITING = "Waiting..."
Payload = Mapping[str, Any]


def banner(state: Mapping[str, Any], now_ns: int) -> tuple[str, str]:
    """(level, markdown) of the global banner: "warning", "error" or "success"."""
    risk_status = state.get("risk_status")
    if risk_status is None:
        return "warning", "Risk Daemon heartbeat missing (`mft:risk:status`). Ensure `src.risk.risk_daemon` is running."
    if risk_status.get("halted"):
        return "error", f"**SYSTEM HALTED** — Active Reasons: `{', '.join(risk_status.get('reasons', []))}`"
    bias = (state.get("metrics") or {}).get("strategy_bias", "CALIBRATING")
    return "success", (
        f"**SYSTEM ARMED & STREAMING** | Regime Age: `{age_seconds(state.get('regime'), now_ns)}s` | "
        f"Surface Age: `{age_seconds(state.get('surface'), now_ns)}s` | Active Bias: **{bias}**"
    )


def vrp_spread(surface: Payload, rv: Payload, metrics: Payload) -> float | None:
    """Published VRP, else ATM IV - RV20 when both are known."""
    if metrics.get("vrp_spread") is not None:
        return metrics.get("vrp_spread")
    atm_iv, rv_20d = surface.get("atm_iv"), rv.get("rv_20d")
    return (atm_iv - rv_20d) if (atm_iv is not None and rv_20d is not None) else None


def actor_health_rows(
    regime: Payload, surface: Payload, rv: Payload, metrics: Payload, port: Payload, now_ns: int
) -> list[dict[str, Any]]:
    greeks = port.get("greeks") or {}
    return [
        {
            "Actor": "1. MacroRegimeActor",
            "Role": "Term-Structure & Stress Gate",
            "Age (s)": age_seconds(regime, now_ns),
            "Primary Output": (
                f"VIX={regime.get('vix')} | VIX3M={regime.get('vix3m')} | Ratio={regime.get('ratio')} | "
                f"ShortVolAllowed={regime.get('short_vol_allowed')}"
                if regime
                else WAITING
            ),
        },
        {
            "Actor": "2. FastVolActor",
            "Role": "SVI Calibration & Arb Validation",
            "Age (s)": age_seconds(surface, now_ns),
            "Primary Output": (
                f"Spot=${surface.get('spot')} | Fwd=${surface.get('forward')} | ATM_IV={fmt(surface.get('atm_iv', 0), '.2%')} | "
                f"RMSE={surface.get('rmse_vol_pts')}pts | ArbFree={surface.get('arb_free')}"
                if surface
                else WAITING
            ),
        },
        {
            "Actor": "3. RealizedVolActor",
            "Role": "P-Measure RV, Jumps & Correlation",
            "Age (s)": age_seconds(rv, now_ns),
            "Primary Output": (
                f"RV5={fmt(rv.get('rv_5d', 0), '.2%')} | RV20={fmt(rv.get('rv_20d', 0), '.2%')} | "
                f"Pct={rv.get('rv_percentile')}% | Jump={fmt(rv.get('jump_ratio', 0), '.1%')} | "
                f"8-ETF Corr={rv.get('avg_pair_corr')}"
                if rv
                else WAITING
            ),
        },
        {
            "Actor": "4. SurfaceRegimeMetricsActor",
            "Role": "Skew/Butterfly, VRP & Strategy Bias",
            "Age (s)": age_seconds(metrics, now_ns),
            "Primary Output": (
                f"RR25={fmt(metrics.get('rr_25d', 0), '+.2%')} | BF25={fmt(metrics.get('bf_25d', 0), '+.2%')} | "
                f"VRP={fmt(metrics.get('vrp_spread', 0), '+.2%')} | Bias={metrics.get('strategy_bias')}"
                if metrics
                else WAITING
            ),
        },
        {
            "Actor": "5. PortfolioAnalyticsActor",
            "Role": "Book Greeks, Stress Grid & Attribution",
            "Age (s)": age_seconds(port, now_ns),
            "Primary Output": (
                f"Open={port.get('open_positions', 0)} | BetaDelta={fmt(greeks.get('beta_delta_shares', 0), '+.1f')}sh | "
                f"Vega=${fmt(greeks.get('net_vega_1v_usd', 0), '+.2f')}/v | "
                f"Theta=${fmt(greeks.get('net_theta_day_usd', 0), '+.2f')}/d"
                if port
                else WAITING
            ),
        },
    ]


def _percentile(payload: Payload) -> float:
    value = payload.get("rv_percentile")
    return float(value) if is_number(value) else 0.0


def universe_snapshot_rows(rv_universe: Mapping[str, Payload]) -> list[dict[str, str]]:
    """Compact universe RV table for the overview, highest 1-year percentile first."""
    return [
        {
            "Symbol": symbol_of(sym),
            "RV 5d": fmt(v.get("rv_5d", 0), ".1%"),
            "RV 20d": fmt(v.get("rv_20d", 0), ".1%"),
            "1Y Pct": f"{fmt(v.get('rv_percentile', 0), '.0f')}%",
            "Jump %": fmt(v.get("jump_ratio", 0), ".0%"),
        }
        for sym, v in sorted(rv_universe.items(), key=lambda kv: _percentile(kv[1]), reverse=True)
    ]


def universe_rv_rows(rv_universe: Mapping[str, Payload]) -> list[dict[str, Any]]:
    """Multi-horizon RV table (percent units) for the realized-vol page."""
    return [
        {
            "Symbol": sym,
            "Spot / Close": d.get("spot"),
            "RV 5d (%)": pct_or_none(d.get("rv_5d"), 2),
            "RV 10d (%)": pct_or_none(d.get("rv_10d"), 2),
            "RV 20d (%)": pct_or_none(d.get("rv_20d"), 2),
            "RV 60d (%)": pct_or_none(d.get("rv_60d"), 2),
            "1Y Percentile": d.get("rv_percentile"),
            "Jump Ratio (%)": pct_or_none(d.get("jump_ratio"), 1),
        }
        for sym, d in rv_universe.items()
    ]


def svi_parameter_rows(svi_params: Sequence[float]) -> list[dict[str, str]]:
    """Raw SVI parameters and the share of Roger Lee's wing-slope bound in use."""
    names = ["a (variance level)", "b (wing slope)", "ρ (skew tilt)", "m (center shift)", "σ (ATM curvature)"]
    rows = [{"Parameter": name, "Value": f"{float(value):.6f}"} for name, value in zip(names, svi_params)]
    rows.append({"Parameter": "Roger Lee Bound", "Value": f"{roger_lee_utilization(svi_params):.1%}"})
    return rows


def surface_ttm_years(surface: Payload) -> float:
    return max(days_to_years(float(surface.get("ttm_days", 25.0))), 1e-4)


def surface_forward(surface: Payload) -> float:
    return float(surface.get("forward", surface.get("spot", 700.0)))


def dense_smile(surface: Payload, points: int = 120) -> tuple[np.ndarray, np.ndarray] | None:
    """(log-moneyness, SVI IV in %) on a dense grid slightly wider than the quoted strikes."""
    params = surface.get("svi_params") or []
    k = np.asarray(surface.get("log_moneyness") or [], dtype=float)
    if len(params) != 5 or k.size == 0:
        return None
    k_dense = np.linspace(float(k.min()) - 0.01, float(k.max()) + 0.01, points)
    return k_dense, svi_implied_vol(k_dense, surface_ttm_years(surface), params) * 100.0


def smile_density(surface: Payload) -> tuple[np.ndarray, np.ndarray] | None:
    """Breeden-Litzenberger risk-neutral density (strikes, q(K)) of the published slice.

    None without SVI parameters or when QuantLib rejects them as inadmissible.
    """
    params = surface.get("svi_params") or []
    if len(params) != 5:
        return None
    try:
        return breeden_litzenberger_density(surface_forward(surface), surface_ttm_years(surface), params)
    except RuntimeError:
        return None


def decision_matrix_rows() -> list[dict[str, str]]:
    return [
        {"Regime Condition": "VIX/VIX3M >= 1.00 OR VRP < -1.5%", "Assigned Bias": BIAS_DEFENSIVE, "Action": "Stand down short-vol; favor long convexity / hedges"},
        {"Regime Condition": "VRP >= +2.0% AND BF25 >= +0.5%", "Assigned Bias": BIAS_SHORT_VRP, "Action": "Harvest rich ATM & wing premium via defined-risk spreads"},
        {"Regime Condition": "RR25 Percentile <= 20%", "Assigned Bias": BIAS_BULLISH_RR, "Action": "Exploit compressed put skew for upside asymmetry"},
        {"Regime Condition": "Balanced Contango & Moderate VRP", "Assigned Bias": BIAS_SELECTIVE, "Action": "Trade only top SurfaceScanner residual dislocations"},
    ]


def attribution_rows(attribution: Payload) -> list[dict[str, str]]:
    components = [
        ("Actual P&L Change", "actual_pnl_change_usd"),
        ("Delta Explain (Δ · ΔS)", "delta_pnl_usd"),
        ("Gamma Explain (½Γ · ΔS²)", "gamma_pnl_usd"),
        ("Vega Explain (𝒱 · Δσ)", "vega_pnl_usd"),
        ("Theta Explain (Θ · Δt)", "theta_pnl_usd"),
        ("Unexplained Residual", "residual_pnl_usd"),
    ]
    return [{"Component": label, "USD ($)": f"${fmt(attribution.get(key, 0), '+.2f')}"} for label, key in components]


def forward_comparison_rows(futures: Payload, surface: Payload) -> list[dict[str, Any]]:
    return [
        {
            "Source": "FastVolActor (Option Put-Call Parity)",
            "Horizon (Days)": surface.get("ttm_days"),
            "Implied Forward ($)": futures.get("parity_fwd"),
            "Implied Carry q (%)": pct_or_none(surface.get("implied_q")),
            "Notes": f"Estimated from {surface.get('parity_pairs', 0)} ATM SPY option pair(s)",
        },
        {
            "Source": "FuturesContextActor (CME MESZ26.CME)",
            "Horizon (Days)": futures.get("mes_ttm_days"),
            "Implied Forward ($)": futures.get("mes_scaled_fwd"),
            "Implied Carry q (%)": pct_or_none(futures.get("futures_implied_q")),
            "Notes": (
                f"Model-free CME futures scaled by {fmt(futures.get('spx_spy_ratio'), '.5f')} "
                f"({futures.get('ratio_source', 'UNAVAILABLE')})"
            ),
        },
        {
            "Source": "Futures-Implied Forward @ Option Expiry",
            "Horizon (Days)": surface.get("ttm_days"),
            "Implied Forward ($)": futures.get("fut_implied_opt_fwd"),
            "Implied Carry q (%)": pct_or_none(futures.get("futures_implied_q")),
            "Notes": (
                f"Parity vs. Futures Divergence = {fmt(futures.get('parity_vs_fut_diff_usd'), '+.3f')} "
                f"({fmt(futures.get('parity_vs_fut_diff_bps'), '+.1f')} bps)"
            ),
        },
    ]


def risk_transition_rows(transitions: Sequence[Payload]) -> list[dict[str, Any]]:
    return [
        {
            "Timestamp (UTC)": t.get("evaluated_at_utc"),
            "Event": t.get("event", "STATE_CHANGE"),
            "Halted": t.get("to_halted", t.get("halted")),
            "VIX Ratio": t.get("vix_ratio"),
            "Regime Age (s)": t.get("regime_age_sec"),
            "Reasons": ", ".join(t.get("reasons", [])) or "None (Healthy)",
        }
        for t in transitions
    ]
