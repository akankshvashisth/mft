from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import redis
import streamlit as st

st.set_page_config(
    page_title="Institutional VRP & Surface Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALKEY_URL = os.environ.get("VALKEY_URL", "redis://127.0.0.1:6379/0")


@st.cache_resource
def get_redis() -> redis.Redis:
    return redis.from_url(VALKEY_URL)


def load_json(rdb: redis.Redis, key: str) -> dict[str, Any] | None:
    raw = rdb.get(key)
    if not raw:
        return None
    try:
        val = json.loads(raw.decode("utf-8"))
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def load_json_list(rdb: redis.Redis, key: str, count: int = 200) -> list[dict[str, Any]]:
    items = rdb.lrange(key, 0, count - 1)
    out: list[dict[str, Any]] = []
    for item in items:
        try:
            obj = json.loads(item.decode("utf-8"))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
    return out


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _bs_price(
    F: float, K: np.ndarray, T: float, sigma: np.ndarray, r: float = 0.0425, is_call: bool = True
) -> np.ndarray:
    T_eff = max(T, 1e-6)
    sig_eff = np.maximum(sigma, 1e-6)
    vol_sqrt_t = sig_eff * np.sqrt(T_eff)
    d1 = (np.log(F / K) + 0.5 * (sig_eff ** 2) * T_eff) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    df = math.exp(-r * T_eff)
    if is_call:
        return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return df * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def _bs_delta(
    F: float, K: np.ndarray, T: float, sigma: np.ndarray, r: float = 0.0425, is_call: bool = True
) -> np.ndarray:
    T_eff = max(T, 1e-6)
    sig_eff = np.maximum(sigma, 1e-6)
    d1 = (np.log(F / K) + 0.5 * (sig_eff ** 2) * T_eff) / (sig_eff * np.sqrt(T_eff))
    df = math.exp(-r * T_eff)
    return df * _norm_cdf(d1) if is_call else df * (_norm_cdf(d1) - 1.0)


def _eval_svi_iv(k: np.ndarray, svi_params: list[float], T: float) -> np.ndarray:
    a, b, rho, m, sigma = svi_params
    dk = k - m
    w = np.maximum(a + b * (rho * dk + np.sqrt(dk * dk + sigma * sigma)), 1e-8)
    return np.sqrt(w / max(T, 1e-6))


def _age_seconds(obj: dict[str, Any] | None) -> float | None:
    if not obj:
        return None
    now_ns = time.time_ns()
    for field in ("ts_ns", "event_ts_ns", "ts_event"):
        val = obj.get(field)
        if isinstance(val, (int, float)) and val > 1e15:
            return round(max(0.0, (now_ns - float(val)) / 1e9), 1)
    return None


rdb = get_redis()

# --- Shared Sidebar Controls ---
with st.sidebar:
    st.header("Command Controls")
    auto_refresh = st.toggle(
        "Auto-refresh (5s)",
        value=True,
        help="Polls Valkey every 5 seconds for live updates from all 5 NautilusTrader actors and the risk daemon.",
    )
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()

    st.divider()
    st.subheader("Fail-Closed Kill-Switch")
    manual_halt_raw = rdb.get("mft:risk:manual_halt")
    manual_halted = bool(
        manual_halt_raw and manual_halt_raw.decode("utf-8").strip().lower() in {"1", "true", "yes", "on"}
    )

    if manual_halted:
        st.error("MANUAL HALT ENGAGED (`mft:risk:manual_halt=1`)")
        if st.button("Release Manual Kill-Switch", type="primary", use_container_width=True):
            rdb.delete("mft:risk:manual_halt")
            st.rerun()
    else:
        st.success("Manual Kill-Switch Disarmed")
        if st.button("ENGAGE MANUAL KILL-SWITCH", type="secondary", use_container_width=True):
            rdb.set("mft:risk:manual_halt", "1")
            st.rerun()

    st.caption(
        "Sets `mft:risk:manual_halt=1` in Valkey. `OutOfProcessRiskDaemon` detects this within 2s "
        "and blocks all new strategy entries."
    )

    st.divider()
    surface_keys = sorted(k.decode("utf-8") for k in rdb.keys("mft:surface:*"))
    selected_surface_key = st.selectbox(
        "Active Underlying Surface",
        options=surface_keys or ["mft:surface:SPY.ARCA"],
        help="Selects the primary underlying symbol inspected across the Overview, FastVolActor, and SurfaceMetrics pages.",
    )
    underlying_sym = selected_surface_key.replace("mft:surface:", "")

# --- Shared Live State Loader ---
def get_live_state() -> dict[str, Any]:
    return {
        "regime": load_json(rdb, "mft:regime:latest"),
        "futures": load_json(rdb, "mft:futures:latest"),
        "vrp_strat": load_json(rdb, "mft:strategy:vrp:latest"),
        "surface": load_json(rdb, selected_surface_key),
        "rv_primary": load_json(rdb, f"mft:rv:{underlying_sym}"),
        "rv_universe": load_json(rdb, "mft:rv:universe"),
        "metrics": load_json(rdb, f"mft:metrics:{underlying_sym}"),
        "portfolio": load_json(rdb, "mft:portfolio:risk"),
        "risk_status": load_json(rdb, "mft:risk:status"),
        "risk_history": load_json_list(rdb, "mft:risk:history", count=200),
        "risk_transitions": load_json_list(rdb, "mft:risk:transitions", count=30),
    }


def render_global_banner(state: dict[str, Any]) -> None:
    risk_status = state["risk_status"]
    regime = state["regime"]
    surface = state["surface"]
    metrics = state["metrics"]

    if risk_status is None:
        st.warning("Risk Daemon heartbeat missing (`mft:risk:status`). Ensure `src.risk.risk_daemon` is running.")
    elif risk_status.get("halted"):
        st.error(f"**SYSTEM HALTED** — Active Reasons: `{', '.join(risk_status.get('reasons', []))}`")
    else:
        bias = (metrics or {}).get("strategy_bias", "CALIBRATING")
        st.success(
            f"**SYSTEM ARMED & STREAMING** | Regime Age: `{_age_seconds(regime)}s` | "
            f"Surface Age: `{_age_seconds(surface)}s` | Active Bias: **{bias}**"
        )


# ==============================================================================
# PAGE 0: EXECUTIVE OVERVIEW (ALL 5 ACTORS SYNTHESIZED)
# ==============================================================================
def page_overview() -> None:
    state = get_live_state()
    regime = state["regime"] or {}
    surface = state["surface"] or {}
    rv = state["rv_primary"] or {}
    metrics = state["metrics"] or {}
    port = state["portfolio"] or {}

    st.title(f"Executive Command Center — {underlying_sym}")
    st.caption(
        "Unified synthesis of all 5 live NautilusTrader actors (`MacroRegimeActor`, `FastVolActor`, "
        "`RealizedVolActor`, `SurfaceRegimeMetricsActor`, and `PortfolioAnalyticsActor`)."
    )
    render_global_banner(state)

    # Row 1: Core Cross-Actor KPIs
    atm_iv = surface.get("atm_iv")
    rv_20d = rv.get("rv_20d")
    vrp = metrics.get("vrp_spread") if metrics.get("vrp_spread") is not None else (
        (atm_iv - rv_20d) if (atm_iv is not None and rv_20d is not None) else None
    )
    ratio = regime.get("ratio")
    arb_free = surface.get("arb_free")
    greeks = port.get("greeks") or {}
    risk_p = port.get("risk") or {}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(
            "ATM IV vs RV20 (VRP)",
            f"{atm_iv:.2%}" if atm_iv is not None else "N/A",
            delta=f"VRP: {vrp:+.2%}" if vrp is not None else None,
            help=(
                "**Source:** `FastVolActor` (ATM IV) minus `RealizedVolActor` (20d Yang-Zhang RV).\n\n"
                "**What it informs:** Positive VRP (> +2%) means options are priced above recent physical volatility, "
                "favoring short-premium spreads. Negative VRP warns realized movement is outpacing implied pricing."
            ),
        )
    with c2:
        st.metric(
            "20d / 5d Realized Vol",
            f"{rv_20d:.2%}" if rv_20d is not None else "N/A",
            delta=f"5d RV: {rv.get('rv_5d', 0):.2%}" if rv.get("rv_5d") is not None else None,
            delta_color="off",
            help=(
                "**Source:** `RealizedVolActor` ($\mathbb{P}$-measure Yang-Zhang estimator).\n\n"
                "**What it informs:** When 5-day RV spikes sharply above 20-day RV, short-term realized gamma risk is accelerating."
            ),
        )
    with c3:
        st.metric(
            "VIX / VIX3M Ratio",
            f"{ratio:.3f}" if ratio is not None else "N/A",
            delta="Contango (Allowed)" if regime.get("short_vol_allowed") else "Backwardation (Halt)",
            delta_color="normal" if regime.get("short_vol_allowed") else "inverse",
            help=(
                "**Source:** `MacroRegimeActor`.\n\n"
                "**What it informs:** Hard term-structure gate. Below `1.00` (contango) permits short-vol entries; "
                "above `1.00` (backwardation) signals acute front-month stress and halts short-vol entries."
            ),
        )
    with c4:
        st.metric(
            "25Δ Skew / Butterfly",
            f"{metrics.get('rr_25d', 0):+.2%}" if metrics.get("rr_25d") is not None else "N/A",
            delta=f"BF25: {metrics.get('bf_25d', 0):+.2%}" if metrics.get("bf_25d") is not None else None,
            delta_color="off",
            help=(
                "**Source:** `SurfaceRegimeMetricsActor`.\n\n"
                "**What it informs:** `RR25` measures put-over-call skew steepness; `BF25` measures tail/wing richness. "
                "Elevated `BF25` favors selling wings (Iron Condors)."
            ),
        )
    with c5:
        st.metric(
            "SVI Fit / Arb-Free",
            "PASS" if arb_free else ("FAIL" if arb_free is False else "N/A"),
            delta=f"RMSE: {surface.get('rmse_vol_pts', 0):.3f} pts" if surface.get("rmse_vol_pts") is not None else None,
            delta_color="off",
            help=(
                "**Source:** `FastVolActor`.\n\n"
                "**What it informs:** Verifies static butterfly and calendar no-arbitrage (`arb_free=True`) and sub-0.25 vol pt "
                "calibration accuracy before pricing or routing spreads."
            ),
        )
    with c6:
        st.metric(
            "Book P&L / ES 99%",
            f"${port.get('total_pnl_usd', 0):+.2f}",
            delta=f"ES99: ${risk_p.get('book_es_99_usd', 0):.2f}",
            delta_color="off",
            help=(
                "**Source:** `PortfolioAnalyticsActor`.\n\n"
                "**What it informs:** Live unrealized + realized P&L across open positions alongside 1-day 99% Expected Shortfall (`ES99`), "
                "which must remain within the 1% ($1,000) sleeve tail-risk budget."
            ),
        )

    st.divider()

    # Row 2: Actor Health & Architecture Matrix
    st.subheader("Live Actor Bus Health & Outputs")
    actor_rows = [
        {
            "Actor": "1. MacroRegimeActor",
            "Role": "Term-Structure & Stress Gate",
            "Age (s)": _age_seconds(regime),
            "Primary Output": (
                f"VIX={regime.get('vix')} | VIX3M={regime.get('vix3m')} | Ratio={regime.get('ratio')} | "
                f"ShortVolAllowed={regime.get('short_vol_allowed')}"
                if regime
                else "Waiting..."
            ),
        },
        {
            "Actor": "2. FastVolActor",
            "Role": "SVI Calibration & Arb Validation",
            "Age (s)": _age_seconds(surface),
            "Primary Output": (
                f"Spot=${surface.get('spot')} | Fwd=${surface.get('forward')} | ATM_IV={surface.get('atm_iv', 0):.2%} | "
                f"RMSE={surface.get('rmse_vol_pts')}pts | ArbFree={surface.get('arb_free')}"
                if surface
                else "Waiting..."
            ),
        },
        {
            "Actor": "3. RealizedVolActor",
            "Role": "P-Measure RV, Jumps & Correlation",
            "Age (s)": _age_seconds(rv),
            "Primary Output": (
                f"RV5={rv.get('rv_5d', 0):.2%} | RV20={rv.get('rv_20d', 0):.2%} | Pct={rv.get('rv_percentile')}% | "
                f"Jump={rv.get('jump_ratio', 0):.1%} | 8-ETF Corr={rv.get('avg_pair_corr')}"
                if rv
                else "Waiting..."
            ),
        },
        {
            "Actor": "4. SurfaceRegimeMetricsActor",
            "Role": "Skew/Butterfly, VRP & Strategy Bias",
            "Age (s)": _age_seconds(metrics),
            "Primary Output": (
                f"RR25={metrics.get('rr_25d', 0):+.2%} | BF25={metrics.get('bf_25d', 0):+.2%} | "
                f"VRP={metrics.get('vrp_spread', 0):+.2%} | Bias={metrics.get('strategy_bias')}"
                if metrics
                else "Waiting..."
            ),
        },
        {
            "Actor": "5. PortfolioAnalyticsActor",
            "Role": "Book Greeks, Stress Grid & Attribution",
            "Age (s)": _age_seconds(port),
            "Primary Output": (
                f"Open={port.get('open_positions', 0)} | BetaDelta={greeks.get('beta_delta_shares', 0):+.1f}sh | "
                f"Vega=${greeks.get('net_vega_1v_usd', 0):+.2f}/v | Theta=${greeks.get('net_theta_day_usd', 0):+.2f}/d"
                if port
                else "Waiting..."
            ),
        },
    ]
    st.dataframe(pd.DataFrame(actor_rows), hide_index=True, use_container_width=True)

    # Row 3: Quick Smile + Universe RV Snapshot
    left_c, right_c = st.columns([1.4, 1.0])
    with left_c:
        if surface and "log_moneyness" in surface and "ivs" in surface:
            k_arr = np.asarray(surface["log_moneyness"], dtype=float)
            svi_iv = np.asarray(surface["ivs"], dtype=float) * 100.0
            raw_iv = np.asarray(surface.get("raw_ivs", surface["ivs"]), dtype=float) * 100.0
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_arr, y=svi_iv, mode="lines", name="SVI Smile", line=dict(color="#00CC96", width=3)))
            fig.add_trace(go.Scatter(x=k_arr, y=raw_iv, mode="markers", name="Market Mid IV", marker=dict(color="#FFA15A", size=8)))
            if rv_20d is not None:
                fig.add_hline(y=rv_20d * 100.0, line_dash="dash", line_color="#EF553B", annotation_text=f"RV20 ({rv_20d:.1%})")
            fig.update_layout(
                title=f"{underlying_sym} Live SVI Smile vs. RV20",
                xaxis_title="Log-Moneyness k = ln(K / F)",
                yaxis_title="Annualized Vol (%)",
                height=340,
                margin=dict(l=15, r=15, t=40, b=15),
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

    with right_c:
        rv_univ = state["rv_universe"] or {}
        if rv_univ:
            u_rows = [
                {
                    "Symbol": sym.split(".")[0],
                    "RV 5d": f"{v.get('rv_5d', 0):.1%}",
                    "RV 20d": f"{v.get('rv_20d', 0):.1%}",
                    "1Y Pct": f"{v.get('rv_percentile', 0):.0f}%",
                    "Jump %": f"{v.get('jump_ratio', 0):.0%}",
                }
                for sym, v in sorted(rv_univ.items(), key=lambda kv: kv[1].get("rv_percentile", 0), reverse=True)
            ]
            st.markdown("**8-ETF Realized Vol Snapshot (`RealizedVolActor`)**")
            st.dataframe(pd.DataFrame(u_rows), hide_index=True, use_container_width=True, height=300)


# ==============================================================================
# PAGE 1: MacroRegimeActor
# ==============================================================================
def page_macro_regime() -> None:
    state = get_live_state()
    regime = state["regime"] or {}
    risk_history = state["risk_history"]

    st.title("1. MacroRegimeActor — Volatility Term Structure & Stress Gate")
    st.markdown(
        "Monitors front-month implied volatility (`VIX` / `VIXY`) against 3-month volatility (`VIX3M` / `VIXM`) "
        "to detect term-structure inversion (backwardation) before short-volatility entries are permitted."
    )
    render_global_banner(state)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "30d Front Volatility (VIX)",
            f"{regime.get('vix', 0):.2f}" if regime.get("vix") is not None else "N/A",
            help=(
                "**What it means:** Spot 30-day implied volatility index level.\n\n"
                "**What it informs:** Measures immediate market demand for 1-month S&P 500 downside protection."
            ),
        )
    with c2:
        st.metric(
            "90d Medium Volatility (VIX3M)",
            f"{regime.get('vix3m', 0):.2f}" if regime.get("vix3m") is not None else "N/A",
            help=(
                "**What it means:** 3-month (93-day) implied volatility index level.\n\n"
                "**What it informs:** Anchors the medium-term volatility expectation; moves slower than spot VIX during acute shocks."
            ),
        )
    with c3:
        ratio = regime.get("ratio")
        st.metric(
            "Term-Structure Ratio (VIX / VIX3M)",
            f"{ratio:.4f}" if ratio is not None else "N/A",
            delta=f"{(ratio - 1.0):+.4f} vs 1.00 Gate" if ratio is not None else None,
            delta_color="inverse",
            help=(
                "**What it means:** Ratio of 30d VIX to 90d VIX3M. `< 1.00` = upward-sloping contango; `> 1.00` = inverted backwardation.\n\n"
                "**What it informs:** Primary macro circuit breaker. Short-vol carry strategies earn structural roll yield in contango "
                "but suffer severe gap losses when the curve inverts into backwardation."
            ),
        )
    with c4:
        allowed = regime.get("short_vol_allowed")
        st.metric(
            "Short-Vol Gate",
            "ALLOWED" if allowed else ("BLOCKED" if allowed is False else "N/A"),
            delta=f"Feed Age: {_age_seconds(regime)}s",
            delta_color="off",
            help=(
                "**What it means:** Boolean authorization flag published by `MacroRegimeActor` (`ratio <= 1.00` and `not is_stale`).\n\n"
                "**What it informs:** Directly gates `VrpIncomeStrategy`; if False, all new short-premium orders are blocked."
            ),
        )

    st.divider()
    if risk_history:
        hist_df = pd.DataFrame(reversed(risk_history))
        if "evaluated_at_utc" in hist_df.columns and "vix_ratio" in hist_df.columns:
            hist_df["time"] = pd.to_datetime(hist_df["evaluated_at_utc"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist_df["time"],
                    y=hist_df["vix_ratio"],
                    mode="lines",
                    name="VIX / VIX3M Ratio",
                    line=dict(color="#FFA15A", width=2.5),
                )
            )
            fig.add_hline(
                y=1.0,
                line_dash="dash",
                line_color="#EF553B",
                annotation_text="Backwardation Halt Threshold (1.00)",
            )
            fig.update_layout(
                title="Live VIX / VIX3M Term-Structure Ratio Trajectory",
                xaxis_title="Time (UTC)",
                yaxis_title="VIX / VIX3M Ratio",
                height=380,
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw `mft:regime:latest` JSON Payload", expanded=False):
        st.json(regime)


# ==============================================================================
# PAGE 2: FastVolActor (SVI SURFACE & CALIBRATION)
# ==============================================================================
def page_fast_vol() -> None:
    state = get_live_state()
    surface = state["surface"] or {}
    rv = state["rv_primary"] or {}
    rv_20d = rv.get("rv_20d")

    st.title(f"2. FastVolActor — SVI Volatility Surface & Calibration ({underlying_sym})")
    st.markdown(
        "Inverts live option mid-quotes in `fp64`, extracts put-call parity forward $F$ and implied dividend yield $q$, "
        "fits an arbitrage-free Gatheral SVI slice, and validates Durrleman butterfly & calendar constraints."
    )
    render_global_banner(state)

    if not surface or "log_moneyness" not in surface:
        st.info(f"Waiting for live SVI snapshot at `{selected_surface_key}`...")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Spot / Implied Forward",
            f"${surface.get('spot', 0):.2f}",
            delta=f"Fwd: ${surface.get('forward', 0):.2f}",
            delta_color="off",
            help=(
                "**What it means:** Underlying spot mid-price and put-call parity implied forward $F = K + e^{rT}(C - P)$.\n\n"
                "**What it informs:** Anchors log-moneyness $k = \\ln(K/F)$ so put and call branches splice seamlessly with zero kink."
            ),
        )
    with c2:
        st.metric(
            "Implied Div Yield (q)",
            f"{surface.get('implied_q', 0):.2%}",
            delta=f"Parity Pairs: {surface.get('parity_pairs', 0)}",
            delta_color="off",
            help=(
                "**What it means:** Continuous dividend yield backed out from ATM put-call parity.\n\n"
                "**What it informs:** Captures expected ex-dividend carry over the option's remaining life ($T$)."
            ),
        )
    with c3:
        st.metric(
            "SVI Fit RMSE",
            f"{surface.get('rmse_vol_pts', 0):.4f} pts",
            delta=f"Max Err: {surface.get('max_resid_pts', 0):.4f} pts",
            delta_color="inverse",
            help=(
                "**What it means:** Root-Mean-Square Error (in vol percentage points) between market mid IVs and the SVI curve.\n\n"
                "**What it informs:** Sub-0.25 vol pt error confirms smooth, high-fidelity calibration across the strike chain."
            ),
        )
    with c4:
        st.metric(
            "Arb-Free Validation",
            "PASS" if surface.get("arb_free") else "FAIL",
            delta=f"Quotes Used: {surface.get('quotes', 0)}",
            delta_color="off",
            help=(
                "**What it means:** Verifies $g(k) \\ge 0$ (no butterfly arbitrage / non-negative risk-neutral density) "
                "and Roger Lee's asymptotic wing slope bound $b(1 + |\\rho|) \\le 2$.\n\n"
                "**What it informs:** Never price or trade spreads off a surface where `arb_free=False`."
            ),
        )
    with c5:
        st.metric(
            "Time to Expiry (T)",
            f"{surface.get('ttm_days', 0):.2f} d",
            delta=f"ATM IV: {surface.get('atm_iv', 0):.2%}",
            delta_color="off",
            help=(
                "**What it means:** Exact calendar days to expiration computed from `OptionContract.expiration_ns`.\n\n"
                "**What it informs:** Scales total variance $w(k) = \\sigma_{\\text{IV}}^2(k) T$ and theta decay rates."
            ),
        )

    k_arr = np.asarray(surface["log_moneyness"], dtype=float)
    svi_iv_arr = np.asarray(surface["ivs"], dtype=float) * 100.0
    raw_iv_arr = np.asarray(surface["raw_ivs"], dtype=float) * 100.0
    resid_pts = raw_iv_arr - svi_iv_arr
    fwd = float(surface.get("forward", surface.get("spot", 700.0)))
    spot_px = float(surface.get("spot", fwd))
    ttm_years = max(float(surface.get("ttm_days", 25.0)) / 365.25, 1e-4)
    approx_strikes = fwd * np.exp(k_arr)
    svi_p = surface.get("svi_params", [])

    left_col, right_col = st.columns([2.2, 1.0])
    with left_col:
        fig_fit = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.68, 0.32],
            subplot_titles=(
                f"{underlying_sym} Implied Volatility Smile: Market Mid IV vs. Fitted SVI",
                "Calibration Residuals (Market Mid IV − SVI Model IV, in Vol Pts)",
            ),
        )
        if len(svi_p) == 5:
            k_dense = np.linspace(float(k_arr.min()) - 0.01, float(k_arr.max()) + 0.01, 120)
            svi_dense_iv = _eval_svi_iv(k_dense, svi_p, ttm_years) * 100.0
            fig_fit.add_trace(
                go.Scatter(
                    x=k_dense,
                    y=svi_dense_iv,
                    mode="lines",
                    name="SVI Fitted Smile",
                    line=dict(color="#00CC96", width=3),
                ),
                row=1,
                col=1,
            )
        fig_fit.add_trace(
            go.Scatter(
                x=k_arr,
                y=raw_iv_arr,
                mode="markers",
                name="Market Mid IV",
                marker=dict(color="#FFA15A", size=9),
                customdata=np.column_stack([approx_strikes, resid_pts]),
                hovertemplate="k=%{x:.4f} (~$%{customdata[0]:.1f})<br>Mkt IV: %{y:.2f}%<br>Resid: %{customdata[1]:+.4f} pts<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if rv_20d is not None:
            fig_fit.add_hline(y=rv_20d * 100.0, line_dash="dash", line_color="#EF553B", row=1, col=1)

        fig_fit.add_trace(
            go.Bar(x=k_arr, y=resid_pts, name="Residual (pts)", marker_color="#636EFA"),
            row=2,
            col=1,
        )
        fig_fit.add_hline(y=0.0, line_color="rgba(255,255,255,0.4)", row=2, col=1)
        fig_fit.update_layout(height=500, margin=dict(l=15, r=15, t=40, b=15), template="plotly_dark")
        st.plotly_chart(fig_fit, use_container_width=True)

    with right_col:
        st.markdown(
            "**Raw SVI Parameters `(a, b, ρ, m, σ)`**",
            help="w(k) = a + b * (ρ*(k - m) + sqrt((k - m)^2 + σ^2)).",
        )
        if len(svi_p) == 5:
            lee_util = (float(svi_p[1]) * (1.0 + abs(float(svi_p[2])))) / 2.0
            st.dataframe(
                pd.DataFrame(
                    {
                        "Parameter": ["a (variance level)", "b (wing slope)", "ρ (skew tilt)", "m (center shift)", "σ (ATM curvature)", "Roger Lee Bound"],
                        "Value": [f"{svi_p[0]:.6f}", f"{svi_p[1]:.6f}", f"{svi_p[2]:.6f}", f"{svi_p[3]:.6f}", f"{svi_p[4]:.6f}", f"{lee_util:.1%}"],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown(
            "**Pre-Solver Hygiene Filter Counts**",
            help="Quotes rejected prior to SVI optimization due to staleness, wide spread, intrinsic violation, deep ITM, or low OI.",
        )
        st.json(surface.get("dropped", {}))

    if len(svi_p) == 5:
        st.subheader("Breeden-Litzenberger Risk-Neutral Implied Density q(K)")
        k_grid = np.linspace(-0.18, 0.14, 240)
        K_grid = fwd * np.exp(k_grid)
        iv_grid = _eval_svi_iv(k_grid, svi_p, ttm_years)
        C_grid = _bs_price(fwd, K_grid, ttm_years, iv_grid, r=0.0425, is_call=True)
        pdf_q = np.maximum(math.exp(0.0425 * ttm_years) * np.gradient(np.gradient(C_grid, K_grid), K_grid), 0.0)
        area = float(np.trapezoid(pdf_q, K_grid)) if hasattr(np, "trapezoid") else float(np.trapz(pdf_q, K_grid))
        if area > 0:
            pdf_q = pdf_q / area

        fig_pdf = go.Figure()
        fig_pdf.add_trace(go.Scatter(x=K_grid, y=pdf_q, mode="lines", fill="tozeroy", name="SVI Density q(K)", line=dict(color="#00CC96")))
        fig_pdf.add_vline(x=spot_px, line_dash="dot", line_color="white", annotation_text=f"Spot ${spot_px:.2f}")
        fig_pdf.update_layout(
            xaxis_title="Terminal Strike Price at Expiry ($)",
            yaxis_title="Risk-Neutral Probability Density",
            height=320,
            margin=dict(l=15, r=15, t=20, b=15),
            template="plotly_dark",
        )
        st.plotly_chart(fig_pdf, use_container_width=True)


# ==============================================================================
# PAGE 3: RealizedVolActor (P-MEASURE ENGINE)
# ==============================================================================
def page_realized_vol() -> None:
    state = get_live_state()
    rv = state["rv_primary"] or {}
    rv_univ = state["rv_universe"] or {}

    st.title("3. RealizedVolActor — P-Measure Volatility, Jumps & Cross-Asset Correlation")
    st.markdown(
        "Warm-starts from 250 historical daily bars in `ParquetDataCatalog` / ThetaData and updates continuously "
        "on live ticks/bars to provide the true physical ($\mathbb{P}$-measure) volatility inputs for `ScenarioPricer`."
    )
    render_global_banner(state)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "5d Yang-Zhang RV",
            f"{rv.get('rv_5d', 0):.2%}" if rv.get("rv_5d") is not None else "N/A",
            help=(
                "**What it means:** Short-window (1-week) annualized Yang-Zhang realized volatility.\n\n"
                "**What it informs:** Rapidly detects sudden intraday range expansion before slower 20d/60d windows react."
            ),
        )
    with c2:
        st.metric(
            "20d Yang-Zhang RV",
            f"{rv.get('rv_20d', 0):.2%}" if rv.get("rv_20d") is not None else "N/A",
            delta=f"10d: {rv.get('rv_10d', 0):.2%}" if rv.get("rv_10d") is not None else None,
            delta_color="off",
            help=(
                "**What it means:** Primary 1-month physical volatility benchmark combining overnight jumps and intraday OHLC range.\n\n"
                "**What it informs:** Directly drives the Monte Carlo path dispersion in `ScenarioPricer`."
            ),
        )
    with c3:
        st.metric(
            "60d Yang-Zhang RV",
            f"{rv.get('rv_60d', 0):.2%}" if rv.get("rv_60d") is not None else "N/A",
            delta=f"1Y Pct: {rv.get('rv_percentile', 0):.1f}%",
            delta_color="off",
            help=(
                "**What it means:** Quarterly realized volatility baseline and 250-day empirical percentile rank.\n\n"
                "**What it informs:** Shows whether current volatility is mean-reverting from an extreme spike (>85th pct)."
            ),
        )
    with c4:
        st.metric(
            "Bipower Jump Share",
            f"{rv.get('jump_ratio', 0):.1%}" if rv.get("jump_ratio") is not None else "N/A",
            help=(
                "**What it means:** Fraction of total realized variance attributable to discrete price jumps rather than continuous diffusion, "
                "estimated via realized variance minus bipower variation $(RV - BV)/RV$.\n\n"
                "**What it informs:** High jump share (>15–20%) warns of gap/event-driven risk where delta-hedging is less effective."
            ),
        )
    with c5:
        st.metric(
            "8-ETF Pairwise Corr",
            f"{rv.get('avg_pair_corr', 0):.2f}" if rv.get("avg_pair_corr") is not None else "N/A",
            help=(
                "**What it means:** Rolling 20-day average pairwise return correlation across `SPY, QQQ, IWM, XBI, GDX, SMH, TLT, GLD`.\n\n"
                "**What it informs:** When cross-asset correlation spikes toward `0.70+`, macro risk-off dominates and cross-ETF diversification disappears."
            ),
        )

    st.divider()
    if rv_univ:
        rows = []
        for sym, d in rv_univ.items():
            rows.append(
                {
                    "Symbol": sym,
                    "Spot / Close": d.get("spot"),
                    "RV 5d (%)": round(float(d.get("rv_5d", 0)) * 100, 2),
                    "RV 10d (%)": round(float(d.get("rv_10d", 0)) * 100, 2),
                    "RV 20d (%)": round(float(d.get("rv_20d", 0)) * 100, 2),
                    "RV 60d (%)": round(float(d.get("rv_60d", 0)) * 100, 2),
                    "1Y Percentile": d.get("rv_percentile"),
                    "Jump Ratio (%)": round(float(d.get("jump_ratio", 0)) * 100, 1),
                }
            )
        df_u = pd.DataFrame(rows).sort_values("1Y Percentile", ascending=False)

        c_tbl, c_cht = st.columns([1.35, 1.0])
        with c_tbl:
            st.subheader("Multi-Horizon Realized Volatility Across Universe")
            st.dataframe(
                df_u,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Spot / Close": st.column_config.NumberColumn(format="$%.2f"),
                    "1Y Percentile": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
                },
            )
        with c_cht:
            fig_u = go.Figure()
            for col_name, color in [("RV 5d (%)", "#FFA15A"), ("RV 20d (%)", "#636EFA"), ("RV 60d (%)", "#00CC96")]:
                fig_u.add_trace(go.Bar(x=df_u["Symbol"], y=df_u[col_name], name=col_name, marker_color=color))
            fig_u.update_layout(
                barmode="group",
                title="Term Structure of Realized Volatility (5d vs 20d vs 60d)",
                yaxis_title="Annualized Vol (%)",
                height=380,
                template="plotly_dark",
            )
            st.plotly_chart(fig_u, use_container_width=True)
    else:
        st.info("Waiting for `mft:rv:universe` payload from `RealizedVolActor`...")


# ==============================================================================
# PAGE 4: SurfaceRegimeMetricsActor (SKEW, BUTTERFLY, VRP & STRATEGY BIAS)
# ==============================================================================
def page_surface_metrics() -> None:
    state = get_live_state()
    m = state["metrics"] or {}
    tech = m.get("technicals") or {}

    st.title(f"4. SurfaceRegimeMetricsActor — Skew, Butterfly, VRP & Strategy Bias ({underlying_sym})")
    st.markdown(
        "Combines 25-delta Risk Reversal (`RR25`), 25-delta Butterfly (`BF25`), live VRP spread (`ATM_IV - RV20`), "
        "and Nautilus native `SMA_20 / SMA_50 / RSI_14` indicators to classify the optimal option structure."
    )
    render_global_banner(state)

    if not m:
        st.info(f"Waiting for `mft:metrics:{underlying_sym}` from `SurfaceRegimeMetricsActor`...")
        return

    st.info(f"**Active Structural Strategy Recommendation:** `{m.get('strategy_bias')}`")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "25Δ Risk Reversal (RR25)",
            f"{m.get('rr_25d', 0):+.2%}",
            delta=f"Session Pct: {m.get('rr_25d_pct', 50):.0f}%",
            delta_color="off",
            help=(
                "**What it means:** `IV(25Δ Put) - IV(25Δ Call)`.\n\n"
                "**What it informs:** Measures downside put skew steepness. Steep positive `RR25` makes put credit spreads rich "
                "relative to call spreads; compressed `RR25` (<20th pct) favors bullish risk reversals."
            ),
        )
    with c2:
        st.metric(
            "25Δ Butterfly (BF25)",
            f"{m.get('bf_25d', 0):+.2%}",
            delta=f"Session Pct: {m.get('bf_25d_pct', 50):.0f}%",
            delta_color="off",
            help=(
                "**What it means:** `0.5 * (IV(25Δ Put) + IV(25Δ Call)) - ATM_IV`.\n\n"
                "**What it informs:** Measures wing curvature / tail premium relative to ATM. Elevated `BF25` favors selling wings "
                "(Iron Condors); flat/negative `BF25` favors buying wings or debit spreads."
            ),
        )
    with c3:
        st.metric(
            "Live VRP Spread",
            f"{m.get('vrp_spread', 0):+.2%}",
            delta=f"ATM {m.get('atm_iv', 0):.2%} - RV20 {m.get('rv_20d', 0):.2%}",
            delta_color="normal" if m.get("vrp_spread", 0) > 0 else "inverse",
            help=(
                "**What it means:** Real-time Volatility Risk Premium (`ATM_IV - RV_20D`).\n\n"
                "**What it informs:** Core edge filter. When `VRP < 0`, realized volatility exceeds implied volatility and "
                "the actor shifts bias to `DEFENSIVE_OR_LONG_CONVEXITY`."
            ),
        )
    with c4:
        st.metric(
            "Trend State (SMA 20/50)",
            tech.get("trend_state", "N/A"),
            delta=f"SMA20 ${tech.get('sma_20', 0):.2f} / SMA50 ${tech.get('sma_50', 0):.2f}",
            delta_color="off",
            help=(
                "**What it means:** Computed via NautilusTrader's native Cython `SimpleMovingAverage(20)` and `SimpleMovingAverage(50)`.\n\n"
                "**What it informs:** Aligns directional credit spread selection (bull put vs. bear call) with underlying trend structure."
            ),
        )
    with c5:
        st.metric(
            "Momentum (RSI 14)",
            f"{tech.get('rsi_14', 50):.1f}",
            help=(
                "**What it means:** 14-period Relative Strength Index maintained by NautilusTrader's native `RelativeStrengthIndex` indicator.\n\n"
                "**What it informs:** Avoids selling naked calls into oversold bounces (`RSI < 30`) or selling puts into overbought blow-offs (`RSI > 75`)."
            ),
        )

    with st.expander("Decision Matrix Rules (`SurfaceRegimeMetricsActor`)", expanded=True):
        rules_df = pd.DataFrame(
            [
                {"Regime Condition": "VIX/VIX3M >= 1.00 OR VRP < -1.5%", "Assigned Bias": "DEFENSIVE_OR_LONG_CONVEXITY", "Action": "Stand down short-vol; favor long convexity / hedges"},
                {"Regime Condition": "VRP >= +2.0% AND BF25 >= +0.5%", "Assigned Bias": "SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD", "Action": "Harvest rich ATM & wing premium via defined-risk spreads"},
                {"Regime Condition": "RR25 Percentile <= 20%", "Assigned Bias": "BULLISH_RISK_REVERSAL_OR_PUT_SPREAD", "Action": "Exploit compressed put skew for upside asymmetry"},
                {"Regime Condition": "Balanced Contango & Moderate VRP", "Assigned Bias": "SELECTIVE_RELATIVE_VALUE_SPREADS", "Action": "Trade only top SurfaceScanner residual dislocations"},
            ]
        )
        st.dataframe(rules_df, hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 5: PortfolioAnalyticsActor (BOOK GREEKS, STRESS GRID & ATTRIBUTION)
# ==============================================================================
def page_portfolio_analytics() -> None:
    state = get_live_state()
    port = state["portfolio"] or {}
    greeks = port.get("greeks") or {}
    risk_p = port.get("risk") or {}
    attr = port.get("attribution") or {}

    st.title("5. PortfolioAnalyticsActor — Live Book Greeks, Stress Matrix & P&L Attribution")
    st.markdown(
        "Aggregates beta-weighted first- and second-order Greeks across all open positions, computes a live 2D `Spot × Vol` "
        "stress matrix, tracks 1-day `ES_99`, and decomposes daily P&L into Delta, Gamma, Vega, and Theta explain."
    )
    render_global_banner(state)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric(
            "Total Book P&L",
            f"${port.get('total_pnl_usd', 0):+.2f}",
            delta=f"Open: {port.get('open_positions', 0)} pos",
            delta_color="off",
            help="Combined realized + mark-to-model unrealized P&L across all open and closed positions.",
        )
    with c2:
        st.metric(
            "Beta-Weighted Delta",
            f"{greeks.get('beta_delta_shares', 0):+.1f} sh",
            delta=f"Raw: {greeks.get('net_delta_shares', 0):+.1f} sh",
            delta_color="off",
            help="Net directional exposure normalized to SPY share-equivalents.",
        )
    with c3:
        st.metric(
            "Net Gamma (1% Move)",
            f"{greeks.get('net_gamma_1pct_shares', 0):+.2f} sh",
            help="Change in SPY-equivalent delta shares for a +1% move in the underlying.",
        )
    with c4:
        st.metric(
            "Net Vega ($ / +1v)",
            f"${greeks.get('net_vega_1v_usd', 0):+.2f}",
            delta=f"Vanna: {greeks.get('net_vanna_1v', 0):+.3f}",
            delta_color="off",
            help="Dollar P&L impact for a +1.00 vol point parallel upward shift in implied volatility.",
        )
    with c5:
        st.metric(
            "Net Theta ($ / Day)",
            f"${greeks.get('net_theta_day_usd', 0):+.2f}",
            delta=f"Charm: {greeks.get('net_charm_day', 0):+.3f}",
            delta_color="off",
            help="Expected daily time-decay carry ($ USD per calendar day) holding spot and IV constant.",
        )
    with c6:
        st.metric(
            "Book 1d ES 99%",
            f"${risk_p.get('book_es_99_usd', 0):.2f}",
            delta=f"Worst 7d Grid: ${risk_p.get('worst_7d_stress_usd', 0):.2f}",
            delta_color="inverse",
            help="Estimated 1-day 99% Expected Shortfall ($ USD) under P-measure realized volatility.",
        )

    st.divider()
    left_g, right_g = st.columns([1.4, 1.0])

    with left_g:
        st.subheader("Live 2D Spot × Vol Book Stress Matrix (7-Day Horizon, $ USD)")
        grid = risk_p.get("stress_grid_usd")
        s_shocks = risk_p.get("spot_shocks_pct", [-10, -5, -2.5, 0, 2.5, 5, 10])
        v_shocks = risk_p.get("vol_shocks_pts", [-5, -2.5, 0, 2.5, 5, 10])
        if grid:
            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=grid,
                    x=[f"{v:+.1f}v" for v in v_shocks],
                    y=[f"{s:+.1f}%" for s in s_shocks],
                    colorscale="RdYlGn",
                    zmid=0.0,
                    text=[[f"${val:+.0f}" for val in row] for row in grid],
                    texttemplate="%{text}",
                    hovertemplate="Spot Shock: %{y}<br>Vol Shock: %{x}<br>Book P&L: %{text}<extra></extra>",
                )
            )
            fig_hm.update_layout(
                xaxis_title="Implied Volatility Shock (Vol Points)",
                yaxis_title="Underlying Spot Shock (%)",
                height=380,
                margin=dict(l=15, r=15, t=20, b=15),
                template="plotly_dark",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    with right_g:
        st.subheader("Daily Greeks P&L Attribution")
        attr_rows = [
            {"Component": "Actual P&L Change", "USD ($)": f"${attr.get('actual_pnl_change_usd', 0):+.2f}"},
            {"Component": "Delta Explain (Δ · ΔS)", "USD ($)": f"${attr.get('delta_pnl_usd', 0):+.2f}"},
            {"Component": "Gamma Explain (½Γ · ΔS²)", "USD ($)": f"${attr.get('gamma_pnl_usd', 0):+.2f}"},
            {"Component": "Vega Explain (𝒱 · Δσ)", "USD ($)": f"${attr.get('vega_pnl_usd', 0):+.2f}"},
            {"Component": "Theta Explain (Θ · Δt)", "USD ($)": f"${attr.get('theta_pnl_usd', 0):+.2f}"},
            {"Component": "Unexplained Residual", "USD ($)": f"${attr.get('residual_pnl_usd', 0):+.2f}"},
        ]
        st.dataframe(pd.DataFrame(attr_rows), hide_index=True, use_container_width=True)

    positions = port.get("positions") or []
    st.subheader("Open Position Marks & Leg Greeks")
    if positions:
        st.dataframe(pd.DataFrame(positions), hide_index=True, use_container_width=True)
    else:
        st.info("No open positions currently in the book (`open_positions = 0`).")



# ==============================================================================
# PAGE 5B: FuturesContextActor (CME FUTURES & FORWARD BASIS VALIDATION)
# ==============================================================================
def page_futures_basis() -> None:
    state = get_live_state()
    fut = state["futures"] or {}
    surf = state["surface"] or {}

    st.title("6. FuturesContextActor — CME Futures & Forward Price Validation")
    st.markdown(
        "Streams front-month CME index futures (`MESZ26.CME`, `MNQZ26.CME`) to benchmark `FastVolActor`'s "
        "option put-call parity forward $F_{\\text{parity}}$ and dividend carry $q_{\\text{parity}}$ against a "
        "model-free futures benchmark."
    )
    render_global_banner(state)

    if not fut:
        st.info("Waiting for `mft:futures:latest` from `FuturesContextActor` (restart `run_paper.py` to activate `MESZ26.CME` & `MNQZ26.CME`)...")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "MES Futures / Scaled Fwd",
            f"{fut.get('mes_quote', 0):.2f}",
            delta=f"SPY-Eq Fwd: ${fut.get('mes_scaled_fwd', 0):.2f} (Divisor: {fut.get('spx_spy_ratio', 10.0):.4f})",
            delta_color="off",
            help=(
                "**What it means:** Front-month Micro E-mini S&P 500 (`MESZ26.CME`) mid-quote and its SPY share-equivalent "
                "forward price (`MES / 10`).\n\n"
                "**What it informs:** Provides a direct, model-free forward benchmark unaffected by American option Early Exercise Premium."
            ),
        )
    with c2:
        st.metric(
            "Parity Fwd vs Futures Fwd",
            f"${fut.get('parity_fwd', 0):.2f}" if fut.get("parity_fwd") else "N/A",
            delta=f"Diff: ${fut.get('parity_vs_fut_diff_usd', 0):+.2f} ({fut.get('parity_vs_fut_diff_bps', 0):+.1f} bps)",
            delta_color="off",
            help=(
                "**What it means:** Compares `FastVolActor`'s put-call parity forward ($F_{\\text{parity}}$) against the "
                "futures-implied forward interpolated to the exact option expiry ($F_{\\text{fut-opt}}$).\n\n"
                "**What it informs:** Divergence > ±15 bps flags option quote staleness, ex-dividend pricing jumps, or box/conversion arbitrage."
            ),
        )
    with c3:
        st.metric(
            "Futures-Implied Carry (q_fut)",
            f"{fut.get('futures_implied_q', 0):.2%}",
            delta=f"vs Parity q: {fut.get('parity_vs_fut_q_diff_bps', 0):+.1f} bps",
            delta_color="off",
            help=(
                "**What it means:** Continuous dividend yield $q_{\\text{fut}} = r - \\frac{1}{T}\\ln(F_{\\text{fut}}/S)$ backed out "
                "from the cash-futures basis.\n\n"
                "**What it informs:** Cross-checks the single-pair put-call parity dividend estimate in `FastVolActor`."
            ),
        )
    with c4:
        st.metric(
            "Cash-Futures Basis",
            f"${fut.get('cash_futures_basis_usd', 0):+.3f}",
            delta=f"{fut.get('cash_futures_basis_pct', 0):+.3%} ({fut.get('mes_ttm_days', 0):.1f}d to expiry)",
            delta_color="off",
            help=(
                "**What it means:** Raw price spread between scaled `MES` futures and `SPY` spot ($F - S$).\n\n"
                "**What it informs:** Reflects net financing cost $(r - q) S T$ into the December CME quarterly expiration."
            ),
        )
    with c5:
        mnq = fut.get("mnq_quote")
        st.metric(
            "MNQ Futures / Hedge Ratio",
            f"{mnq:.2f}" if mnq else "N/A",
            delta=f"1 MES ≈ {fut.get('mes_spy_equiv_shares', 50):.0f} SPY sh",
            delta_color="off",
            help=(
                "**What it means:** Front-month Micro E-mini Nasdaq-100 (`MNQZ26.CME`) quote and the SPY delta-equivalent "
                "notional of 1 `MES` contract (`$5 × S&P 500 ≈ 50 SPY shares`).\n\n"
                "**What it informs:** Enables capital-efficient, 24/5 delta hedging via `MES` instead of tying up stock margin."
            ),
        )

    st.divider()
    comp_rows = [
        {
            "Source": "FastVolActor (Option Put-Call Parity)",
            "Horizon (Days)": surf.get("ttm_days"),
            "Implied Forward ($)": fut.get("parity_fwd"),
            "Implied Carry q (%)": round(float(surf.get("implied_q", 0)) * 100, 3),
            "Notes": f"Estimated from {surf.get('parity_pairs', 0)} ATM SPY option pair(s)",
        },
        {
            "Source": "FuturesContextActor (CME MESZ26.CME)",
            "Horizon (Days)": fut.get("mes_ttm_days"),
            "Implied Forward ($)": fut.get("mes_scaled_fwd"),
            "Implied Carry q (%)": round(float(fut.get("futures_implied_q", 0)) * 100, 3),
            "Notes": f"Model-free CME futures scaled by {fut.get('spx_spy_ratio', 10.0):.5f} ({fut.get('ratio_source', 'DEFAULT_10X')})",
        },
        {
            "Source": "Futures-Implied Forward @ Option Expiry",
            "Horizon (Days)": surf.get("ttm_days"),
            "Implied Forward ($)": fut.get("fut_implied_opt_fwd"),
            "Implied Carry q (%)": round(float(fut.get("futures_implied_q", 0)) * 100, 3),
            "Notes": f"Parity vs. Futures Divergence = {fut.get('parity_vs_fut_diff_usd', 0):+.3f} ({fut.get('parity_vs_fut_diff_bps', 0):+.1f} bps)",
        },
    ]
    st.subheader("Side-by-Side Forward & Implied Dividend Yield Comparison")
    st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    with st.expander("Raw `mft:futures:latest` JSON Payload", expanded=False):
        st.json(fut)


# ==============================================================================
# PAGE 6: RISK DAEMON & AUDIT TRAIL
# ==============================================================================
def page_risk_daemon() -> None:
    state = get_live_state()
    risk_status = state["risk_status"] or {}
    risk_history = state["risk_history"]
    risk_transitions = state["risk_transitions"]

    st.title("6. Out-of-Process Risk Daemon & Audit Trail")
    st.markdown(
        "Independent fail-closed safety supervisor (`src/risk/risk_daemon.py`) monitoring feed staleness, "
        "term-structure backwardation, sleeve drawdown, and manual kill-switch overrides."
    )
    render_global_banner(state)

    if risk_history:
        hist_df = pd.DataFrame(reversed(risk_history))
        if "evaluated_at_utc" in hist_df.columns:
            hist_df["time"] = pd.to_datetime(hist_df["evaluated_at_utc"])
            hist_df["surface_age_sec"] = hist_df["surface_ages_sec"].apply(
                lambda d: d.get(selected_surface_key) if isinstance(d, dict) else None
            )
            fig_risk = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.10,
                subplot_titles=(
                    "Feed Freshness Ages (seconds since last Valkey tick)",
                    "VIX / VIX3M Ratio vs. Backwardation Halt Threshold (1.00)",
                ),
            )
            fig_risk.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["regime_age_sec"], mode="lines", name="Regime Age (s)", line=dict(color="#00CC96")), row=1, col=1)
            fig_risk.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["surface_age_sec"], mode="lines", name="Surface Age (s)", line=dict(color="#636EFA")), row=1, col=1)
            fig_risk.add_hline(y=120.0, line_dash="dash", line_color="#EF553B", row=1, col=1)

            fig_risk.add_trace(go.Scatter(x=hist_df["time"], y=hist_df["vix_ratio"], mode="lines", name="VIX / VIX3M Ratio", line=dict(color="#FFA15A")), row=2, col=1)
            fig_risk.add_hline(y=1.0, line_dash="dash", line_color="#EF553B", row=2, col=1)
            fig_risk.update_layout(height=420, margin=dict(l=15, r=15, t=40, b=15), template="plotly_dark")
            st.plotly_chart(fig_risk, use_container_width=True)

    if risk_transitions:
        st.subheader("State Transition Audit Log (`mft:risk:transitions`)")
        t_rows = [
            {
                "Timestamp (UTC)": t.get("evaluated_at_utc"),
                "Event": t.get("event", "STATE_CHANGE"),
                "Halted": t.get("to_halted", t.get("halted")),
                "VIX Ratio": t.get("vix_ratio"),
                "Regime Age (s)": t.get("regime_age_sec"),
                "Reasons": ", ".join(t.get("reasons", [])) or "None (Healthy)",
            }
            for t in risk_transitions
        ]
        st.dataframe(pd.DataFrame(t_rows), hide_index=True, use_container_width=True)



def page_vrp_strategy() -> None:
    state = get_live_state()
    vrp = state.get("vrp_strat") or {}

    st.title("8. VrpIncomeStrategy — Shadow Signals & Passive Spread Execution")
    st.markdown(
        "Evaluates live OTM credit spreads on the SVI surface using $\\mathbb{P}$-measure realized volatility (`ScenarioPricer`), "
        "enforces all 6 actor & mid-cap hygiene gates (`PreTradeGate`), and slices passive `OptionSpread` limit orders "
        "near mid-price so the engine never crosses the full bid-ask spread."
    )
    render_global_banner(state)

    if not vrp:
        st.info("Waiting for `mft:strategy:vrp:latest` from `VrpIncomeStrategy` (ensure `run_paper.py` is running)...")
        return

    top = vrp.get("top_candidate") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            "Execution Mode",
            vrp.get("mode", "SHADOW_DRY_RUN"),
            delta=f"Action: {vrp.get('action', 'STAND_DOWN')}",
            delta_color="off",
            help="SHADOW_DRY_RUN logs full candidate scoring & passive clip plans without sending broker orders.",
        )
    with c2:
        st.metric(
            "Top Candidate Combo",
            top.get("combo", "None"),
            delta=f"Mid: ${top.get('mid_credit', 0):.2f} vs Nat: ${top.get('nat_credit', 0):.2f}",
            delta_color="off",
            help="Highest net-EV spread candidate scanned from the live SVI slice.",
        )
    with c3:
        st.metric(
            "Net EV / Lot (P-Measure)",
            f"${top.get('net_ev_per_lot', 0):+.2f}" if top else "N/A",
            delta=f"PoP: {top.get('pop_pct', 0):.1f}%" if top else None,
            delta_color="normal",
            help="Expected profit per contract under physical P-measure RV20 after IBKR commissions & half-spread friction.",
        )
    with c4:
        st.metric(
            "Tail ES_99 / Lot",
            f"${top.get('es99_per_lot', 0):.0f}" if top else "N/A",
            delta=f"Approved Qty: {top.get('approved_qty', 0)} lots" if top else None,
            delta_color="off",
            help="99% Expected Shortfall tail loss per contract against the $1,000 sleeve tail-risk cap.",
        )
    with c5:
        reasons = top.get("gate_reasons") or []
        st.metric(
            "PreTradeGate Status",
            "APPROVED" if top.get("gate_approved") else "BLOCKED",
            delta=", ".join(reasons[:2]) if reasons else "All 6 Actors Green",
            delta_color="off",
            help="Exact reason codes from PreTradeGate (e.g. SURFACE_BIAS_DEFENSIVE when RV20 > ATM_IV).",
        )

    st.divider()
    if vrp.get("passive_clips"):
        st.subheader("Passive OptionSpread Child Clips (Anti-Spread-Crossing Plan)")
        st.dataframe(pd.DataFrame(vrp["passive_clips"]), hide_index=True, use_container_width=True)

    if vrp.get("all_candidates"):
        st.subheader("Live Candidate Spread Ranking & Gate Audit")
        st.dataframe(pd.DataFrame(vrp["all_candidates"]), hide_index=True, use_container_width=True)

    with st.expander("Raw `mft:strategy:vrp:latest` JSON Payload", expanded=False):
        st.json(vrp)


# --- Multi-Page Navigation ---
pg = st.navigation(
    [
        st.Page(page_overview, title="Executive Overview", icon="🧭", default=True),
        st.Page(page_macro_regime, title="1. MacroRegimeActor", icon="🌐"),
        st.Page(page_fast_vol, title="2. FastVolActor (SVI)", icon="📈"),
        st.Page(page_realized_vol, title="3. RealizedVolActor", icon="📊"),
        st.Page(page_surface_metrics, title="4. SurfaceMetricsActor", icon="🎯"),
        st.Page(page_portfolio_analytics, title="5. PortfolioAnalyticsActor", icon="💼"),
        st.Page(page_futures_basis, title="6. Futures & Forward Basis", icon="📐"),
        st.Page(page_risk_daemon, title="7. Risk Daemon & Audit", icon="🛡️"),
        st.Page(page_vrp_strategy, title="8. VrpIncomeStrategy", icon="🎯"),
    ]
)
pg.run()

if auto_refresh:
    time.sleep(5)
    st.rerun()
