"""Deterministic payloads and objects shared by the tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common import keys
from src.common.constants import NS_PER_DAY, NS_PER_SEC
from src.common.store import JsonStore


def utc_ns(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> int:
    """Exact epoch nanoseconds of a UTC wall time."""
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=second, tz="UTC").value


# Tuesday 2026-11-24 15:00 UTC = 10:00 New York (regular session, standard time)
NOW_NS = utc_ns(2026, 11, 24, 15, 0, 0)
DAY_NS = NS_PER_DAY
UNDERLYING = "SPY.ARCA"

# A realistic SPY smile: ATM IV about 16% at 30 days, negative skew, Lee bound well inside 2
SVI_PARAMS = (0.0008, 0.02, -0.7, 0.01, 0.06)


def ago(secs: float, now_ns: int = NOW_NS) -> int:
    return now_ns - int(secs * NS_PER_SEC)


def daily_ohlc_frame(
    n: int = 80,
    seed: int = 7,
    start: float = 100.0,
    daily_vol: float = 0.01,
    first_ts_ns: int | None = None,
) -> pd.DataFrame:
    """Seeded random-walk daily OHLC rows (ts_ns one day apart, 16:00 New York) with high >= open/close >= low."""
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(0.0, daily_vol, n)))
    prev_closes = np.concatenate([[start], closes[:-1]])
    opens = prev_closes * np.exp(rng.normal(0.0, daily_vol / 4.0, n))
    highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.0, daily_vol, n))
    lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.0, daily_vol, n))
    first = utc_ns(2026, 6, 1, 20, 0, 0) if first_ts_ns is None else first_ts_ns
    ts = first + np.arange(n, dtype=np.int64) * DAY_NS
    return pd.DataFrame(
        {"ts_ns": ts, "open": opens, "high": highs, "low": lows, "close": closes, "volume": 1_000.0}
    )


def surface_payload(underlying: str = UNDERLYING, now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "underlying": underlying,
        "spot": 700.0,
        "spot_age_secs": 1.0,
        "forward": 701.0,
        "parity_pairs": 1,
        "implied_q": 0.0125,
        "ttm_days": 24.0,
        "expiry_ns": now_ns + 24 * DAY_NS,
        "atm_iv": 0.16,
        "skew_25d": 0.02,
        "arb_free": True,
        "rmse_vol_pts": 0.1,
        "max_resid_pts": 0.2,
        "svi_params": list(SVI_PARAMS),
        "quotes": 12,
        "dropped": {"stale": 0, "spread": 0, "intrinsic": 0, "itm": 0, "oi": 0},
        "log_moneyness": [-0.05, 0.0, 0.05],
        "ivs": [0.18, 0.16, 0.15],
        "raw_ivs": [0.181, 0.16, 0.149],
        "legs": [],
        "ts_ns": now_ns,
        "event_ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def regime_payload(now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "vix": 15.0,
        "vix3m": 17.0,
        "ratio": 0.8824,
        "short_vol_allowed": True,
        "is_stale": False,
        "max_age_secs": 5.0,
        "ts_ns": now_ns,
        "event_ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def metrics_payload(underlying: str = UNDERLYING, now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "underlying": underlying,
        "spot": 700.0,
        "atm_iv": 0.16,
        "rr_25d": 0.03,
        "bf_25d": 0.006,
        "vrp_spread": 0.03,
        "strategy_bias": "SHORT_VRP_IRON_CONDOR_OR_CREDIT_SPREAD",
        "ts_ns": now_ns,
        "event_ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def rv_payload(symbol: str = UNDERLYING, now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "symbol": symbol,
        "spot": 700.0,
        "rv_5d": 0.11,
        "rv_10d": 0.12,
        "rv_20d": 0.13,
        "rv_60d": 0.14,
        "rv_percentile": 40.0,
        "jump_ratio": 0.05,
        "avg_pair_corr": 0.3,
        "ts_ns": now_ns,
        "event_ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def futures_payload(now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "cross_check_valid": True,
        "parity_vs_fut_diff_bps": 2.0,
        "ratio_source": "LIVE_SPX_SPY_1MIN_BARS",
        "ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def portfolio_payload(now_ns: int = NOW_NS, book_es_99_usd: float = 0.0, **overrides: Any) -> dict[str, Any]:
    payload = {"risk": {"book_es_99_usd": book_es_99_usd}, "open_positions": 0, "ts_ns": now_ns}
    payload.update(overrides)
    return payload


def positions_payload(now_ns: int = NOW_NS, **overrides: Any) -> dict[str, Any]:
    payload = {
        "allocated_capital_usd": 100_000.0,
        "equity": 100_000.0,
        "high_water_mark": 100_000.0,
        "drawdown_pct": 0.0,
        "realized_total_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "open_positions": 0,
        "ts_ns": now_ns,
    }
    payload.update(overrides)
    return payload


def risk_status_payload(now_ns: int = NOW_NS, halted: bool = False, **overrides: Any) -> dict[str, Any]:
    payload = {"halted": halted, "reasons": [], "reason_codes": [], "evaluated_at_ns": now_ns}
    payload.update(overrides)
    return payload


def seed_healthy_inputs(store: JsonStore, underlying: str = UNDERLYING, now_ns: int = NOW_NS) -> JsonStore:
    """Writes every snapshot the gate, daemon and strategy need, all fresh and permissive."""
    store.set_text(keys.RISK_HALTED, "0")
    store.set_json(keys.RISK_STATUS, risk_status_payload(now_ns))
    store.set_json(keys.REGIME_LATEST, regime_payload(now_ns))
    store.set_json(keys.surface_key(underlying), surface_payload(underlying, now_ns))
    store.set_json(keys.metrics_key(underlying), metrics_payload(underlying, now_ns))
    store.set_json(keys.rv_key(underlying), rv_payload(underlying, now_ns))
    store.set_json(keys.FUTURES_LATEST, futures_payload(now_ns))
    store.set_json(keys.PORTFOLIO_RISK, portfolio_payload(now_ns))
    store.set_json(keys.POSITIONS_LATEST, positions_payload(now_ns))
    return store


def surface_leg(
    instrument_id: str,
    strike: float,
    right: str,
    bid: float,
    ask: float,
    **overrides: Any,
) -> dict[str, Any]:
    """One row of a surface payload's "legs" table."""
    leg = {
        "instrument_id": instrument_id,
        "strike": strike,
        "right": right,
        "bid": bid,
        "ask": ask,
        "mid": round(0.5 * (bid + ask), 4),
        "raw_iv": 0.16,
        "svi_iv": 0.16,
        "residual_pts": 0.0,
        "vega": 0.5,
        "edge_to_half_spread": 0.0,
        "open_interest": 1_000,
    }
    leg.update(overrides)
    return leg


def risk_assessment(**overrides: Any) -> Any:
    """ScenarioPricer RiskAssessment with plausible per-lot values for a small credit spread."""
    from src.risk.scenario_pricer import RiskAssessment

    fields = {
        "combo_name": "SPY 690/685P Bull Put",
        "strategy_type": "BULL_PUT_SPREAD",
        "horizon_days": 14.0,
        "realized_vol_p": 0.13,
        "entry_credit_usd": 60.0,
        "friction_cost_usd": 21.3,
        "net_ev_usd": 20.0,
        "prob_profit": 0.7,
        "var_95_usd": -80.0,
        "es_95_usd": -110.0,
        "var_99_usd": -130.0,
        "es_99_usd": -150.0,
        "max_margin_est_usd": 440.0,
        "ev_to_es99_ratio": 0.1333,
        "recommended_contracts": 6,
        "approved": True,
        "spot_shocks_pct": [],
        "vol_shocks_pts": [],
        "scenario_grid_usd": [],
    }
    fields.update(overrides)
    return RiskAssessment(**fields)
