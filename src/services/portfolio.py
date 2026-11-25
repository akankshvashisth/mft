"""Real-time beta-weighted book Greeks, spot x vol stress, VaR/ES, P&L attribution and sleeve drawdown."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.common import keys
from src.common.constants import DEFAULT_RISK_FREE_RATE, DEFAULT_VALKEY_URL, NS_PER_SEC
from src.common.log import LogLike
from src.common.market import OptionSpec, QuoteObs, symbol_of
from src.common.store import JsonStore, NullStore
from src.common.timeutil import ny_date
from src.quant.risk_measures import parametric_book_var_es, portfolio_horizon_var_es
from src.services.portfolio_marks import (
    GREEK_KEYS,
    SPOT_SHOCKS_PCT,
    VOL_SHOCKS_PTS,
    book_stress_grid,
    linear_stress_leg,
    option_exposure,
)
from src.services.portfolio_tracking import AttributionTracker, SleeveTracker

OPTION = "option"
OPTION_SPREAD = "option_spread"
OTHER = "other"


@dataclass(frozen=True)
class PortfolioSettings:
    benchmark_id: str
    allocated_capital_usd: float = 100_000.0
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    interval_secs: int = 15
    max_mark_quote_age_secs: float = 300.0
    valkey_url: str = DEFAULT_VALKEY_URL


@dataclass(frozen=True)
class PositionView:
    instrument_id: str
    signed_qty: float
    avg_px_open: float
    realized_pnl: float = 0.0


@dataclass(frozen=True)
class InstrumentInfo:
    kind: str                   # OPTION, OPTION_SPREAD or OTHER
    multiplier: float
    option: OptionSpec | None = None
    spread_legs: tuple[tuple[str, int], ...] | None = None  # (instrument_id, ratio) pairs


class BookMarket(Protocol):
    """Instrument and quote lookups the book needs (backed by the Nautilus cache in production)."""

    def instrument(self, instrument_id: str) -> InstrumentInfo | None: ...

    def quote(self, instrument_id: str) -> QuoteObs | None: ...

    def instruments_for_symbol(self, symbol: str) -> list[str]:
        """Ids of non-option instruments whose symbol is `symbol` (e.g. the underlying ETF)."""
        ...


class EmptyBookMarket:
    def instrument(self, instrument_id: str) -> InstrumentInfo | None:
        return None

    def quote(self, instrument_id: str) -> QuoteObs | None:
        return None

    def instruments_for_symbol(self, symbol: str) -> list[str]:
        return []


def _symbol_spot(symbol: str, surfaces: dict[str, dict], rv_by_symbol: dict[str, dict]) -> float | None:
    val = (surfaces.get(symbol) or {}).get("spot") or (rv_by_symbol.get(symbol) or {}).get("spot")
    return float(val) if val else None


class PortfolioService:
    def __init__(
        self,
        settings: PortfolioSettings,
        store: JsonStore | None = None,
        log: LogLike | None = None,
        market: BookMarket | None = None,
    ) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        self.market: BookMarket = market if market is not None else EmptyBookMarket()
        self.attribution = AttributionTracker()
        self.sleeve = SleeveTracker(settings.allocated_capital_usd)

    def restore(self) -> None:
        """Carries realized P&L and the high-water mark over from the last published sleeve snapshot."""
        self.sleeve.restore(self.store.get_json(keys.POSITIONS_LATEST))

    # --- Market lookups --------------------------------------------------------------------------

    def live_mid(self, instrument_id: str, now_ns: int) -> float | None:
        quote = self.market.quote(instrument_id)
        if quote is None or quote.mid is None:
            return None
        if (now_ns - quote.ts_ns) / NS_PER_SEC > self.settings.max_mark_quote_age_secs:
            return None
        return quote.mid

    def surfaces_by_symbol(self) -> dict[str, dict]:
        """Latest SVI surface payload per underlying symbol (e.g. "SPY"), whatever its venue."""
        return {
            symbol_of(keys.instrument_from_surface_key(key)): payload
            for key, payload in self.store.scan_json(keys.SURFACE_PATTERN).items()
        }

    def underlying_spot(
        self, symbol: str, surfaces: dict[str, dict], rv_by_symbol: dict[str, dict], now_ns: int
    ) -> float | None:
        spot = _symbol_spot(symbol, surfaces, rv_by_symbol)
        if spot:
            return spot
        for instrument_id in self.market.instruments_for_symbol(symbol):
            mid = self.live_mid(instrument_id, now_ns)
            if mid is not None:
                return mid
        return None

    def spread_option_legs(self, spread_id: str) -> list[tuple[InstrumentInfo, int]] | None:
        """Leg (instrument, ratio) pairs: prefer info.spread_legs, fall back to mft:spread_legs:<id>."""
        # First check InstrumentInfo for embedded legs (from OptionSpread.info["legs"])
        info = self.market.instrument(spread_id)
        if info is not None and info.spread_legs:
            legs: list[tuple[InstrumentInfo, int]] = []
            for iid, ratio in info.spread_legs:
                leg_info = self.market.instrument(iid)
                if leg_info is None or leg_info.kind != OPTION or leg_info.option is None:
                    return None
                legs.append((leg_info, ratio))
            return legs or None
        # Fall back to Valkey-stored spread legs
        raw_legs = self.store.get_json_value(keys.spread_legs_key(spread_id))
        if not isinstance(raw_legs, list):
            return None
        legs: list[tuple[InstrumentInfo, int]] = []
        try:
            for leg in raw_legs:
                info = self.market.instrument(str(leg["instrument_id"]))
                if info is None or info.kind != OPTION or info.option is None:
                    return None
                legs.append((info, int(leg["ratio"])))
        except (KeyError, TypeError, ValueError):
            return None
        return legs or None

    # --- Evaluation ------------------------------------------------------------------------------

    def _exposure(
        self,
        info: InstrumentInfo,
        qty: float,
        now_ns: int,
        surfaces: dict[str, dict],
        rv_by_symbol: dict[str, dict],
        bench_spot: float,
        bench_atm_iv: float,
    ) -> dict[str, Any] | None:
        option = info.option
        und_symbol = symbol_of(option.underlying)
        spot = self.underlying_spot(und_symbol, surfaces, rv_by_symbol, now_ns)
        if spot is None or spot <= 0.0:
            return None
        return option_exposure(
            option,
            info.multiplier,
            qty,
            now_ns,
            spot=spot,
            surface=surfaces.get(und_symbol) or {},
            rv=rv_by_symbol.get(und_symbol) or {},
            bench_spot=bench_spot,
            bench_atm_iv=bench_atm_iv,
            r=self.settings.risk_free_rate,
            live_mid=self.live_mid(option.instrument_id, now_ns),
        )

    def evaluate(
        self,
        positions: Sequence[PositionView],
        closed_realized_usd: float,
        wall_ns: int,
    ) -> dict[str, Any]:
        """Marks every open position, aggregates exposures and risk, and publishes the book and sleeve."""
        now_ns = wall_ns
        bench_id = str(self.settings.benchmark_id)
        bench_symbol = symbol_of(bench_id)
        surfaces = self.surfaces_by_symbol()
        surf = surfaces.get(bench_symbol) or {}
        rv_univ = self.store.get_json(keys.RV_UNIVERSE)
        rv_by_symbol = {symbol_of(k): v for k, v in rv_univ.items() if isinstance(v, dict)}
        bench_rv = rv_univ.get(bench_id) or self.store.get_json(keys.rv_key(bench_id))

        bench_spot = float(surf.get("spot") or bench_rv.get("spot") or 0.0)
        bench_atm_iv = float(surf.get("atm_iv") or bench_rv.get("rv_20d") or 0.20)

        totals = dict.fromkeys(GREEK_KEYS, 0.0)
        unrealized_pnl_usd = 0.0
        session_realized_usd = closed_realized_usd + sum(float(p.realized_pnl) for p in positions)
        legs_spec: list[dict] = []
        position_rows: list[dict] = []
        unmodeled_positions: list[str] = []

        for pos in positions:
            iid = str(pos.instrument_id)
            info = self.market.instrument(iid)
            signed_qty = float(pos.signed_qty)
            avg_px = float(pos.avg_px_open)
            pos_greeks = dict.fromkeys(GREEK_KEYS, 0.0)

            option_legs: list[tuple[InstrumentInfo, int]] | None = None
            if info is not None and info.kind == OPTION and info.option is not None:
                option_legs = [(info, 1)]
            elif info is not None:
                option_legs = self.spread_option_legs(iid)

            if option_legs is not None:
                exposures: list[tuple[int, dict]] | None = []
                for leg_info, ratio in option_legs:
                    exp = self._exposure(
                        leg_info, signed_qty * ratio, now_ns, surfaces, rv_by_symbol, bench_spot, bench_atm_iv
                    )
                    if exp is None:
                        exposures = None
                        break
                    exposures.append((ratio, exp))

                if exposures is None:
                    mark_px, mark_source, pos_pnl = avg_px, "UNPRICED_UNDERLYING", 0.0
                else:
                    # Spread price = sum(ratio x leg price); per-share like the legs, so use the leg multiplier
                    mult = exposures[0][1]["multiplier"]
                    mark_px = sum(ratio * exp["mark"] for ratio, exp in exposures)
                    sources = {exp["mark_source"] for _, exp in exposures}
                    mark_source = sources.pop() if len(sources) == 1 else "MIXED"
                    pos_pnl = signed_qty * mult * (mark_px - avg_px)
                    for _, exp in exposures:
                        for key in GREEK_KEYS:
                            pos_greeks[key] += exp["greeks"][key]
                        legs_spec.append(exp["stress_leg"])
            elif info is not None and info.kind == OPTION_SPREAD:
                # Spread without registered legs: no reliable model, never mark it at the underlying's price
                live_mid = self.live_mid(iid, now_ns)
                mark_px = live_mid if live_mid is not None else avg_px
                mark_source = "UNMODELED_SPREAD"
                pos_pnl = signed_qty * info.multiplier * (mark_px - avg_px)
                unmodeled_positions.append(iid)
            else:
                mult = info.multiplier if info is not None else 1.0
                mark_px = self.live_mid(iid, now_ns)
                mark_source = "LIVE_MID"
                if mark_px is None:
                    mark_px = _symbol_spot(symbol_of(iid), surfaces, rv_by_symbol)
                    mark_source = "ACTOR_SPOT"
                if mark_px is None or mark_px <= 0.0:
                    mark_px, mark_source = avg_px, "ENTRY_PRICE"
                pos_pnl = signed_qty * mult * (mark_px - avg_px)
                pos_greeks["delta"] = signed_qty * mult
                # Notional-weighted SPY share equivalents (e.g. 1 MES ~ 5 x index / SPY spot)
                pos_greeks["beta_delta"] = (
                    signed_qty * mult * mark_px / bench_spot if bench_spot > 0 else signed_qty * mult
                )
                legs_spec.append(linear_stress_leg(signed_qty, mult, mark_px, underlying=iid))

            unrealized_pnl_usd += pos_pnl
            for key in GREEK_KEYS:
                totals[key] += pos_greeks[key]
            position_rows.append(
                {
                    "instrument": iid,
                    "qty": signed_qty,
                    "avg_px": round(avg_px, 3),
                    "mark_px": round(mark_px, 3),
                    "mark_source": mark_source,
                    "pnl_usd": round(pos_pnl, 2),
                    "delta_sh": round(pos_greeks["delta"], 2),
                    "vega_1v": round(pos_greeks["vega_1v"], 2),
                    "theta_day": round(pos_greeks["theta_day"], 2),
                }
            )

        net_delta_shares = totals["delta"]
        beta_delta_shares = totals["beta_delta"]
        net_gamma_1pct = totals["gamma_1pct"]
        net_vega_1v = totals["vega_1v"]
        net_theta_day = totals["theta_day"]
        realized_pnl_usd = self.sleeve.realized_carry_usd + session_realized_usd
        total_pnl_usd = realized_pnl_usd + unrealized_pnl_usd

        book_key = tuple(sorted((str(p.instrument_id), round(float(p.signed_qty), 6)) for p in positions))
        attribution = self.attribution.update(
            ny_date(now_ns).isoformat(),
            book_key,
            bench_spot,
            bench_atm_iv,
            total_pnl_usd,
            {
                "delta_sh": net_delta_shares,
                "gamma_1pct": net_gamma_1pct,
                "vega_1v": net_vega_1v,
                "theta_day": net_theta_day,
            },
            now_ns,
        )

        stress_grid = book_stress_grid(legs_spec, self.settings.risk_free_rate)
        worst_stress_usd = float(np.min(stress_grid))

        # 14-day nonlinear portfolio Monte Carlo ES99 (same horizon as ScenarioPricer)
        book_var_99_usd, book_es_99_usd = portfolio_horizon_var_es(
            legs_spec,
            horizon_days=14.0,
            r=self.settings.risk_free_rate,
            n_scenarios=5_000,
            seed=42,
        )
        # Keep 1-day linear parametric as a secondary diagnostic
        rv_20d = float(bench_rv.get("rv_20d") or bench_atm_iv)
        book_1d_linear_var_99_usd, book_1d_linear_es_99_usd = parametric_book_var_es(beta_delta_shares, net_vega_1v, bench_spot, rv_20d)

        # Sleeve equity is decoupled from IBKR NLV: allocated capital + this book's P&L
        sleeve = self.sleeve.snapshot(realized_pnl_usd, unrealized_pnl_usd, len(positions), now_ns)

        payload = {
            "benchmark": bench_id,
            "benchmark_spot": round(bench_spot, 2),
            "open_positions": len(positions),
            "realized_pnl_usd": round(realized_pnl_usd, 2),
            "unrealized_pnl_usd": round(unrealized_pnl_usd, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "greeks": {
                "net_delta_shares": round(net_delta_shares, 2),
                "beta_delta_shares": round(beta_delta_shares, 2),
                "net_gamma_1pct_shares": round(net_gamma_1pct, 3),
                "net_vega_1v_usd": round(net_vega_1v, 2),
                "net_theta_day_usd": round(net_theta_day, 2),
                "net_vanna_1v": round(totals["vanna"], 3),
                "net_charm_day": round(totals["charm_day"], 3),
            },
            "risk": {
                "book_var_99_usd": book_var_99_usd,
                "book_es_99_usd": book_es_99_usd,
                "book_1d_linear_var_99_usd": book_1d_linear_var_99_usd,
                "book_1d_linear_es_99_usd": book_1d_linear_es_99_usd,
                "unmodeled_positions": unmodeled_positions,
                "has_unmodeled_positions": bool(unmodeled_positions),
                "worst_7d_stress_usd": round(worst_stress_usd, 2),
                "spot_shocks_pct": list(SPOT_SHOCKS_PCT),
                "vol_shocks_pts": list(VOL_SHOCKS_PTS),
                "stress_grid_usd": stress_grid,
            },
            "attribution": attribution,
            "sleeve": sleeve,
            "positions": position_rows,
            "ts_ns": now_ns,
        }

        batch = self.store.batch()
        batch.set_json(keys.PORTFOLIO_RISK, payload)
        # Drawdown input for the out-of-process risk daemon
        batch.set_json(keys.POSITIONS_LATEST, sleeve)
        batch.execute()

        self.log.info(
            f"PortfolioBook open={len(positions)} PnL=${total_pnl_usd:+.2f} DD={sleeve['drawdown_pct']:+.2f}% "
            f"BetaDelta={beta_delta_shares:+.1f}sh Vega=${net_vega_1v:+.2f}/v "
            f"Theta=${net_theta_day:+.2f}/d ES99=${book_es_99_usd:.2f}"
        )
        return payload
