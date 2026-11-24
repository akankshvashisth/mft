"""CME micro-futures carry cross-check of the SVI parity forward (MES vs SPY, via a live SPX/SPY ratio)."""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from src.common import keys
from src.common.constants import DEFAULT_RISK_FREE_RATE, DEFAULT_VALKEY_URL, NS_PER_SEC
from src.common.freshness import is_fresh
from src.common.log import LogLike
from src.common.market import BarObs, LatestValue, QuoteObs, TradeObs
from src.common.numeric import is_number, round_or_none
from src.common.store import JsonStore, NullStore
from src.common.timeutil import days_to_years, days_until

RATIO_SOURCE_LIVE = "LIVE_SPX_SPY_1MIN_BARS"
RATIO_SOURCE_NONE = "UNAVAILABLE"
MES_INDEX_MULTIPLIER = 5.0   # 1 MES contract = $5 x SPX


@dataclass(frozen=True)
class FuturesSettings:
    spy_id: str
    mes_id: str
    spx_id: str | None = None
    mnq_id: str | None = None
    qqq_id: str | None = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    interval_secs: int = 15
    max_quote_age_secs: float = 90.0
    max_surface_age_secs: float = 120.0
    max_ratio_age_secs: float = 900.0
    valkey_url: str = DEFAULT_VALKEY_URL


class SameMinuteRatio:
    """numerator / denominator from 1-minute closes stamped at the same bar boundary (timing-consistent)."""

    def __init__(self, numerator_id: str, denominator_id: str, max_bars_kept: int = 30) -> None:
        self.numerator_id = numerator_id
        self.denominator_id = denominator_id
        self.max_bars_kept = max_bars_kept
        self._closes: dict[str, OrderedDict[int, float]] = {numerator_id: OrderedDict(), denominator_id: OrderedDict()}
        self.ratio: float | None = None
        self.bar_ts_ns: int = 0
        self.wall_ns: int = 0   # Wall time the current ratio was computed

    def on_close(self, instrument_id: str, close: float, ts_ns: int, wall_ns: int) -> bool:
        """Records a close; True when it completed a newer same-minute pair (ratio updated)."""
        closes = self._closes.get(instrument_id)
        if closes is None or close <= 0.0:
            return False
        closes[ts_ns] = close
        while len(closes) > self.max_bars_kept:
            closes.popitem(last=False)

        num = self._closes[self.numerator_id].get(ts_ns)
        den = self._closes[self.denominator_id].get(ts_ns)
        if num and den and ts_ns >= self.bar_ts_ns:
            self.ratio = num / den
            self.bar_ts_ns = ts_ns
            self.wall_ns = wall_ns
            return True
        return False

    def age_secs(self, wall_ns: int) -> float | None:
        return (wall_ns - self.wall_ns) / NS_PER_SEC if self.ratio is not None else None


@dataclass(frozen=True)
class FuturesBasis:
    mes_scaled_fwd: float | None = None
    basis_usd: float | None = None
    basis_pct: float | None = None
    q_fut: float | None = None
    fut_implied_opt_fwd: float | None = None
    fwd_diff_usd: float | None = None
    fwd_diff_bps: float | None = None
    q_diff_bps: float | None = None


def compute_futures_basis(
    spy_spot: float,
    mes_px: float,
    ratio: float | None,
    mes_ttm_days: float | None,
    r: float,
    parity_fwd: float = 0.0,
    parity_spot: float = 0.0,
    parity_q: Any = None,
    opt_ttm_days: float = 0.0,
) -> FuturesBasis:
    """Futures-implied carry and its comparison with the option-parity forward.

    Everything is None without a ratio or a futures expiry; the parity comparison also needs a surface
    forward, spot and maturity.
    """
    if ratio is None or mes_ttm_days is None:
        return FuturesBasis()
    mes_scaled_fwd = mes_px / ratio
    basis_usd = mes_scaled_fwd - spy_spot
    basis_pct = basis_usd / spy_spot
    # MES and SPY quotes are synchronous, so q_fut is a clean carry estimate
    q_fut = r - math.log(max(mes_scaled_fwd / spy_spot, 1e-6)) / days_to_years(mes_ttm_days)

    fut_implied_opt_fwd = fwd_diff_usd = fwd_diff_bps = q_diff_bps = None
    if parity_fwd > 0.0 and parity_spot > 0.0 and opt_ttm_days > 0.0:
        # Re-project from the surface's own spot so only carry (not spot drift since the fit) is compared
        fut_implied_opt_fwd = parity_spot * math.exp((r - q_fut) * days_to_years(opt_ttm_days))
        fwd_diff_usd = parity_fwd - fut_implied_opt_fwd
        fwd_diff_bps = (fwd_diff_usd / parity_spot) * 10_000.0
        if is_number(parity_q):
            q_diff_bps = (float(parity_q) - q_fut) * 10_000.0

    return FuturesBasis(
        mes_scaled_fwd=mes_scaled_fwd,
        basis_usd=basis_usd,
        basis_pct=basis_pct,
        q_fut=q_fut,
        fut_implied_opt_fwd=fut_implied_opt_fwd,
        fwd_diff_usd=fwd_diff_usd,
        fwd_diff_bps=fwd_diff_bps,
        q_diff_bps=q_diff_bps,
    )


class FuturesBasisService:
    """Fresh MES/MNQ/SPY mids, the SPX/SPY ratio and the published mft:futures:latest cross-check.

    Without a live ratio the cross-check is published as unavailable rather than anchored to the parity
    forward it is supposed to validate.
    """

    def __init__(self, settings: FuturesSettings, store: JsonStore | None = None, log: LogLike | None = None) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        self.mids: dict[str, LatestValue] = {}
        self.index_ratio = SameMinuteRatio(settings.spx_id, settings.spy_id) if settings.spx_id else None

    def quote_ids(self) -> list[str]:
        """Instruments whose quotes feed this service (cached-instrument filtering is the shell's job)."""
        s = self.settings
        return [iid for iid in (s.spy_id, s.mes_id, s.mnq_id, s.qqq_id) if iid is not None]

    def _set_mid(self, instrument_id: str, price: float, ts_ns: int) -> None:
        self.mids.setdefault(instrument_id, LatestValue()).update(price, ts_ns)

    def on_quote(self, quote: QuoteObs) -> None:
        mid = quote.mid
        if mid is not None:
            self._set_mid(quote.instrument_id, mid, quote.ts_ns)

    def on_trade(self, trade: TradeObs) -> None:
        if trade.price > 0.0:
            self._set_mid(trade.instrument_id, trade.price, trade.ts_ns)

    def on_bar(self, bar: BarObs, wall_ns: int) -> None:
        s = self.settings
        if self.index_ratio is None or bar.instrument_id not in (s.spx_id, s.spy_id) or bar.close <= 0.0:
            return
        if bar.instrument_id == s.spx_id:
            self._set_mid(bar.instrument_id, bar.close, bar.ts_ns)
        self.index_ratio.on_close(bar.instrument_id, bar.close, bar.ts_ns, wall_ns)

    def fresh_mid(self, instrument_id: str | None, now_ns: int) -> float | None:
        if instrument_id is None or instrument_id not in self.mids:
            return None
        return self.mids[instrument_id].fresh(now_ns, self.settings.max_quote_age_secs)

    def live_ratio(self, wall_ns: int) -> tuple[float | None, float | None, str]:
        """(ratio if fresh enough, its age, source label)."""
        if self.index_ratio is None:
            return None, None, RATIO_SOURCE_NONE
        age = self.index_ratio.age_secs(wall_ns)
        if age is not None and age <= self.settings.max_ratio_age_secs:
            return self.index_ratio.ratio, age, RATIO_SOURCE_LIVE
        return None, age, RATIO_SOURCE_NONE

    def evaluate(self, wall_ns: int, event_ts_ns: int, mes_expiry_ns: int = 0) -> dict[str, Any] | None:
        s = self.settings
        r = s.risk_free_rate
        surf = self.store.get_json(keys.surface_key(s.spy_id))
        surf_fresh = is_fresh(surf, wall_ns, s.max_surface_age_secs)

        spy_spot = self.fresh_mid(s.spy_id, wall_ns) or 0.0
        mes_px = self.fresh_mid(s.mes_id, wall_ns) or 0.0
        mnq_px = self.fresh_mid(s.mnq_id, wall_ns)
        spx_px = self.mids[s.spx_id].value if s.spx_id in self.mids else None

        if spy_spot <= 0.0 or mes_px <= 0.0:
            self.log.info(f"FuturesContext waiting for fresh quotes (SPY={spy_spot:.2f}, MES={mes_px:.2f})")
            return None

        ratio, ratio_age_secs, ratio_source = self.live_ratio(wall_ns)
        ttm_fut_days = max(days_until(mes_expiry_ns, wall_ns), 0.1) if mes_expiry_ns > wall_ns else None

        parity_fwd = float(surf.get("forward") or 0.0) if surf_fresh else 0.0
        parity_spot = float(surf.get("spot") or 0.0) if surf_fresh else 0.0
        parity_q = surf.get("implied_q") if surf_fresh else None
        ttm_opt_days = float(surf.get("ttm_days") or 0.0) if surf_fresh else 0.0

        basis = compute_futures_basis(
            spy_spot, mes_px, ratio, ttm_fut_days, r, parity_fwd, parity_spot, parity_q, ttm_opt_days
        )

        payload = {
            "mes_symbol": str(s.mes_id),
            "mnq_symbol": str(s.mnq_id) if s.mnq_id else None,
            "spx_symbol": str(s.spx_id) if s.spx_id else None,
            "spy_spot": round(spy_spot, 3),
            "spx_quote": round_or_none(spx_px, 2),
            "spx_spy_ratio": round_or_none(ratio, 5),
            "ratio_source": ratio_source,
            "ratio_age_secs": round_or_none(ratio_age_secs, 1),
            "mes_quote": round(mes_px, 2),
            "mnq_quote": round_or_none(mnq_px, 2),
            "mes_ttm_days": round_or_none(ttm_fut_days, 2),
            "mes_scaled_fwd": round_or_none(basis.mes_scaled_fwd, 3),
            "cash_futures_basis_usd": round_or_none(basis.basis_usd, 3),
            "cash_futures_basis_pct": round_or_none(basis.basis_pct, 6),
            "futures_implied_q": round_or_none(basis.q_fut, 6),
            "parity_fwd": round(parity_fwd, 3) if parity_fwd > 0 else None,
            "fut_implied_opt_fwd": round_or_none(basis.fut_implied_opt_fwd, 3),
            "parity_vs_fut_diff_usd": round_or_none(basis.fwd_diff_usd, 3),
            "parity_vs_fut_diff_bps": round_or_none(basis.fwd_diff_bps, 2),
            "parity_vs_fut_q_diff_bps": round_or_none(basis.q_diff_bps, 2),
            "cross_check_valid": basis.fwd_diff_bps is not None,
            # 1 MES contract ($5 x SPX) in SPY share-equivalents
            "mes_spy_equiv_shares": round(MES_INDEX_MULTIPLIER * ratio, 2) if ratio is not None else None,
            "ts_ns": wall_ns,
            "event_ts_ns": event_ts_ns,
        }
        self.store.set_json(keys.FUTURES_LATEST, payload)

        if basis.fwd_diff_bps is not None:
            self.log.info(
                f"FuturesBasis MES={mes_px:.2f} (Ratio={ratio:.4f} [{ratio_source}], "
                f"ScaledF={basis.mes_scaled_fwd:.2f}, q_fut={basis.q_fut:.2%}) vs ParityF={parity_fwd:.2f} | "
                f"Diff={basis.fwd_diff_usd:+.2f} ({basis.fwd_diff_bps:+.1f}bps)"
            )
        else:
            self.log.info(
                f"FuturesBasis MES={mes_px:.2f} cross-check unavailable "
                f"(ratio={ratio_source}, surface_fresh={surf_fresh}, mes_ttm_days={ttm_fut_days})"
            )
        return payload
