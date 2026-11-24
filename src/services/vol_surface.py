"""Full-hygiene IV inversion and spread-weighted, warm-started SVI calibration of one expiry slice."""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import max_error

from src.common import keys
from src.common.constants import DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE, DEFAULT_VALKEY_URL, NS_PER_SEC, NS_PER_YEAR
from src.common.log import LogLike
from src.common.market import BarObs, OptionSpec, QuoteObs, symbol_of
from src.common.store import JsonStore, NullStore
from src.common.timeutil import years_until
from src.quant.black_scholes import black_vega
from src.quant.parity import estimate_forward_and_q
from src.quant.quote_hygiene import SliceFilter, SliceQuote, select_slice_quotes
from src.quant.surface_metrics import smile_wings
from src.quant.svi import fit_svi_slice

MIN_CLEAN_QUOTES = 5
IV_BOUNDS = (0.01, 3.0)
STANDARD_MULTIPLIER = 100

# iv_solver(prices, S, K, t, r, flags) -> implied vols; S is the dividend-discounted spot S * exp(-qT)
IvSolver = Callable[..., np.ndarray]
# arb_validator(log_moneyness, ttm, ivs) -> True when the slice is free of static arbitrage
ArbValidator = Callable[[np.ndarray, float, np.ndarray], bool]
# svi_fitter(log_m, ivs, weights, ttm, x0_warm) -> (fitted ivs, params, rmse in vol units)
SviFitter = Callable[..., tuple[np.ndarray, np.ndarray, float]]


def fast_vollib_iv(prices, S, K, t, r, flags) -> np.ndarray:
    from fast_vollib import fast_implied_volatility  # Lazy: tests inject their own solver

    return np.asarray(
        fast_implied_volatility(price=prices, S=S, K=K, t=t, r=r, flag=flags, return_as="numpy"),
        dtype=np.float64,
    )


def fast_vollib_arb_free(log_m: np.ndarray, ttm: float, ivs: np.ndarray) -> bool:
    """Static-arbitrage check of the fitted slice (a near-copy second slice one day later stands in for term)."""
    from fast_vollib.surface import IVSurface  # Lazy: tests inject their own validator

    t_grid = np.array([ttm, ttm + (1.0 / DAYS_PER_YEAR)])
    iv_grid = np.column_stack([ivs, ivs * 1.0001])
    return bool(IVSurface.from_logmoneyness(log_m, t_grid, iv_grid).validate().passed)


@dataclass(frozen=True)
class VolSurfaceSettings:
    underlying_id: str
    option_ids: tuple[str, ...] = ()
    strikes: tuple[float, ...] = ()
    is_calls: tuple[bool, ...] = ()
    ttm_years: float = 25.0 / DAYS_PER_YEAR
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    dividend_yield: float = 0.0125
    interval_secs: int = 15
    strike_min: float = 685.0
    strike_max: float = 755.0
    max_rel_spread: float = 0.25
    min_mid_price: float = 0.15
    min_open_interest: int = 0
    atm_overlap_pct: float = 0.006
    max_quote_age_secs: float = 90.0
    valkey_url: str = DEFAULT_VALKEY_URL


@dataclass(frozen=True)
class OptionSlot:
    instrument_id: str
    strike: float
    is_call: bool
    open_interest: float = 0.0   # 0 = unknown


def select_slice_options(
    specs: Iterable[OptionSpec], underlying_id: str, strike_min: float, strike_max: float
) -> tuple[list[OptionSlot], int | None]:
    """Standard 100-multiplier options of the underlying inside the strike range, ordered by (strike, kind).

    Returns the slots and the expiry of the first one (None when nothing qualifies).
    """
    symbol = symbol_of(underlying_id)
    chosen = [
        spec
        for spec in specs
        if symbol in spec.instrument_id
        and int(spec.multiplier) == STANDARD_MULTIPLIER  # Reject non-standard post-split/M&A contracts
        and strike_min <= spec.strike <= strike_max
    ]
    chosen.sort(key=lambda spec: (spec.strike, "CALL" if spec.is_call else "PUT"))
    slots = [OptionSlot(s.instrument_id, s.strike, s.is_call, float(s.open_interest or 0.0)) for s in chosen]
    return slots, (int(chosen[0].expiry_ns) if chosen else None)


class SliceQuoteBook:
    """Underlying mid and latest (bid, ask, mid, ts_ns) per option of one expiry slice."""

    def __init__(self, underlying_id: str, slots: Sequence[OptionSlot] = ()) -> None:
        self.underlying_id = underlying_id
        self.spot_mid: float | None = None
        self.spot_ts_ns: int = 0
        self.quotes: dict[str, tuple[float, float, float, int]] = {}
        self.set_slots(slots)

    def set_slots(self, slots: Sequence[OptionSlot]) -> None:
        self.slots = list(slots)
        self._index = {slot.instrument_id: i for i, slot in enumerate(self.slots)}

    @property
    def option_ids(self) -> list[str]:
        return [slot.instrument_id for slot in self.slots]

    def on_quote(self, quote: QuoteObs) -> None:
        mid = quote.mid
        if mid is None:
            return
        if quote.instrument_id == self.underlying_id:
            self.spot_mid = mid
            self.spot_ts_ns = quote.ts_ns
        elif quote.instrument_id in self._index:
            self.quotes[quote.instrument_id] = (quote.bid, quote.ask, mid, quote.ts_ns)

    def on_bar(self, bar: BarObs) -> None:
        px = bar.close
        if px <= 0.0:
            return
        if bar.instrument_id == self.underlying_id:
            self.spot_mid = px
            self.spot_ts_ns = bar.ts_ns
        elif bar.instrument_id in self._index:
            # Bar closes stand in for quotes with a synthetic 1% wide market
            self.quotes[bar.instrument_id] = (px * 0.995, px * 1.005, px, bar.ts_ns)

    def slice_quotes(self) -> list[SliceQuote]:
        """Quoted options in slot order."""
        out: list[SliceQuote] = []
        for slot in self.slots:
            quote = self.quotes.get(slot.instrument_id)
            if quote is None:
                continue
            bid, ask, mid, ts_ns = quote
            out.append(SliceQuote(slot.instrument_id, slot.strike, slot.is_call, bid, ask, mid, ts_ns, slot.open_interest))
        return out


def aggregate_by_log_moneyness(log_m: np.ndarray, ivs: np.ndarray, weights: np.ndarray):
    """Spread-weighted IV per distinct log-moneyness (an ATM call and put can share a strike).

    Returns (grid, weighted ivs, summed weights, inverse index mapping each input onto the grid).
    """
    grid, inv_idx = np.unique(np.round(log_m, 6), return_inverse=True)
    agg_w_iv = np.zeros_like(grid)
    agg_w = np.zeros_like(grid)
    np.add.at(agg_w_iv, inv_idx, ivs * weights)
    np.add.at(agg_w, inv_idx, weights)
    return grid, agg_w_iv / agg_w, agg_w, inv_idx


def leg_row(quote: SliceQuote, raw_iv: float, svi_iv: float, forward: float, ttm: float, discount: float) -> dict[str, Any]:
    """Per-leg quote and model residual row for the relative-value scanner and the strategy."""
    # Black vega at the model vol converts the vol residual into a dollar edge versus the half-spread
    vega = black_vega(forward, quote.strike, ttm, svi_iv, discount)
    dollar_edge = abs(raw_iv - svi_iv) * vega
    half_spread = max(0.5 * (quote.ask - quote.bid), 0.005)
    return {
        "instrument_id": quote.instrument_id,
        "strike": round(quote.strike, 2),
        "right": "C" if quote.is_call else "P",
        "bid": round(quote.bid, 4),
        "ask": round(quote.ask, 4),
        "mid": round(quote.mid, 4),
        "raw_iv": round(raw_iv, 4),
        "svi_iv": round(svi_iv, 4),
        "residual_pts": round((raw_iv - svi_iv) * 100.0, 4),
        "vega": round(vega, 4),
        "edge_to_half_spread": round(dollar_edge / half_spread, 3),
        "open_interest": int(quote.open_interest) if quote.open_interest > 0 else None,
    }


@dataclass(frozen=True)
class SurfaceFit:
    """One calibrated slice: the SviSliceParams fields plus the published payload."""

    underlying_id: str
    expiry_ns: int
    forward: float
    atm_iv: float
    skew_25d: float
    arb_free: bool
    quote_count: int
    payload: dict[str, Any]


class VolSurfaceService:
    def __init__(
        self,
        settings: VolSurfaceSettings,
        store: JsonStore | None = None,
        log: LogLike | None = None,
        *,
        iv_solver: IvSolver | None = None,
        arb_validator: ArbValidator | None = None,
        svi_fitter: SviFitter | None = None,
    ) -> None:
        self.settings = settings
        self.store = store if store is not None else NullStore()
        self.log = log if log is not None else logging.getLogger(__name__)
        self.iv_solver: IvSolver = iv_solver or fast_vollib_iv
        self.arb_validator: ArbValidator = arb_validator or fast_vollib_arb_free
        self.svi_fitter: SviFitter = svi_fitter or fit_svi_slice
        slots = [
            OptionSlot(str(iid), float(k), bool(c))
            for iid, k, c in zip(settings.option_ids, settings.strikes, settings.is_calls)
        ]
        self.book = SliceQuoteBook(settings.underlying_id, slots)
        self.expiry_ns: int | None = None
        self.prev_svi_params: np.ndarray | None = None

    def configure(self, specs: Iterable[OptionSpec]) -> int:
        """Replaces the configured slice with the qualifying cached options (if any); returns the slice size."""
        s = self.settings
        slots, expiry_ns = select_slice_options(specs, s.underlying_id, s.strike_min, s.strike_max)
        if slots:
            self.book.set_slots(slots)
            self.expiry_ns = expiry_ns
        return len(self.book.slots)

    def on_quote(self, quote: QuoteObs) -> None:
        self.book.on_quote(quote)

    def on_bar(self, bar: BarObs) -> None:
        self.book.on_bar(bar)

    def current_ttm_years(self, now_ns: int) -> float:
        if self.expiry_ns is not None and self.expiry_ns > now_ns:
            return years_until(self.expiry_ns, now_ns, floor=1e-4)
        return self.settings.ttm_years

    def slice_filter(self) -> SliceFilter:
        s = self.settings
        return SliceFilter(
            max_age_secs=s.max_quote_age_secs,
            min_mid=s.min_mid_price,
            max_rel_spread=s.max_rel_spread,
            min_open_interest=s.min_open_interest,
            atm_overlap_pct=s.atm_overlap_pct,
        )

    def evaluate(self, event_ts_ns: int, wall_ns: int) -> SurfaceFit | None:
        s = self.settings
        now_ns = event_ts_ns
        spot = self.book.spot_mid
        spot_age_secs = (now_ns - self.book.spot_ts_ns) / NS_PER_SEC if self.book.spot_ts_ns > 0 else 9999.0
        if spot is None or (s.max_quote_age_secs > 0 and spot_age_secs > s.max_quote_age_secs):
            self.log.warning(f"FastVolActor underlying stale/missing (spot={spot}, age={spot_age_secs:.1f}s)")
            return None

        T = self.current_ttm_years(now_ns)
        r = s.risk_free_rate
        quotes = self.book.slice_quotes()
        forward, q_eff, parity_pairs = estimate_forward_and_q(
            quotes, spot, T, r, s.dividend_yield, now_ns,
            max_age_secs=s.max_quote_age_secs, min_mid=s.min_mid_price, max_rel_spread=s.max_rel_spread,
        )
        disc = math.exp(-r * T)

        selection = select_slice_quotes(quotes, forward, disc, now_ns, self.slice_filter())
        if len(selection.quotes) < MIN_CLEAN_QUOTES:
            self.log.warning(f"FastVolActor insufficient clean quotes ({len(selection.quotes)}): {selection.dropped}")
            return None

        prices = np.asarray([qt.mid for qt in selection.quotes], dtype=np.float64)
        weights = 1.0 / (np.asarray(selection.spreads, dtype=np.float64) ** 2)
        K = np.asarray([qt.strike for qt in selection.quotes], dtype=np.float64)
        flags = np.asarray(["c" if qt.is_call else "p" for qt in selection.quotes])
        # Dividend-adjusted inversion: discounted spot S*exp(-qT) with the true rate r
        S_eff = np.full_like(K, spot * math.exp(-q_eff * T))
        T_arr = np.full_like(K, T)
        r_arr = np.full_like(K, r)

        ivs = np.asarray(self.iv_solver(prices, S_eff, K, T_arr, r_arr, flags), dtype=np.float64)
        mask = np.isfinite(ivs) & (ivs > IV_BOUNDS[0]) & (ivs < IV_BOUNDS[1])
        quote_count = int(np.count_nonzero(mask))
        if quote_count < MIN_CLEAN_QUOTES:
            return None

        clean_idx = np.flatnonzero(mask)
        K_clean = K[mask]
        iv_clean = ivs[mask]
        log_m = np.log(K_clean / forward)
        log_m_grid, iv_raw_1d, agg_w, inv_idx = aggregate_by_log_moneyness(log_m, iv_clean, weights[mask])
        if len(log_m_grid) < MIN_CLEAN_QUOTES:
            return None

        iv_svi_1d, svi_params, rmse_vol = self.svi_fitter(
            log_m_grid, iv_raw_1d, agg_w, T, x0_warm=self.prev_svi_params
        )

        svi_by_clean_idx = iv_svi_1d[inv_idx]
        legs_table = [
            leg_row(selection.quotes[j], float(iv_clean[pos]), float(svi_by_clean_idx[pos]), forward, T, disc)
            for pos, j in enumerate(clean_idx)
        ]

        arb_free = bool(self.arb_validator(log_m_grid, T, iv_svi_1d))
        if arb_free:
            self.prev_svi_params = np.array(svi_params, dtype=np.float64)

        atm_iv = float(np.interp(0.0, log_m_grid, iv_svi_1d))
        # Same 25-delta convention as the surface metrics (SVI smile at +/- k_25)
        skew_25d = smile_wings(atm_iv, T, svi_params).rr_25d
        max_resid_pts = max_error(iv_raw_1d, iv_svi_1d) * 100.0
        expiry_ns = self.expiry_ns or (now_ns + int(T * NS_PER_YEAR))

        payload = {
            "underlying": str(s.underlying_id),
            "spot": round(spot, 2),
            "spot_age_secs": round(spot_age_secs, 2),
            "forward": round(float(forward), 2),
            "parity_pairs": parity_pairs,
            "implied_q": round(float(q_eff), 4),
            "ttm_days": round(T * DAYS_PER_YEAR, 2),
            "expiry_ns": int(expiry_ns),
            "atm_iv": round(atm_iv, 4),
            "skew_25d": round(skew_25d, 4),
            "arb_free": arb_free,
            "rmse_vol_pts": round(rmse_vol * 100.0, 4),
            "max_resid_pts": round(max_resid_pts, 4),
            "svi_params": [round(float(x), 6) for x in svi_params],
            "quotes": quote_count,
            "dropped": selection.dropped,
            "log_moneyness": log_m_grid.round(4).tolist(),
            "ivs": iv_svi_1d.round(4).tolist(),
            "raw_ivs": iv_raw_1d.round(4).tolist(),
            "legs": legs_table,
            "ts_ns": wall_ns,
            "event_ts_ns": now_ns,
        }
        self.store.set_json(keys.surface_key(s.underlying_id), payload)
        self.log.info(
            f"Surface {s.underlying_id} S={spot:.2f} F={forward:.2f} "
            f"(pairs={parity_pairs}, q={q_eff:.2%}) T={T * DAYS_PER_YEAR:.1f}d ATM_IV={atm_iv:.2%} "
            f"RMSE={rmse_vol * 100:.3f}pts arb_free={arb_free}"
        )
        return SurfaceFit(
            underlying_id=str(s.underlying_id),
            expiry_ns=int(expiry_ns),
            forward=float(forward),
            atm_iv=atm_iv,
            skew_25d=skew_25d,
            arb_free=arb_free,
            quote_count=quote_count,
            payload=payload,
        )
