import time
from datetime import timedelta
import json
import numpy as np
import redis
from scipy.optimize import least_squares
from fast_vollib import fast_implied_volatility
from fast_vollib.surface import IVSurface
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import (
    Bar,
    CustomData,
    DataType,
    InstrumentId,
    OptionContract,
    QuoteTick,
)
from src.models.vol_surface import SviSliceParams


def _svi_total_var(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = params
    dk = k - m
    return a + b * (rho * dk + np.sqrt(dk * dk + sigma * sigma))


def _fit_svi_slice(
    log_m: np.ndarray,
    ivs: np.ndarray,
    weights: np.ndarray,
    ttm: float,
    x0_warm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fits spread-weighted Raw SVI total variance with warm-start + cold-start fallback."""
    w_mkt = (ivs ** 2) * ttm
    w_atm = float(np.interp(0.0, log_m, w_mkt))
    w_norm = np.sqrt(weights / max(float(np.mean(weights)), 1e-12))

    lb = np.array([-0.5, 1e-4, -0.999, -0.5, 5e-3], dtype=np.float64)
    ub = np.array([2.0, 2.0, 0.999, 0.5, 2.0], dtype=np.float64)

    def residuals(p: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = p
        # Soft penalty enforcing Roger Lee wing bound b*(1 + |rho|) <= 2 & positive min variance
        wing_excess = max(0.0, b * (1.0 + abs(rho)) - 1.98)
        min_var = a + b * sigma * np.sqrt(max(1.0 - rho * rho, 1e-6))
        var_deficit = max(0.0, 1e-6 - min_var)
        res = w_norm * (_svi_total_var(log_m, p) - w_mkt)
        return np.append(res, [10.0 * wing_excess, 10.0 * var_deficit])

    cold_x0 = np.array([max(w_atm * 0.85, 1e-5), 0.08, -0.55, 0.0, 0.08], dtype=np.float64)
    start_x0 = np.clip(x0_warm, lb + 1e-6, ub - 1e-6) if x0_warm is not None else cold_x0

    res = least_squares(residuals, x0=start_x0, bounds=(lb, ub), max_nfev=120)
    w_fit = np.maximum(_svi_total_var(log_m, res.x), 1e-6)
    iv_fit = np.sqrt(w_fit / ttm)
    rmse_vol = float(np.sqrt(np.mean((iv_fit - ivs) ** 2)))

    # If warm-start drifted into a poor fit (> 0.5 vol pts RMSE), re-run from cold start
    if x0_warm is not None and rmse_vol > 0.005:
        res_cold = least_squares(residuals, x0=cold_x0, bounds=(lb, ub), max_nfev=150)
        w_cold = np.maximum(_svi_total_var(log_m, res_cold.x), 1e-6)
        iv_cold = np.sqrt(w_cold / ttm)
        rmse_cold = float(np.sqrt(np.mean((iv_cold - ivs) ** 2)))
        if rmse_cold < rmse_vol:
            return iv_cold, res_cold.x, rmse_cold

    return iv_fit, res.x, rmse_vol


class FastVolActorConfig(DataActorConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        option_ids: tuple[InstrumentId, ...] = (),
        strikes: tuple[float, ...] = (),
        is_calls: tuple[bool, ...] = (),
        ttm_years: float = 25.0 / 365.25,
        risk_free_rate: float = 0.0425,
        dividend_yield: float = 0.0125,
        interval_secs: int = 15,
        strike_min: float = 685.0,
        strike_max: float = 755.0,
        max_rel_spread: float = 0.25,
        min_mid_price: float = 0.15,
        min_open_interest: int = 0,
        atm_overlap_pct: float = 0.006,
        max_quote_age_secs: float = 90.0,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.option_ids = option_ids
        self.strikes = strikes
        self.is_calls = is_calls
        self.ttm_years = ttm_years
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.interval_secs = interval_secs
        self.strike_min = strike_min
        self.strike_max = strike_max
        self.max_rel_spread = max_rel_spread
        self.min_mid_price = min_mid_price
        self.min_open_interest = min_open_interest
        self.atm_overlap_pct = atm_overlap_pct
        self.max_quote_age_secs = max_quote_age_secs
        self.valkey_url = valkey_url


class FastVolActor(DataActor):
    """Full-hygiene IV inversion + spread-weighted warm-start SVI surface calibrator."""

    def __init__(self, config: FastVolActorConfig) -> None:
        super().__init__(config)
        self.active_option_ids: list[InstrumentId] = list(config.option_ids)
        self.strikes = np.asarray(config.strikes, dtype=np.float64)
        self.is_calls = np.asarray(config.is_calls, dtype=bool)
        self.open_interests = np.zeros_like(self.strikes, dtype=np.float64)
        self.expiry_ns: int | None = None
        self.spot_mid: float | None = None
        self.spot_ts_ns: int = 0
        # Store (bid, ask, mid, ts_event_ns) per option instrument
        self.opt_quotes: dict[InstrumentId, tuple[float, float, float, int]] = {}
        self._idx_map = {oid: i for i, oid in enumerate(self.active_option_ids)}
        self._prev_svi_params: np.ndarray | None = None
        self._prev_valid_payload: dict | None = None
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None

    def on_start(self) -> None:
        self.subscribe_quotes(self.config.underlying_id)

        all_cached = list(self.cache.instruments())
        cached_opts = [
            inst
            for inst in all_cached
            if isinstance(inst, OptionContract)
            and self.config.underlying_id.symbol.value in str(inst.id)
            and int(inst.multiplier) == 100  # Reject non-standard post-split/M&A contracts
            and self.config.strike_min <= float(inst.strike_price) <= self.config.strike_max
        ]
        if cached_opts:
            cached_opts.sort(key=lambda c: (float(c.strike_price), str(c.option_kind)))
            self.active_option_ids = [inst.id for inst in cached_opts]
            self.strikes = np.asarray(
                [float(inst.strike_price) for inst in cached_opts], dtype=np.float64
            )
            self.is_calls = np.asarray(
                ["CALL" in str(inst.option_kind).upper() for inst in cached_opts],
                dtype=bool,
            )
            self.open_interests = np.asarray(
                [
                    float((inst.info or {}).get("open_interest", 0.0))
                    if getattr(inst, "info", None)
                    else 0.0
                    for inst in cached_opts
                ],
                dtype=np.float64,
            )
            self.expiry_ns = int(cached_opts[0].expiration_ns)
            self._idx_map = {oid: i for i, oid in enumerate(self.active_option_ids)}

        for oid in self.active_option_ids:
            self.subscribe_quotes(oid)

        self.clock.set_timer(
            "fast_vol_fit",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(
            f"FastVolActor watching {self.config.underlying_id} + "
            f"{len(self.active_option_ids)} standard 100-mult options"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return
        mid = 0.5 * (bid + ask)
        ts_ns = int(tick.ts_event)
        if tick.instrument_id == self.config.underlying_id:
            self.spot_mid = mid
            self.spot_ts_ns = ts_ns
        elif tick.instrument_id in self._idx_map:
            self.opt_quotes[tick.instrument_id] = (bid, ask, mid, ts_ns)

    def on_bar(self, bar: Bar) -> None:
        px = float(bar.close)
        if px <= 0.0:
            return
        iid = bar.bar_type.instrument_id
        ts_ns = int(bar.ts_event)
        if iid == self.config.underlying_id:
            self.spot_mid = px
            self.spot_ts_ns = ts_ns
        elif iid in self._idx_map:
            self.opt_quotes[iid] = (px * 0.995, px * 1.005, px, ts_ns)

    def _current_ttm_years(self, now_ns: int) -> float:
        if self.expiry_ns is not None and self.expiry_ns > now_ns:
            return max((self.expiry_ns - now_ns) / (365.25 * 86400.0 * 1e9), 1e-4)
        return self.config.ttm_years

    def _estimate_forward_and_q(
        self, spot: float, T: float, now_ns: int
    ) -> tuple[float, float, int]:
        """Solves median forward F and implied dividend yield q across all sane near-ATM pairs."""
        r = self.config.risk_free_rate
        q_default = self.config.dividend_yield
        f_prior = spot * np.exp((r - q_default) * T)
        max_age_ns = int(self.config.max_quote_age_secs * 1e9)

        calls_by_k: dict[float, float] = {}
        puts_by_k: dict[float, float] = {}
        for i, oid in enumerate(self.active_option_ids):
            q_tuple = self.opt_quotes.get(oid)
            if q_tuple is None:
                continue
            bid, ask, mid, ts_ns = q_tuple
            if max_age_ns > 0 and (now_ns - ts_ns) > max_age_ns:
                continue
            if mid < self.config.min_mid_price or (ask - bid) / mid > self.config.max_rel_spread:
                continue
            k = float(self.strikes[i])
            if abs(k - spot) / spot > 0.025:
                continue
            if self.is_calls[i]:
                calls_by_k[k] = mid
            else:
                puts_by_k[k] = mid

        common_strikes = sorted(set(calls_by_k).intersection(puts_by_k))
        if not common_strikes:
            return float(f_prior), float(q_default), 0

        f_estimates = [
            k + np.exp(r * T) * (calls_by_k[k] - puts_by_k[k]) for k in common_strikes
        ]
        f_median = float(np.median(f_estimates))
        q_implied = float(r - np.log(max(f_median, 1e-6) / spot) / T)
        if -0.05 <= q_implied <= 0.08:
            return f_median, q_implied, len(common_strikes)
        return float(f_prior), float(q_default), 0

    def on_timer(self, event) -> None:
        now_ns = int(event.ts_event)
        max_age_ns = int(self.config.max_quote_age_secs * 1e9)
        spot_age_secs = (now_ns - self.spot_ts_ns) / 1e9 if self.spot_ts_ns > 0 else 9999.0

        if self.spot_mid is None or (max_age_ns > 0 and spot_age_secs > self.config.max_quote_age_secs):
            self.log.warning(
                f"FastVolActor underlying stale/missing (spot={self.spot_mid}, age={spot_age_secs:.1f}s)"
            )
            return

        T = self._current_ttm_years(now_ns)
        r = self.config.risk_free_rate
        forward, q_eff, parity_pairs = self._estimate_forward_and_q(self.spot_mid, T, now_ns)
        r_carry = r - q_eff
        disc = float(np.exp(-r * T))

        valid_indices: list[int] = []
        prices_list: list[float] = []
        spreads_list: list[float] = []
        dropped = {"stale": 0, "spread": 0, "intrinsic": 0, "itm": 0, "oi": 0}

        for i, oid in enumerate(self.active_option_ids):
            q_tuple = self.opt_quotes.get(oid)
            if q_tuple is None:
                continue
            bid, ask, mid, ts_ns = q_tuple

            # 1. Quote freshness filter
            if max_age_ns > 0 and (now_ns - ts_ns) > max_age_ns:
                dropped["stale"] += 1
                continue

            # 2. Sane bid/ask & minimum premium filter
            spread = ask - bid
            if mid < self.config.min_mid_price or (spread / mid) > self.config.max_rel_spread:
                dropped["spread"] += 1
                continue

            # 3. Open-interest gate (when OI metadata > 0)
            if self.config.min_open_interest > 0 and self.open_interests[i] > 0:
                if self.open_interests[i] < self.config.min_open_interest:
                    dropped["oi"] += 1
                    continue

            # 4. Strict ATM + OTM splice filter
            k = float(self.strikes[i])
            is_call = bool(self.is_calls[i])
            if is_call and k < forward * (1.0 - self.config.atm_overlap_pct):
                dropped["itm"] += 1
                continue
            if (not is_call) and k > forward * (1.0 + self.config.atm_overlap_pct):
                dropped["itm"] += 1
                continue

            # 5. Pre-solver intrinsic arbitrage bound (eliminates slow IV fallback spikes)
            intrinsic = disc * max(forward - k, 0.0) if is_call else disc * max(k - forward, 0.0)
            if mid <= intrinsic + 1e-4:
                dropped["intrinsic"] += 1
                continue

            valid_indices.append(i)
            prices_list.append(mid)
            spreads_list.append(max(spread, 0.01))

        if len(valid_indices) < 5:
            self.log.warning(f"FastVolActor insufficient clean quotes ({len(valid_indices)}): {dropped}")
            return

        prices = np.asarray(prices_list, dtype=np.float64)
        spreads = np.asarray(spreads_list, dtype=np.float64)
        weights = 1.0 / (spreads ** 2)
        K = self.strikes[valid_indices]
        flags = np.where(self.is_calls[valid_indices], "c", "p")
        S = np.full_like(K, self.spot_mid)
        T_arr = np.full_like(K, T)
        r_arr = np.full_like(K, r_carry)

        ivs = np.asarray(
            fast_implied_volatility(
                price=prices, S=S, K=K, t=T_arr, r=r_arr, flag=flags, return_as="numpy"
            ),
            dtype=np.float64,
        )
        mask = np.isfinite(ivs) & (ivs > 0.01) & (ivs < 3.0)
        if np.count_nonzero(mask) < 5:
            return

        K_clean = K[mask]
        iv_clean = ivs[mask]
        w_clean = weights[mask]
        log_m = np.log(K_clean / forward)

        # Spread-weighted aggregation when both ATM Call & Put share a strike
        unique_lm, inv_idx = np.unique(np.round(log_m, 6), return_inverse=True)
        agg_w_iv = np.zeros_like(unique_lm)
        agg_w = np.zeros_like(unique_lm)
        np.add.at(agg_w_iv, inv_idx, iv_clean * w_clean)
        np.add.at(agg_w, inv_idx, w_clean)
        log_m_grid = unique_lm
        iv_raw_1d = agg_w_iv / agg_w

        if len(log_m_grid) < 5:
            return

        iv_svi_1d, svi_params, rmse_vol = _fit_svi_slice(
            log_m_grid, iv_raw_1d, agg_w, T, x0_warm=self._prev_svi_params
        )

        # Build per-leg quote & model residual table for the relative-value scanner
        svi_by_clean_idx = iv_svi_1d[inv_idx]
        clean_valid_indices = [valid_indices[j] for j in np.flatnonzero(mask)]
        legs_table = []
        for idx_pos, orig_i in enumerate(clean_valid_indices):
            oid = self.active_option_ids[orig_i]
            b_px, a_px, m_px, _ = self.opt_quotes[oid]
            raw_v = float(iv_clean[idx_pos])
            svi_v = float(svi_by_clean_idx[idx_pos])
            # Approximate BS vega to convert vol edge into dollar edge vs half-spread
            k_val = float(K_clean[idx_pos])
            d1 = (np.log(forward / k_val) + 0.5 * (svi_v ** 2) * T) / max(svi_v * np.sqrt(T), 1e-6)
            vega = float(self.spot_mid * np.exp(-q_eff * T) * np.sqrt(T) * (np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)))
            dollar_edge = abs(raw_v - svi_v) * vega
            half_spread = max(0.5 * (a_px - b_px), 0.005)
            legs_table.append(
                {
                    "instrument_id": str(oid),
                    "strike": round(k_val, 2),
                    "right": "C" if self.is_calls[orig_i] else "P",
                    "bid": round(b_px, 4),
                    "ask": round(a_px, 4),
                    "mid": round(m_px, 4),
                    "raw_iv": round(raw_v, 4),
                    "svi_iv": round(svi_v, 4),
                    "residual_pts": round((raw_v - svi_v) * 100.0, 4),
                    "vega": round(vega, 4),
                    "edge_to_half_spread": round(dollar_edge / half_spread, 3),
                }
            )

        t_grid = np.array([T, T + (1.0 / 365.25)])
        iv_grid = np.column_stack([iv_svi_1d, iv_svi_1d * 1.0001])
        surf = IVSurface.from_logmoneyness(log_m_grid, t_grid, iv_grid)
        arb_report = surf.validate()
        arb_free = bool(arb_report.passed)

        if arb_free:
            self._prev_svi_params = svi_params.copy()

        atm_iv = float(np.interp(0.0, log_m_grid, iv_svi_1d))
        put_25d_iv = float(np.interp(-0.05, log_m_grid, iv_svi_1d))
        call_25d_iv = float(np.interp(0.05, log_m_grid, iv_svi_1d))
        skew_25d = put_25d_iv - call_25d_iv
        max_resid_pts = float(np.max(np.abs(iv_raw_1d - iv_svi_1d)) * 100.0)

        snapshot = SviSliceParams(
            underlying_id=self.config.underlying_id,
            expiry_ns=self.expiry_ns or (now_ns + int(T * 365.25 * 86400 * 1e9)),
            forward=float(forward),
            atm_iv=atm_iv,
            skew_25d=skew_25d,
            arb_free=arb_free,
            quote_count=int(np.count_nonzero(mask)),
            ts_event=now_ns,
            ts_init=self.clock.timestamp_ns(),
        )
        self.publish_data(
            DataType("SviSliceParams"),
            CustomData(DataType("SviSliceParams"), snapshot),
        )
        if self.rdb is not None:
            payload = {
                "underlying": str(self.config.underlying_id),
                "spot": round(self.spot_mid, 2),
                "spot_age_secs": round(spot_age_secs, 2),
                "forward": round(float(forward), 2),
                "parity_pairs": parity_pairs,
                "implied_q": round(float(q_eff), 4),
                "ttm_days": round(T * 365.25, 2),
                "atm_iv": round(atm_iv, 4),
                "skew_25d": round(skew_25d, 4),
                "arb_free": arb_free,
                "rmse_vol_pts": round(rmse_vol * 100.0, 4),
                "max_resid_pts": round(max_resid_pts, 4),
                "svi_params": [round(float(x), 6) for x in svi_params],
                "quotes": int(np.count_nonzero(mask)),
                "dropped": dropped,
                "log_moneyness": log_m_grid.round(4).tolist(),
                "ivs": iv_svi_1d.round(4).tolist(),
                "raw_ivs": iv_raw_1d.round(4).tolist(),
                        "legs": legs_table,
                "ts_ns": time.time_ns(),
                "event_ts_ns": now_ns,
            }
            self.rdb.set(f"mft:surface:{self.config.underlying_id}", json.dumps(payload))
        self.log.info(
            f"Surface {self.config.underlying_id} S={self.spot_mid:.2f} F={forward:.2f} "
            f"(pairs={parity_pairs}, q={q_eff:.2%}) T={T*365.25:.1f}d ATM_IV={atm_iv:.2%} "
            f"RMSE={rmse_vol*100:.3f}pts arb_free={arb_free}"
        )
