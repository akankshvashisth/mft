from datetime import timedelta
import json
import math
import time
import redis
from nautilus_trader.common import DataActor, DataActorConfig
from nautilus_trader.model import InstrumentId, QuoteTick, TradeTick


class FuturesContextActorConfig(DataActorConfig):
    def __init__(
        self,
        spy_id: InstrumentId,
        mes_id: InstrumentId,
        spx_id: InstrumentId | None = None,
        mnq_id: InstrumentId | None = None,
        qqq_id: InstrumentId | None = None,
        risk_free_rate: float = 0.0425,
        default_index_to_etf_ratio: float = 10.0,
        interval_secs: int = 15,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.spy_id = spy_id
        self.mes_id = mes_id
        self.spx_id = spx_id
        self.mnq_id = mnq_id
        self.qqq_id = qqq_id
        self.risk_free_rate = risk_free_rate
        self.default_index_to_etf_ratio = default_index_to_etf_ratio
        self.interval_secs = interval_secs
        self.valkey_url = valkey_url


class FuturesContextActor(DataActor):
    """Streams front-month CME futures (MES/MNQ) + ^SPX.CBOE to benchmark SVI parity forward & carry."""

    def __init__(self, config: FuturesContextActorConfig) -> None:
        super().__init__(config)
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None
        self.mids: dict[InstrumentId, float] = {}
        self.last_ts_ns: dict[InstrumentId, int] = {}

    def on_start(self) -> None:
        subs = []
        for iid in (
            self.config.spy_id,
            self.config.spx_id,
            self.config.mes_id,
            self.config.mnq_id,
            self.config.qqq_id,
        ):
            if iid is not None and self.cache.instrument(iid) is not None:
                if iid != self.config.spx_id:
                    self.subscribe_quotes(iid)
                subs.append(str(iid))

        self.clock.set_timer(
            "futures_context_eval",
            timedelta(seconds=self.config.interval_secs),
            callback=self.on_timer,
        )
        self.log.info(f"FuturesContextActor subscribed to {', '.join(subs)}")

    def on_quote(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return
        self.mids[tick.instrument_id] = 0.5 * (bid + ask)
        self.last_ts_ns[tick.instrument_id] = int(tick.ts_event)

    def on_trade(self, tick: TradeTick) -> None:
        px = float(tick.price)
        if px > 0.0:
            self.mids[tick.instrument_id] = px
            self.last_ts_ns[tick.instrument_id] = int(tick.ts_event)

    def _read_json(self, key: str) -> dict:
        if self.rdb is None:
            return {}
        raw = self.rdb.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def on_timer(self, event) -> None:
        now_wall_ns = time.time_ns()
        surf = self._read_json(f"mft:surface:{self.config.spy_id}")

        spy_spot = self.mids.get(self.config.spy_id) or float(surf.get("spot") or 0.0)
        spx_px = self.mids.get(self.config.spx_id, 0.0) if self.config.spx_id else 0.0
        mes_px = self.mids.get(self.config.mes_id, 0.0)
        mnq_px = self.mids.get(self.config.mnq_id, 0.0) if self.config.mnq_id else 0.0

        if spy_spot <= 0.0 or mes_px <= 0.0:
            self.log.info(
                f"FuturesContext waiting for quotes (SPY={spy_spot:.2f}, SPX={spx_px:.2f}, MES={mes_px:.2f})"
            )
            return

        parity_fwd_pre = float(surf.get("forward") or 0.0)
        if spx_px > 0.0:
            scale = spx_px / spy_spot
            ratio_source = "LIVE_SPX_SPY"
        elif parity_fwd_pre > 0.0:
            if not hasattr(self, "_anchored_scale"):
                self._anchored_scale = mes_px / parity_fwd_pre
            scale = self._anchored_scale
            ratio_source = "PARITY_ANCHORED"
        else:
            scale = self.config.default_index_to_etf_ratio
            ratio_source = "DEFAULT_10X"

        mes_inst = self.cache.instrument(self.config.mes_id)
        mes_exp_ns = int(getattr(mes_inst, "expiration_ns", 0)) if mes_inst is not None else 0
        ttm_fut_days = max((mes_exp_ns - now_wall_ns) / (86400.0 * 1e9), 0.1) if mes_exp_ns > now_wall_ns else 25.0
        ttm_fut_yrs = ttm_fut_days / 365.25

        mes_scaled_fwd = mes_px / scale
        basis_usd = mes_scaled_fwd - spy_spot
        basis_pct = basis_usd / spy_spot
        q_fut = self.config.risk_free_rate - math.log(max(mes_scaled_fwd / spy_spot, 1e-6)) / ttm_fut_yrs

        parity_fwd = float(surf.get("forward") or 0.0)
        parity_q = float(surf.get("implied_q") or q_fut)
        ttm_opt_days = float(surf.get("ttm_days") or ttm_fut_days)
        ttm_opt_yrs = max(ttm_opt_days / 365.25, 1e-4)

        fut_implied_opt_fwd = spy_spot * math.exp((self.config.risk_free_rate - q_fut) * ttm_opt_yrs)
        fwd_diff_usd = (parity_fwd - fut_implied_opt_fwd) if parity_fwd > 0.0 else 0.0
        fwd_diff_bps = (fwd_diff_usd / spy_spot) * 10_000.0 if parity_fwd > 0.0 else 0.0
        q_diff_bps = (parity_q - q_fut) * 10_000.0 if parity_fwd > 0.0 else 0.0

        # 1 MES contract ($5 * SPX) normalized to SPY share-equivalents
        mes_spy_equiv_shares = 5.0 * scale

        payload = {
            "mes_symbol": str(self.config.mes_id),
            "mnq_symbol": str(self.config.mnq_id) if self.config.mnq_id else None,
            "spx_symbol": str(self.config.spx_id) if self.config.spx_id else None,
            "spy_spot": round(spy_spot, 3),
            "spx_quote": round(spx_px, 2) if spx_px > 0 else None,
            "spx_spy_ratio": round(scale, 5),
            "ratio_source": ratio_source,
            "mes_quote": round(mes_px, 2),
            "mnq_quote": round(mnq_px, 2) if mnq_px > 0 else None,
            "mes_ttm_days": round(ttm_fut_days, 2),
            "mes_scaled_fwd": round(mes_scaled_fwd, 3),
            "cash_futures_basis_usd": round(basis_usd, 3),
            "cash_futures_basis_pct": round(basis_pct, 6),
            "futures_implied_q": round(q_fut, 6),
            "parity_fwd": round(parity_fwd, 3) if parity_fwd > 0 else None,
            "fut_implied_opt_fwd": round(fut_implied_opt_fwd, 3),
            "parity_vs_fut_diff_usd": round(fwd_diff_usd, 3),
            "parity_vs_fut_diff_bps": round(fwd_diff_bps, 2),
            "parity_vs_fut_q_diff_bps": round(q_diff_bps, 2),
            "mes_spy_equiv_shares": round(mes_spy_equiv_shares, 2),
            "ts_ns": now_wall_ns,
            "event_ts_ns": int(event.ts_event),
        }

        if self.rdb is not None:
            self.rdb.set("mft:futures:latest", json.dumps(payload))

        self.log.info(
            f"FuturesBasis MES={mes_px:.2f} (Ratio={scale:.4f} [{ratio_source}], ScaledF={mes_scaled_fwd:.2f}, "
            f"q_fut={q_fut:.2%}) vs ParityF={parity_fwd:.2f} (q_par={parity_q:.2%}) | "
            f"Diff={fwd_diff_usd:+.2f} ({fwd_diff_bps:+.1f}bps)"
        )
