from __future__ import annotations

import json
import time
from typing import Any

import redis

# ==============================================================================
# NAUTILUSTRADER LIVE / SHADOW STRATEGY INTEGRATION
# ==============================================================================
from datetime import date, datetime, timedelta, timezone
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import (
    InstrumentId,
    OptionSpread,
    OrderSide,
    QuoteTick,
    TimeInForce,
)
from nautilus_trader.trading import Strategy

from src.execution.passive_combo import PassiveComboExecutor, PassiveExecutionConfig
from src.risk.pre_trade_gate import CandidateLegSpec, GateConfig, PreTradeGate
from src.risk.scenario_pricer import ScenarioPricer
from src.scanner.surface_scanner import CandidateCombo, CandidateLeg


class VrpIncomeStrategyConfig(StrategyConfig):
    def __init__(
        self,
        underlying_id: InstrumentId,
        option_ids: tuple[InstrumentId, ...],
        paper_routing_enabled: bool = False,
        eval_interval_secs: int = 15,
        sleeve_equity_usd: float = 100_000.0,
        profit_target_fraction: float = 0.50,
        min_exit_dte: int = 14,
        valkey_url: str = "redis://127.0.0.1:6379/0",
    ) -> None:
        super().__init__()
        self.underlying_id = underlying_id
        self.option_ids = option_ids
        self.paper_routing_enabled = paper_routing_enabled
        self.eval_interval_secs = eval_interval_secs
        self.sleeve_equity_usd = sleeve_equity_usd
        self.profit_target_fraction = profit_target_fraction
        self.min_exit_dte = min_exit_dte
        self.valkey_url = valkey_url


class VrpIncomeStrategy(Strategy):
    """Multi-actor VRP & relative-value strategy with PreTradeGate + Passive OptionSpread execution."""

    def __init__(self, config: VrpIncomeStrategyConfig) -> None:
        super().__init__(config)
        self.rdb = redis.from_url(config.valkey_url) if config.valkey_url else None
        self.quotes: dict[InstrumentId, tuple[float, float, int, int]] = {}
        self.spot_mid: float = 0.0
        self.gate = PreTradeGate(
            config=GateConfig(sleeve_equity_usd=config.sleeve_equity_usd),
            valkey_url=config.valkey_url,
        )
        self.pricer = ScenarioPricer(
            n_scenarios=5_000,
            commission_per_leg=0.65,
            slippage_fraction=0.50,
        )
        self.executor = PassiveComboExecutor(PassiveExecutionConfig())

    def on_start(self) -> None:
        self.subscribe_quotes(self.config.underlying_id)
        for oid in self.config.option_ids:
            if self.cache.instrument(oid) is not None:
                self.subscribe_quotes(oid)

        self.clock.set_timer(
            "vrp_strategy_eval",
            timedelta(seconds=self.config.eval_interval_secs),
            callback=self.on_timer,
        )
        mode = "LIVE_PAPER_ROUTING" if self.config.paper_routing_enabled else "SHADOW_DRY_RUN"
        self.log.info(
            f"VrpIncomeStrategy started in {mode} mode watching {self.config.underlying_id} "
            f"+ {len(self.config.option_ids)} options"
        )

    def on_quote(self, tick: QuoteTick) -> None:
        bid = float(tick.bid_price)
        ask = float(tick.ask_price)
        if bid <= 0.0 or ask <= 0.0 or ask < bid:
            return
        if tick.instrument_id == self.config.underlying_id:
            self.spot_mid = 0.5 * (bid + ask)
            # Check working passive clips for adverse underlying moves
            actions = self.executor.on_market_tick(
                underlying=str(self.config.underlying_id),
                current_spot=self.spot_mid,
                new_mid_price=0.0,
                new_natural_price=0.0,
            )
            for act in actions:
                self.log.warning(f"PassiveCombo adverse-selection action: {act}")
        else:
            b_sz = int(getattr(tick, "bid_size", 10) or 10)
            a_sz = int(getattr(tick, "ask_size", 10) or 10)
            self.quotes[tick.instrument_id] = (bid, ask, b_sz, a_sz)

    def _read_json(self, key: str) -> dict[str, Any]:
        if self.rdb is None:
            return {}
        raw = self.rdb.get(key)
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _build_candidate_spreads(self, surf: dict[str, Any]) -> list[tuple[CandidateCombo, list[CandidateLegSpec], float, float]]:
        """Builds defined-risk spreads from surface legs overlaid with live strategy/cache quotes."""
        spot = self.spot_mid or float(surf.get("spot") or 0.0)
        fwd = float(surf.get("forward") or spot)
        ttm_days = float(surf.get("ttm_days") or 25.0)
        imp_q = float(surf.get("implied_q") or 0.02)
        atm_iv = float(surf.get("atm_iv") or 0.23)
        raw_legs = surf.get("legs") or []
        if spot <= 0.0 or not raw_legs:
            return []

        default_exp = date.today() + timedelta(days=max(1, int(round(ttm_days))))
        enriched: list[dict[str, Any]] = []

        for rl in raw_legs:
            oid_str = str(rl.get("instrument_id", ""))
            if not oid_str:
                continue
            oid = InstrumentId.from_str(oid_str)
            bid = float(rl.get("bid") or 0.0)
            ask = float(rl.get("ask") or 0.0)
            b_sz = 10
            a_sz = 10

            if oid in self.quotes:
                q_bid, q_ask, q_bsz, q_asz = self.quotes[oid]
                if q_bid > 0.0 and q_ask >= q_bid:
                    bid, ask, b_sz, a_sz = q_bid, q_ask, q_bsz, q_asz
            else:
                qt = self.cache.quote(oid)
                if qt is not None:
                    q_bid, q_ask = float(qt.bid_price), float(qt.ask_price)
                    if q_bid > 0.0 and q_ask >= q_bid:
                        bid, ask = q_bid, q_ask
                        b_sz = int(getattr(qt, "bid_size", 10) or 10)
                        a_sz = int(getattr(qt, "ask_size", 10) or 10)

            if bid <= 0.0 or ask < bid:
                continue

            mid = 0.5 * (bid + ask)
            inst = self.cache.instrument(oid)
            exp_dt = default_exp
            if inst is not None:
                exp_ns = int(getattr(inst, "expiration_ns", 0) or 0)
                if exp_ns > 0:
                    exp_dt = datetime.fromtimestamp(exp_ns / 1e9, tz=timezone.utc).date()

            enriched.append(
                {
                    "oid": oid,
                    "oid_str": oid_str,
                    "strike": float(rl.get("strike") or 0.0),
                    "right": str(rl.get("right") or "").upper(),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "b_sz": b_sz,
                    "a_sz": a_sz,
                    "exp_dt": exp_dt,
                    "raw_iv": float(rl.get("raw_iv") or atm_iv),
                    "svi_iv": float(rl.get("svi_iv") or atm_iv),
                    "res_pts": float(rl.get("residual_pts") or 0.0),
                    "ehs": float(rl.get("edge_to_half_spread") or 0.0),
                }
            )

        puts = sorted([x for x in enriched if x["right"] in ("P", "PUT")], key=lambda x: x["strike"])
        calls = sorted([x for x in enriched if x["right"] in ("C", "CALL")], key=lambda x: x["strike"])

        candidates: list[tuple[CandidateCombo, list[CandidateLegSpec], float, float]] = []
        sym_code = self.config.underlying_id.symbol

        # Bull Put Credit Spreads
        for i in range(1, len(puts)):
            lp, sp = puts[i - 1], puts[i]
            width = sp["strike"] - lp["strike"]
            if width <= 0.0 or width > 15.0:
                continue
            mid_credit = sp["mid"] - lp["mid"]
            nat_credit = sp["bid"] - lp["ask"]
            if mid_credit <= 0.05:
                continue

            combo = CandidateCombo(
                name=f"{sym_code} {int(sp['strike'])}/{int(lp['strike'])}P Bull Put",
                strategy_type="BULL_PUT_SPREAD",
                underlying=str(self.config.underlying_id),
                spot=spot,
                forward=fwd,
                ttm_days=ttm_days,
                risk_free_rate=0.0425,
                dividend_yield=imp_q,
                legs=[
                    CandidateLeg(sp["oid_str"], sp["strike"], "P", -1, 1, sp["bid"], sp["ask"], sp["mid"], sp["raw_iv"], sp["svi_iv"], sp["res_pts"], sp["ehs"]),
                    CandidateLeg(lp["oid_str"], lp["strike"], "P", +1, 1, lp["bid"], lp["ask"], lp["mid"], lp["raw_iv"], lp["svi_iv"], lp["res_pts"], lp["ehs"]),
                ],
                net_credit=round(mid_credit, 4),
                max_width=round(width, 2),
            )
            gate_legs = [
                CandidateLegSpec(sp["oid_str"], "P", "SELL", sp["strike"], sp["exp_dt"], qty=2, bid=sp["bid"], ask=sp["ask"], bid_size=sp["b_sz"], ask_size=sp["a_sz"], open_interest=2500),
                CandidateLegSpec(lp["oid_str"], "P", "BUY",  lp["strike"], lp["exp_dt"], qty=2, bid=lp["bid"], ask=lp["ask"], bid_size=lp["b_sz"], ask_size=lp["a_sz"], open_interest=2500),
            ]
            candidates.append((combo, gate_legs, mid_credit, nat_credit))

        # Bear Call Credit Spreads
        for i in range(0, len(calls) - 1):
            sc, lc = calls[i], calls[i + 1]
            width = lc["strike"] - sc["strike"]
            if width <= 0.0 or width > 15.0:
                continue
            mid_credit = sc["mid"] - lc["mid"]
            nat_credit = sc["bid"] - lc["ask"]
            if mid_credit <= 0.05:
                continue

            combo = CandidateCombo(
                name=f"{sym_code} {int(sc['strike'])}/{int(lc['strike'])}C Bear Call",
                strategy_type="BEAR_CALL_SPREAD",
                underlying=str(self.config.underlying_id),
                spot=spot,
                forward=fwd,
                ttm_days=ttm_days,
                risk_free_rate=0.0425,
                dividend_yield=imp_q,
                legs=[
                    CandidateLeg(sc["oid_str"], sc["strike"], "C", -1, 1, sc["bid"], sc["ask"], sc["mid"], sc["raw_iv"], sc["svi_iv"], sc["res_pts"], sc["ehs"]),
                    CandidateLeg(lc["oid_str"], lc["strike"], "C", +1, 1, lc["bid"], lc["ask"], lc["mid"], lc["raw_iv"], lc["svi_iv"], lc["res_pts"], lc["ehs"]),
                ],
                net_credit=round(mid_credit, 4),
                max_width=round(width, 2),
            )
            gate_legs = [
                CandidateLegSpec(sc["oid_str"], "C", "SELL", sc["strike"], sc["exp_dt"], qty=2, bid=sc["bid"], ask=sc["ask"], bid_size=sc["b_sz"], ask_size=sc["a_sz"], open_interest=2500),
                CandidateLegSpec(lc["oid_str"], "C", "BUY",  lc["strike"], lc["exp_dt"], qty=2, bid=lc["bid"], ask=lc["ask"], bid_size=lc["b_sz"], ask_size=lc["a_sz"], open_interest=2500),
            ]
            candidates.append((combo, gate_legs, mid_credit, nat_credit))

        return candidates

    def _route_atomic_spread_order(
        self,
        combo: CandidateCombo,
        gate_legs: list[CandidateLegSpec],
        lots: int,
        limit_credit: float,
    ) -> str | None:
        """Constructs an atomic Nautilus OptionSpread contract and submits a passive LIMIT order."""
        try:
            leg_specs = []
            for gl in gate_legs:
                iid = InstrumentId.from_str(gl.instrument_id)
                ratio = -1 if gl.side == "SELL" else 1
                leg_specs.append((iid, ratio))
            spread_inst = OptionSpread.from_legs(leg_specs)
            self.cache.add_instrument(spread_inst)
            order = self.order_factory.limit(
                instrument_id=spread_inst.id,
                order_side=OrderSide.SELL,
                quantity=spread_inst.make_qty(lots),
                price=spread_inst.make_price(limit_credit),
                time_in_force=TimeInForce.GTC,
                post_only=False,
            )
            self.submit_order(order)
            return str(order.client_order_id)
        except Exception as exc:
            self.log.error(f"Failed to submit atomic OptionSpread order: {exc}")
            return None

    def on_timer(self, event) -> None:
        underlying = str(self.config.underlying_id)
        surf = self._read_json(f"mft:surface:{underlying}")
        rv_snap = self._read_json(f"mft:rv:{underlying}")
        rv_20d = float(rv_snap.get("rv_20d") or 0.25)
        spot = self.spot_mid or float(surf.get("spot") or 0.0)

        candidates = self._build_candidate_spreads(surf)
        evaluated_rows: list[dict[str, Any]] = []
        best_approved = None

        for combo, gate_legs, mid_credit, nat_credit in candidates:
            risk = self.pricer.evaluate_combo(
                combo,
                realized_vol_p=rv_20d,
                horizon_days=min(14.0, combo.ttm_days),
                sleeve_equity_usd=self.config.sleeve_equity_usd,
            )
            per_lot_es99 = abs(risk.es_99_usd) / max(1, risk.recommended_contracts)
            per_lot_ev = risk.net_ev_usd / max(1, risk.recommended_contracts)
            for gl in gate_legs:
                object.__setattr__(gl, "qty", max(1, risk.recommended_contracts))

            gate_dec = self.gate.evaluate_combo(
                underlying=underlying,
                spot_price=spot,
                legs=gate_legs,
                incremental_es99_per_lot_usd=per_lot_es99,
                net_ev_per_lot_usd=per_lot_ev,
            )
            row = {
                "combo": combo.name,
                "mid_credit": round(mid_credit, 3),
                "nat_credit": round(nat_credit, 3),
                "net_ev_per_lot": round(per_lot_ev, 2),
                "es99_per_lot": round(per_lot_es99, 2),
                "pop_pct": round(risk.prob_profit * 100.0, 1),
                "gate_approved": gate_dec.approved,
                "approved_qty": gate_dec.approved_qty,
                "gate_reasons": gate_dec.reasons,
                "gate_warnings": gate_dec.warnings,
            }
            evaluated_rows.append(row)
            if gate_dec.approved and (
                best_approved is None or per_lot_ev > best_approved[0]["net_ev_per_lot"]
            ):
                best_approved = (row, combo, gate_legs, gate_dec)

        evaluated_rows.sort(key=lambda r: r["net_ev_per_lot"], reverse=True)
        top_row = evaluated_rows[0] if evaluated_rows else None

        action = "STAND_DOWN"
        clips_planned = []
        routed_order_ids = []

        if best_approved is not None:
            row, combo, gate_legs, gate_dec = best_approved
            clips = self.executor.create_parent_plan(
                underlying=underlying,
                combo_label=combo.name,
                side="CREDIT_OPEN",
                total_lots=gate_dec.approved_qty,
                spot_price=spot,
                net_mid_price=row["mid_credit"],
                net_natural_price=row["nat_credit"],
            )
            clips_planned = [
                {
                    "clip_id": c.clip_id,
                    "lots": c.lots,
                    "mid_credit": c.net_mid_price,
                    "natural_credit": c.net_natural_price,
                    "passive_limit": c.current_limit_price,
                }
                for c in clips
            ]
            if self.config.paper_routing_enabled:
                action = "LIVE_PAPER_ORDER_ROUTED"
                for c in clips:
                    oid_str = self._route_atomic_spread_order(
                        combo, gate_legs, c.lots, c.current_limit_price
                    )
                    if oid_str:
                        routed_order_ids.append(oid_str)
            else:
                action = "SHADOW_SIGNAL_READY"

        payload = {
            "mode": "LIVE_PAPER_ROUTING" if self.config.paper_routing_enabled else "SHADOW_DRY_RUN",
            "action": action,
            "underlying": underlying,
            "spot": round(spot, 2),
            "rv_20d": round(rv_20d, 4),
            "candidates_scanned": len(evaluated_rows),
            "top_candidate": top_row,
            "all_candidates": evaluated_rows[:8],
            "passive_clips": clips_planned,
            "routed_order_ids": routed_order_ids,
            "ts_ns": time.time_ns(),
            "event_ts_ns": int(event.ts_event),
        }

        if self.rdb is not None:
            raw_json = json.dumps(payload)
            self.rdb.set("mft:strategy:vrp:latest", raw_json)
            self.rdb.lpush("mft:strategy:vrp:history", raw_json)
            self.rdb.ltrim("mft:strategy:vrp:history", 0, 199)

        if top_row:
            self.log.info(
                f"VRP[{payload['mode']}] {action} | Top={top_row['combo']} "
                f"Mid=${top_row['mid_credit']:.2f} EV=${top_row['net_ev_per_lot']:+.2f}/lot "
                f"ES99=${top_row['es99_per_lot']:.0f} | Reasons={top_row['gate_reasons']}"
            )
