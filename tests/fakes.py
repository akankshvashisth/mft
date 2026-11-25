"""Test doubles.

Infrastructure doubles come from libraries: Valkey is fakeredis, and sinks and calendars are unittest.mock
autospecs of the real classes, so every call is checked against the real signature (calendar results are the
real EventCheckResult). The classes further down fake project interfaces whose behaviour no library provides:
the portfolio book market, the VRP engine collaborators and the Nautilus ports.
"""
from __future__ import annotations

from typing import Any
from unittest import mock

import fakeredis

from src.common.jsonl import JsonlSink
from src.data.event_calendar import EventCalendar, EventCheckResult


# ------------------------------------------------------------------------------
# Library-backed infrastructure doubles
# ------------------------------------------------------------------------------

def valkey_client(connected: bool = True) -> fakeredis.FakeRedis:
    """Isolated in-memory Valkey (fakeredis); with connected=False every command raises redis.ConnectionError."""
    server = fakeredis.FakeServer()
    server.connected = connected
    return fakeredis.FakeRedis(server=server)


def recording_sink(error: Exception | None = None) -> mock.NonCallableMagicMock:
    """JsonlSink autospec that records write() calls, or raises `error` from them."""
    sink = mock.create_autospec(JsonlSink, instance=True)
    sink.write.side_effect = error
    return sink


def written(sink: mock.NonCallableMagicMock) -> list[dict[str, Any]]:
    """Records handed to a recording_sink, oldest first."""
    return [call.args[0] for call in sink.write.call_args_list]


def stub_calendar(result: EventCheckResult | None = None) -> mock.NonCallableMagicMock:
    """EventCalendar autospec whose check_entry records its calls and returns `result` (allowed by default)."""
    calendar = mock.create_autospec(EventCalendar, instance=True)
    calendar.check_entry.return_value = result if result is not None else EventCheckResult()
    return calendar


# ------------------------------------------------------------------------------
# Portfolio book market (src.services.portfolio.BookMarket)
# ------------------------------------------------------------------------------

class FakeBookMarket:
    """Instrument infos, quotes and symbol -> instrument ids held in plain dicts."""

    def __init__(self, instruments=None, quotes=None, symbols=None) -> None:
        self.instruments: dict[str, Any] = dict(instruments or {})
        self.quotes: dict[str, Any] = dict(quotes or {})
        self.symbols: dict[str, list[str]] = dict(symbols or {})

    def instrument(self, instrument_id: str) -> Any:
        return self.instruments.get(instrument_id)

    def quote(self, instrument_id: str) -> Any:
        return self.quotes.get(instrument_id)

    def instruments_for_symbol(self, symbol: str) -> list[str]:
        return list(self.symbols.get(symbol, []))


# ------------------------------------------------------------------------------
# VRP engine collaborators (gate, pricer, order gateway, market view)
# ------------------------------------------------------------------------------

class FakeGate:
    """PreTradeGate stand-in: approves the requested size unless `decide` says otherwise."""

    def __init__(self, decide=None, block_reason: str | None = None) -> None:
        self.decide = decide
        self.block_reason = block_reason
        self.calls: list[dict[str, Any]] = []
        self.submissions = 0
        self.daemon_checks = 0

    def evaluate_combo(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.decide is not None:
            return self.decide(**kwargs)
        from src.risk.gate_types import GateDecision

        qty = min((leg.qty for leg in kwargs["legs"]), default=0)
        return GateDecision(approved=qty >= 1, approved_qty=qty, reasons=[], warnings=[])

    def risk_daemon_block_reason(self, now_ns: int | None = None) -> str | None:
        self.daemon_checks += 1
        return self.block_reason

    def record_order_submission(self) -> None:
        self.submissions += 1


class FakePricer:
    """ScenarioPricer stand-in returning assess(combo) (default: tests.factories.risk_assessment)."""

    def __init__(self, assess=None) -> None:
        if assess is None:
            from tests.factories import risk_assessment

            def assess(combo):
                return risk_assessment(combo_name=combo.name)

        self.assess = assess
        self.calls: list[dict[str, Any]] = []

    def evaluate_combo(
        self,
        combo: Any,
        realized_vol_p: float = 0.165,
        horizon_days: float = 7.0,
        sleeve_equity_usd: float = 100_000.0,
        max_es_risk_pct: float = 0.01,
    ) -> Any:
        self.calls.append(
            {"combo": combo, "realized_vol_p": realized_vol_p, "horizon_days": horizon_days, "sleeve_equity_usd": sleeve_equity_usd}
        )
        return self.assess(combo)


class FakeOrderGateway:
    """Records prepared/submitted/modified/cancelled orders; order views default to open."""

    def __init__(self) -> None:
        self.prepared: list[Any] = []
        self.submitted: list[str] = []
        self.modified: list[tuple[str, float]] = []
        self.cancelled: list[str] = []
        self.views: dict[str, Any] = {}
        self.fail_prepare: Exception | None = None
        self.fail_submit: Exception | None = None
        self.modify_ok = True
        self.on_submit = None   # Optional hook(order) to simulate synchronous venue events
        self._seq = 0

    @staticmethod
    def spread_id(legs: list[dict]) -> str:
        return "SPREAD[" + ",".join(f"{leg['instrument_id']}:{int(leg['ratio'])}" for leg in legs) + "]"

    def prepare(self, legs: list[dict], side: str, lots: int, limit_price: float) -> Any:
        from src.strategies.vrp_engine import OrderView, PreparedOrder

        if self.fail_prepare is not None:
            raise self.fail_prepare
        self._seq += 1
        order = PreparedOrder(
            order_key=f"O-{self._seq}",
            spread_id=self.spread_id(legs),
            handle={"legs": legs, "side": side, "lots": lots, "limit": limit_price},
        )
        self.prepared.append(order)
        self.views[order.order_key] = OrderView(is_open=True, is_closed=False, status="ACCEPTED")
        return order

    def submit(self, order: Any) -> None:
        if self.fail_submit is not None:
            raise self.fail_submit
        self.submitted.append(order.order_key)
        if self.on_submit is not None:
            self.on_submit(order)

    def modify_price(self, order_key: str, price: float) -> bool:
        self.modified.append((order_key, price))
        return self.modify_ok

    def cancel(self, order_key: str) -> None:
        self.cancelled.append(order_key)

    def order_view(self, order_key: str) -> Any:
        return self.views.get(order_key)

    def set_view(self, order_key: str, **fields: Any) -> None:
        from src.strategies.vrp_engine import OrderView

        base = {"is_open": False, "is_closed": True, "status": "FILLED"}
        base.update(fields)
        self.views[order_key] = OrderView(**base)

    def open_order_keys(self) -> list[str]:
        return [k for k, v in self.views.items() if v.is_open]

    def release(self, order_key: str) -> None:
        pass


class FakeMarket:
    """MarketView stand-in: cache quotes, option specs, net positions and subscriptions."""

    def __init__(self) -> None:
        self.quotes: dict[str, Any] = {}
        self.specs: dict[str, Any] = {}
        self.positions: dict[str, float] = {}
        self.subscribed: list[str] = []

    def cache_quote(self, instrument_id: str) -> Any:
        return self.quotes.get(instrument_id)

    def option_spec(self, instrument_id: str) -> Any:
        return self.specs.get(instrument_id)

    def net_position_qty(self, instrument_id: str) -> float:
        return self.positions.get(instrument_id, 0.0)

    def ensure_subscribed(self, instrument_id: str) -> None:
        if instrument_id not in self.subscribed:
            self.subscribed.append(instrument_id)


# ------------------------------------------------------------------------------
# Actor / strategy ports (src.adapters.ports)
# ------------------------------------------------------------------------------

class FakeActorPorts:
    """Records subscriptions, timers, publications and logs; cache lookups come from plain dicts."""

    def __init__(self, now_ns: int = 0, instruments=None, quotes=None, positions_open=None, positions_closed=None) -> None:
        self.clock_ns = now_ns
        self.instruments_by_id: dict[str, Any] = dict(instruments or {})
        self.quotes: dict[str, Any] = dict(quotes or {})
        self.open_positions: list[Any] = list(positions_open or [])
        self.closed_positions: list[Any] = list(positions_closed or [])
        self.quote_subscriptions: list[str] = []
        self.bar_subscriptions: list[str] = []
        self.timers: list[tuple[str, float, Any]] = []
        self.published: list[tuple[str, Any]] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def now_ns(self) -> int:
        return self.clock_ns

    def instrument(self, instrument_id: Any) -> Any:
        return self.instruments_by_id.get(str(instrument_id))

    def instruments(self) -> list[Any]:
        return list(self.instruments_by_id.values())

    def quote(self, instrument_id: Any) -> Any:
        return self.quotes.get(str(instrument_id))

    def positions_open(self, instrument_id: Any = None) -> list[Any]:
        return [p for p in self.open_positions if instrument_id is None or str(p.instrument_id) == str(instrument_id)]

    def positions_closed(self) -> list[Any]:
        return list(self.closed_positions)

    def subscribe_quotes(self, instrument_id: Any) -> None:
        self.quote_subscriptions.append(str(instrument_id))

    def subscribe_bars(self, bar_type: Any) -> None:
        self.bar_subscriptions.append(str(bar_type))

    def set_timer(self, name: str, interval_secs: float, callback: Any) -> None:
        self.timers.append((name, interval_secs, callback))

    def publish_custom(self, type_name: str, data: Any) -> None:
        self.published.append((type_name, data))

    def info(self, msg: str) -> None:
        self.infos.append(msg)

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)


class FakeStrategyPorts(FakeActorPorts):
    """Adds order handling: limit orders are SimpleNamespace stand-ins stored by client order id."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.orders: dict[str, Any] = {}
        self.added_instruments: list[Any] = []
        self.limit_orders: list[Any] = []
        self.submitted: list[Any] = []
        self.modified: list[tuple[Any, Any]] = []
        self.cancelled: list[Any] = []

    def order(self, client_order_id: Any) -> Any:
        return self.orders.get(str(client_order_id))

    def add_instrument(self, instrument: Any) -> None:
        if self.instrument(instrument.id) is None:
            self.instruments_by_id[str(instrument.id)] = instrument
            self.added_instruments.append(instrument)

    def limit_order(self, instrument_id: Any, order_side: Any, quantity: Any, price: Any, time_in_force: Any) -> Any:
        from types import SimpleNamespace

        order = SimpleNamespace(
            client_order_id=f"O-{len(self.limit_orders) + 1}",
            instrument_id=instrument_id,
            side=order_side,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            is_open=True,
            is_closed=False,
            is_pending_update=False,
            is_pending_cancel=False,
            status="INITIALIZED",
        )
        self.limit_orders.append(order)
        self.orders[order.client_order_id] = order
        return order

    def submit(self, order: Any) -> None:
        self.submitted.append(order)

    def modify_price(self, order: Any, price: Any) -> None:
        self.modified.append((order, price))

    def cancel(self, order: Any) -> None:
        self.cancelled.append(order)
