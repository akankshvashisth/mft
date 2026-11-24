"""Every engine call the actor and strategy shells make, behind one replaceable object per component.

Tests swap `component.ports` for a fake that records subscriptions, timers, publications and orders.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

from nautilus_trader.model import CustomData, DataType

from src.adapters.nautilus import to_instrument_id


class ActorPorts:
    def __init__(self, component: Any) -> None:
        self._component = component

    # Clock and cache
    def now_ns(self) -> int:
        return int(self._component.clock.timestamp_ns())

    def instrument(self, instrument_id: Any) -> Any:
        return self._component.cache.instrument(to_instrument_id(instrument_id))

    def instruments(self) -> list[Any]:
        return list(self._component.cache.instruments())

    def quote(self, instrument_id: Any) -> Any:
        return self._component.cache.quote(to_instrument_id(instrument_id))

    def positions_open(self, instrument_id: Any = None) -> list[Any]:
        if instrument_id is None:
            return list(self._component.cache.positions_open())
        return list(self._component.cache.positions_open(instrument_id=to_instrument_id(instrument_id)))

    def positions_closed(self) -> list[Any]:
        return list(self._component.cache.positions_closed())

    # Subscriptions and timers
    def subscribe_quotes(self, instrument_id: Any) -> None:
        self._component.subscribe_quotes(to_instrument_id(instrument_id))

    def subscribe_bars(self, bar_type: Any) -> None:
        self._component.subscribe_bars(bar_type)

    def set_timer(self, name: str, interval_secs: float, callback: Callable[[Any], None]) -> None:
        self._component.clock.set_timer(name, timedelta(seconds=interval_secs), callback=callback)

    # Output
    def publish_custom(self, type_name: str, data: Any) -> None:
        data_type = DataType(type_name)
        self._component.publish_data(data_type, CustomData(data_type, data))

    def info(self, msg: str) -> None:
        self._component.log.info(msg)

    def warning(self, msg: str) -> None:
        self._component.log.warning(msg)

    def error(self, msg: str) -> None:
        self._component.log.error(msg)


class StrategyPorts(ActorPorts):
    def order(self, client_order_id: Any) -> Any:
        return self._component.cache.order(client_order_id)

    def add_instrument(self, instrument: Any) -> None:
        if self._component.cache.instrument(instrument.id) is None:
            self._component.cache.add_instrument(instrument)

    def limit_order(self, instrument_id: Any, order_side: Any, quantity: Any, price: Any, time_in_force: Any) -> Any:
        return self._component.order_factory.limit(
            instrument_id=instrument_id,
            order_side=order_side,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            post_only=False,
        )

    def submit(self, order: Any) -> None:
        self._component.submit_order(order)

    def modify_price(self, order: Any, price: Any) -> None:
        self._component.modify_order(order, price=price)

    def cancel(self, order: Any) -> None:
        self._component.cancel_order(order)


class PortsLog:
    """LogLike that forwards to `owner.ports` at call time, so swapping the ports also redirects logs."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def info(self, msg: str) -> None:
        self._owner.ports.info(msg)

    def warning(self, msg: str) -> None:
        self._owner.ports.warning(msg)

    def error(self, msg: str) -> None:
        self._owner.ports.error(msg)


class PortsProxy:
    """Resolves `owner.ports` on every attribute access (gateways keep working after a ports swap)."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._owner.ports, name)
