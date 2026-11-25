"""Nautilus implementations of the VRP engine's OrderGateway and MarketView protocols."""
from __future__ import annotations

from typing import Any

from nautilus_trader.model import (
    InstrumentId,
    OptionContract,
    OptionSpread,
    OrderSide,
    Symbol,
    TimeInForce,
)

from src.adapters.extract import option_spec_from_instrument
from src.adapters.nautilus import quote_obs
from src.common.market import OptionSpec, QuoteObs
from src.execution.passive_combo import CREDIT_OPEN
from src.strategies.vrp_engine import OrderView, PreparedOrder


class NautilusOrderGateway:
    """Builds (and caches) OptionSpread instruments and DAY limit orders through the strategy ports."""

    def __init__(self, ports: Any) -> None:
        self.ports = ports
        self._spreads: dict[tuple[tuple[str, int], ...], Any] = {}
        self._client_ids: dict[str, Any] = {}

    def spread_instrument(self, legs: list[dict]) -> Any:
        signature = tuple((str(leg["instrument_id"]), int(leg["ratio"])) for leg in legs)
        spread = self._spreads.get(signature)
        if spread is None:
            parsed_legs = [(InstrumentId.from_str(iid), ratio) for iid, ratio in signature]
            contracts = [self.ports.instrument(instrument_id) for instrument_id, _ in parsed_legs]
            if not all(isinstance(contract, OptionContract) for contract in contracts):
                raise ValueError("Every spread leg must refer to a cached OptionContract")

            first = contracts[0]
            if any(
                (contract.underlying, contract.expiration_ns, contract.currency, contract.multiplier)
                != (first.underlying, first.expiration_ns, first.currency, first.multiplier)
                for contract in contracts[1:]
            ):
                raise ValueError("Spread legs must share an underlying, expiration, currency, and multiplier")

            venues = {str(instrument_id.venue) for instrument_id, _ in parsed_legs}
            if len(venues) != 1:
                raise ValueError("Spread legs must share a venue")

            raw_symbol = "SPREAD-" + "-".join(
                f"{instrument_id.symbol}".replace(" ", "-")
                + ("-P" if ratio >= 0 else "-M")
                + str(abs(ratio))
                for instrument_id, ratio in parsed_legs
            )
            symbol = Symbol.from_str(raw_symbol)
            venue = next(iter(venues))
            now_ns = self.ports.now_ns()
            spread = OptionSpread(
                instrument_id=InstrumentId.from_str(f"{symbol}.{venue}"),
                raw_symbol=symbol,
                asset_class=first.asset_class,
                underlying=first.underlying,
                strategy_type="CUSTOM",
                activation_ns=first.activation_ns,
                expiration_ns=first.expiration_ns,
                currency=first.currency,
                price_precision=first.price_precision,
                price_increment=first.price_increment,
                multiplier=first.multiplier,
                lot_size=first.lot_size,
                ts_event=now_ns,
                ts_init=now_ns,
                info={
                    "legs": [
                        {"instrument_id": str(instrument_id), "ratio": ratio}
                        for instrument_id, ratio in parsed_legs
                    ]
                },
            )
            self.ports.add_instrument(spread)
            self._spreads[signature] = spread
        return spread

    def prepare(self, legs: list[dict], side: str, lots: int, limit_price: float) -> PreparedOrder:
        spread = self.spread_instrument(legs)
        order = self.ports.limit_order(
            spread.id,
            OrderSide.SELL if side == CREDIT_OPEN else OrderSide.BUY,
            spread.make_qty(lots),
            spread.make_price(limit_price),
            TimeInForce.DAY,
        )
        order_key = str(order.client_order_id)
        self._client_ids[order_key] = order.client_order_id
        return PreparedOrder(order_key=order_key, spread_id=str(spread.id), handle=order)

    def submit(self, order: PreparedOrder) -> None:
        self.ports.submit(order.handle)

    def _order(self, order_key: str) -> Any:
        client_order_id = self._client_ids.get(order_key)
        return self.ports.order(client_order_id) if client_order_id is not None else None

    def order_view(self, order_key: str) -> OrderView | None:
        order = self._order(order_key)
        if order is None:
            return None
        return OrderView(
            is_open=bool(order.is_open),
            is_closed=bool(order.is_closed),
            is_pending_update=bool(order.is_pending_update),
            is_pending_cancel=bool(order.is_pending_cancel),
            status=str(order.status),
        )

    def modify_price(self, order_key: str, price: float) -> bool:
        order = self._order(order_key)
        instrument = self.ports.instrument(order.instrument_id) if order is not None else None
        if instrument is None:
            return False
        self.ports.modify_price(order, instrument.make_price(price))
        return True

    def cancel(self, order_key: str) -> None:
        order = self._order(order_key)
        if order is not None:
            self.ports.cancel(order)

    def open_order_keys(self) -> list[str]:
        """Returns keys of all pre-existing open DAY orders (for crash restart reconciliation).

        Queries ``self.ports.orders_open()``, registers each found order's ``client_order_id``
        into ``self._client_ids`` so ``cancel(key)`` works, and returns their string keys.
        """
        keys: list[str] = []
        for order in self.ports.orders_open():
            cid = str(order.client_order_id)
            self._client_ids[cid] = order.client_order_id
            keys.append(cid)
        return keys

    def release(self, order_key: str) -> None:
        """Prunes a finalized order key from _client_ids."""
        self._client_ids.pop(order_key, None)


class NautilusMarketView:
    """Cache quotes, option metadata, net positions and on-demand quote subscriptions."""

    def __init__(self, ports: Any) -> None:
        self.ports = ports
        self.subscribed: set[str] = set()

    def cache_quote(self, instrument_id: str) -> QuoteObs | None:
        tick = self.ports.quote(instrument_id)
        return quote_obs(tick) if tick is not None else None

    def option_spec(self, instrument_id: str) -> OptionSpec | None:
        instrument = self.ports.instrument(instrument_id)
        return option_spec_from_instrument(instrument) if isinstance(instrument, OptionContract) else None

    def net_position_qty(self, instrument_id: str) -> float:
        return sum(float(p.signed_qty) for p in self.ports.positions_open(instrument_id))

    def ensure_subscribed(self, instrument_id: str) -> None:
        if instrument_id in self.subscribed or self.ports.instrument(instrument_id) is None:
            return
        self.ports.subscribe_quotes(instrument_id)
        self.subscribed.add(instrument_id)
