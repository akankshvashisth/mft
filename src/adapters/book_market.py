"""Portfolio BookMarket over actor ports (Nautilus cache instruments and quotes)."""
from __future__ import annotations

from typing import Any

from src.adapters.extract import instrument_info
from src.adapters.nautilus import instrument_kind, quote_obs
from src.common.market import QuoteObs, symbol_of
from src.services.portfolio import OTHER, InstrumentInfo


class NautilusBookMarket:
    def __init__(self, ports: Any) -> None:
        self.ports = ports

    def instrument(self, instrument_id: str) -> InstrumentInfo | None:
        instrument = self.ports.instrument(instrument_id)
        return instrument_info(instrument, instrument_kind(instrument)) if instrument is not None else None

    def quote(self, instrument_id: str) -> QuoteObs | None:
        tick = self.ports.quote(instrument_id)
        return quote_obs(tick) if tick is not None else None

    def instruments_for_symbol(self, symbol: str) -> list[str]:
        return [
            str(inst.id)
            for inst in self.ports.instruments()
            if symbol_of(inst.id) == symbol and instrument_kind(inst) == OTHER
        ]
