"""Duck-typed extraction from Nautilus configs and instruments (no Nautilus import, testable with stand-ins)."""
from __future__ import annotations

import dataclasses
from typing import Any, TypeVar

from src.common.market import OptionSpec
from src.services.portfolio import OPTION, OPTION_SPREAD, InstrumentInfo, PositionView

T = TypeVar("T")


def id_str(value: Any) -> str | None:
    return None if value is None else str(value)


def plain_value(value: Any) -> Any:
    """Primitives pass through, sequences become tuples, anything else (e.g. InstrumentId) becomes its string."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(plain_value(item) for item in value)
    return str(value)


def plain_settings(config: Any, settings_cls: type[T], **overrides: Any) -> T:
    """Builds a settings dataclass from same-named config attributes (missing ones keep their defaults)."""
    values: dict[str, Any] = {}
    for f in dataclasses.fields(settings_cls):
        if f.name in overrides:
            values[f.name] = overrides[f.name]
        elif hasattr(config, f.name):
            values[f.name] = plain_value(getattr(config, f.name))
    return settings_cls(**values)


def _open_interest(instrument: Any) -> float | None:
    info = getattr(instrument, "info", None) or {}
    raw = info.get("open_interest") if isinstance(info, dict) else None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def option_spec_from_instrument(instrument: Any) -> OptionSpec:
    """OptionSpec of an option contract: id, underlying, strike, kind, expiry, multiplier, open interest."""
    return OptionSpec(
        instrument_id=str(instrument.id),
        underlying=str(getattr(instrument, "underlying", "") or ""),
        strike=float(instrument.strike_price),
        is_call="CALL" in str(instrument.option_kind).upper(),
        expiry_ns=int(getattr(instrument, "expiration_ns", 0) or 0),
        multiplier=float(getattr(instrument, "multiplier", 100.0) or 100.0),
        open_interest=_open_interest(instrument),
    )


def instrument_info(instrument: Any, kind: str) -> InstrumentInfo:
    """Portfolio view of an instrument; options default to a 100 multiplier, other instruments to 1."""
    if kind == OPTION:
        spec = option_spec_from_instrument(instrument)
        return InstrumentInfo(OPTION, spec.multiplier, spec)
    default = 100.0 if kind == OPTION_SPREAD else 1.0
    return InstrumentInfo(kind, float(getattr(instrument, "multiplier", default) or default))


def position_view(position: Any) -> PositionView:
    return PositionView(
        instrument_id=str(position.instrument_id),
        signed_qty=float(position.signed_qty),
        avg_px_open=float(position.avg_px_open),
        realized_pnl=float(position.realized_pnl),
    )
