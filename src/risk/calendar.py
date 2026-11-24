"""Event-calendar seam for the pre-trade gate: a protocol plus a lazily imported default."""
from __future__ import annotations

from datetime import date
from typing import Protocol


class EventCheckLike(Protocol):
    allowed: bool
    reasons: list[str]
    macro_event_within_2d: str | None


class EventCalendarLike(Protocol):
    def check_entry(
        self,
        symbol: str,
        as_of: date,
        expiry_date: date,
        has_short_call: bool = False,
        short_call_strike: float | None = None,
        spot_price: float | None = None,
        short_call_extrinsic: float | None = None,
    ) -> EventCheckLike: ...


def default_event_calendar() -> EventCalendarLike:
    """File-backed EventCalendar (config/events.yaml), imported only when no calendar is injected."""
    from src.data.event_calendar import EventCalendar

    return EventCalendar()
