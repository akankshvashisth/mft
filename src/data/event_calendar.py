"""Minimal corporate-event and macro-release calendar backed by config/events.yaml.

Entry rules used by the pre-trade gate:
- Earnings inside [as_of, expiry] block the entry (EARNINGS_IN_WINDOW).
- An ex-dividend date inside [as_of, expiry] blocks an entry that sells a call when early assignment is plausible:
  the call is in the money (or its moneyness is unknown) and its extrinsic value is below the dividend (or unknown).
- A macro release within macro_window_days of as_of is informational; the gate turns it into a warning.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.common.market import symbol_of

DEFAULT_EVENTS_PATH = Path(__file__).resolve().parents[2] / "config" / "events.yaml"


@dataclass(frozen=True)
class MacroEvent:
    date: date
    name: str
    severity: str = ""


@dataclass(frozen=True)
class DividendEvent:
    date: date
    amount: float


@dataclass(frozen=True)
class CorporateEvents:
    sector: str = ""
    htb_fee_annual: float = 0.0
    ex_dividends: tuple[DividendEvent, ...] = ()
    earnings: tuple[date, ...] = ()


@dataclass
class EventCheckResult:
    allowed: bool = True
    reasons: list[str] = field(default_factory=list)
    macro_event_within_2d: str | None = None   # "YYYY-MM-DD Name" of the nearest upcoming release


def as_date(value: Any) -> date:
    """date from a date, datetime or ISO "YYYY-MM-DD" string (YAML may yield any of them)."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _event_date(item: Any) -> date:
    return as_date(item["date"] if isinstance(item, Mapping) else item)


def parse_macro_events(items: Any) -> tuple[MacroEvent, ...]:
    events = [
        MacroEvent(date=_event_date(item), name=str(item.get("name", "")), severity=str(item.get("severity", "")))
        for item in (items or [])
        if isinstance(item, Mapping) and item.get("date")
    ]
    return tuple(sorted(events, key=lambda ev: ev.date))


def parse_corporate_events(items: Any) -> dict[str, CorporateEvents]:
    out: dict[str, CorporateEvents] = {}
    for symbol, spec in (items or {}).items():
        spec = spec or {}
        dividends = tuple(
            sorted(
                (
                    DividendEvent(date=_event_date(div), amount=float(div.get("amount", 0.0)))
                    for div in (spec.get("ex_dividends") or [])
                    if isinstance(div, Mapping)
                ),
                key=lambda div: div.date,
            )
        )
        earnings = tuple(sorted(_event_date(item) for item in (spec.get("earnings") or [])))
        out[str(symbol)] = CorporateEvents(
            sector=str(spec.get("sector", "")),
            htb_fee_annual=float(spec.get("htb_fee_annual", 0.0) or 0.0),
            ex_dividends=dividends,
            earnings=earnings,
        )
    return out


def load_events_file(path: str | Path) -> dict[str, Any]:
    import yaml  # Lazy: only the file-backed calendar needs PyYAML

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def early_assignment_plausible(
    dividend: float,
    short_call_strike: float | None,
    spot_price: float | None,
    short_call_extrinsic: float | None,
) -> bool:
    """Short call worth exercising before the ex-date: in the money and extrinsic below the dividend.

    Unknown moneyness or extrinsic value counts as plausible (fail closed).
    """
    in_the_money = short_call_strike is None or spot_price is None or spot_price > short_call_strike
    cheap_time_value = short_call_extrinsic is None or short_call_extrinsic < dividend
    return in_the_money and cheap_time_value


class EventCalendar:
    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        path: str | Path | None = None,
        macro_window_days: int = 2,
    ) -> None:
        if data is None:
            data = load_events_file(path or DEFAULT_EVENTS_PATH)
        self.macro_window_days = int(macro_window_days)
        self.macro_events = parse_macro_events(data.get("macro_events"))
        self.corporate = parse_corporate_events(data.get("corporate_events"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], macro_window_days: int = 2) -> "EventCalendar":
        return cls(data=data, macro_window_days=macro_window_days)

    def events_for(self, symbol: str) -> CorporateEvents | None:
        """Corporate events by exact symbol ("SPY.ARCA"), else by root symbol ("SPY")."""
        events = self.corporate.get(str(symbol))
        if events is not None:
            return events
        root = symbol_of(symbol)
        for key, candidate in self.corporate.items():
            if symbol_of(key) == root:
                return candidate
        return None

    def next_macro_event(self, as_of: date) -> MacroEvent | None:
        """Earliest macro release in [as_of, as_of + macro_window_days]."""
        horizon = as_of + timedelta(days=self.macro_window_days)
        for event in self.macro_events:
            if as_of <= event.date <= horizon:
                return event
        return None

    def check_entry(
        self,
        symbol: str,
        as_of: date,
        expiry_date: date,
        has_short_call: bool = False,
        short_call_strike: float | None = None,
        spot_price: float | None = None,
        short_call_extrinsic: float | None = None,
    ) -> EventCheckResult:
        reasons: list[str] = []
        events = self.events_for(symbol)
        if events is not None:
            for earnings_date in events.earnings:
                if as_of <= earnings_date <= expiry_date:
                    reasons.append(f"EARNINGS_IN_WINDOW({earnings_date.isoformat()})")
            if has_short_call:
                for div in events.ex_dividends:
                    if as_of <= div.date <= expiry_date and early_assignment_plausible(
                        div.amount, short_call_strike, spot_price, short_call_extrinsic
                    ):
                        reasons.append(f"EX_DIVIDEND_ASSIGNMENT_RISK({div.date.isoformat()},{div.amount:.2f})")

        macro = self.next_macro_event(as_of)
        return EventCheckResult(
            allowed=not reasons,
            reasons=reasons,
            macro_event_within_2d=f"{macro.date.isoformat()} {macro.name}" if macro is not None else None,
        )
