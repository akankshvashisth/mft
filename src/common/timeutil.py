"""Time conversions and New York session helpers."""
from __future__ import annotations

from datetime import date, datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import QuantLib as ql

from src.common.constants import DAYS_PER_YEAR, NS_PER_DAY, NS_PER_SEC, NS_PER_YEAR, NY_TZ_NAME

NY_TZ = ZoneInfo(NY_TZ_NAME)
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)
NYSE_CALENDAR = ql.UnitedStates(ql.UnitedStates.NYSE)


def age_secs(ts_ns: float, now_ns: float) -> float:
    return (now_ns - ts_ns) / NS_PER_SEC


def years_until(expiry_ns: float, now_ns: float, floor: float = 1e-4) -> float:
    """Year fraction (365.25-day years) from now_ns to expiry_ns, never below floor."""
    return max((expiry_ns - now_ns) / NS_PER_YEAR, floor)


def days_until(expiry_ns: float, now_ns: float) -> float:
    return (expiry_ns - now_ns) / NS_PER_DAY


def days_to_years(days: float) -> float:
    return days / DAYS_PER_YEAR


def years_to_days(years: float) -> float:
    return years * DAYS_PER_YEAR


def ny_datetime(ts_ns: float) -> datetime:
    return datetime.fromtimestamp(ts_ns / NS_PER_SEC, tz=NY_TZ)


def ny_date(ts_ns: float) -> date:
    """New York calendar (trading) date of an epoch-ns timestamp."""
    return ny_datetime(ts_ns).date()


def utc_date(ts_ns: float) -> date:
    return datetime.fromtimestamp(ts_ns / NS_PER_SEC, tz=timezone.utc).date()


def iso_utc(ts_ns: float) -> str:
    return datetime.fromtimestamp(ts_ns / NS_PER_SEC, tz=timezone.utc).isoformat()


def is_nyse_trading_day(day: date) -> bool:
    """Weekday that is not an NYSE holiday (QuantLib's UnitedStates.NYSE calendar)."""
    return NYSE_CALENDAR.isBusinessDay(ql.Date(day.day, day.month, day.year))


def is_regular_session(ts_ns: float) -> bool:
    """True on NYSE trading days between 09:30 and 16:00 New York time (both inclusive)."""
    local = ny_datetime(ts_ns)
    return RTH_OPEN <= local.time() <= RTH_CLOSE and is_nyse_trading_day(local.date())
