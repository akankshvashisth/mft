from datetime import date

import pytest

from src.common.constants import NS_PER_DAY, NS_PER_SEC
from src.common.timeutil import (
    days_to_years,
    days_until,
    is_nyse_trading_day,
    is_regular_session,
    iso_utc,
    ny_date,
    utc_date,
    years_to_days,
    years_until,
)
from tests.factories import utc_ns


def test_ny_date_differs_from_utc_date_late_in_the_new_york_evening():
    ts = utc_ns(2026, 11, 25, 3, 0)  # 22:00 New York on the 24th (EST, UTC-5)

    assert ny_date(ts) == date(2026, 11, 24)
    assert utc_date(ts) == date(2026, 11, 25)


def test_ny_date_follows_daylight_saving_time():
    assert ny_date(utc_ns(2026, 7, 1, 3, 59, 59)) == date(2026, 6, 30)  # 23:59:59 EDT (UTC-4)
    assert ny_date(utc_ns(2026, 7, 1, 4, 0, 0)) == date(2026, 7, 1)


@pytest.mark.parametrize(
    "ts, expected",
    [
        (utc_ns(2026, 11, 24, 14, 29, 59), False),  # 09:29:59 New York
        (utc_ns(2026, 11, 24, 14, 30, 0), True),    # 09:30:00 open
        (utc_ns(2026, 11, 24, 21, 0, 0), True),     # 16:00:00 close is included
        (utc_ns(2026, 11, 24, 21, 0, 1), False),
        (utc_ns(2026, 11, 28, 15, 0, 0), False),    # Saturday
        (utc_ns(2026, 11, 26, 15, 0, 0), False),    # Thanksgiving: NYSE holiday
        (utc_ns(2026, 7, 1, 13, 29, 59), False),    # 09:29:59 EDT
        (utc_ns(2026, 7, 1, 13, 30, 0), True),      # 09:30:00 EDT
    ],
)
def test_is_regular_session(ts, expected):
    assert is_regular_session(ts) is expected


@pytest.mark.parametrize(
    "day, expected",
    [
        (date(2026, 11, 24), True),
        (date(2026, 11, 26), False),  # Thanksgiving
        (date(2026, 7, 3), False),  # Independence Day observed on the Friday
        (date(2026, 11, 28), False),  # Saturday
    ],
)
def test_nyse_trading_days(day, expected):
    assert is_nyse_trading_day(day) is expected


def test_years_until_uses_365_25_day_years_and_a_floor():
    now = utc_ns(2026, 1, 1)

    assert years_until(now + int(365.25 * NS_PER_DAY), now) == pytest.approx(1.0)
    assert years_until(now - NS_PER_DAY, now) == 1e-4
    assert years_until(now, now, floor=0.5) == 0.5


def test_day_and_year_conversions():
    now = utc_ns(2026, 1, 1)

    assert days_until(now + 36 * 3_600 * NS_PER_SEC, now) == pytest.approx(1.5)
    assert days_to_years(365.25) == pytest.approx(1.0)
    assert years_to_days(0.5) == pytest.approx(182.625)


def test_iso_utc():
    assert iso_utc(0) == "1970-01-01T00:00:00+00:00"
    assert iso_utc(utc_ns(2026, 11, 24, 15, 0)) == "2026-11-24T15:00:00+00:00"
