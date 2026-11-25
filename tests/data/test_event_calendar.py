from datetime import date

import pytest

from src.data.event_calendar import (
    DEFAULT_EVENTS_PATH,
    DividendEvent,
    EventCalendar,
    early_assignment_plausible,
)

DATA = {
    "macro_events": [
        {"date": "2026-12-09", "name": "FOMC Rate Decision", "severity": "HIGH"},
        {"date": "2026-12-04", "name": "US Non-Farm Payrolls", "severity": "MEDIUM"},
    ],
    "corporate_events": {
        "SPY.ARCA": {"sector": "INDEX_ETF", "ex_dividends": [{"date": "2026-12-18", "amount": 1.78}], "earnings": []},
        "XBI.ARCA": {"sector": "BIOTECH", "ex_dividends": [], "earnings": ["2026-12-01", date(2027, 1, 15)]},
    },
}
AS_OF = date(2026, 11, 24)
DEC_EXPIRY = date(2026, 12, 18)


@pytest.fixture
def calendar():
    return EventCalendar.from_dict(DATA)


def test_earnings_inside_the_holding_window_block_entry(calendar):
    result = calendar.check_entry("XBI.ARCA", as_of=AS_OF, expiry_date=DEC_EXPIRY)
    assert not result.allowed
    assert result.reasons == ["EARNINGS_IN_WINDOW(2026-12-01)"]


@pytest.mark.parametrize(
    "as_of, expiry, allowed",
    [
        (date(2026, 12, 1), DEC_EXPIRY, False),  # Window starts on the earnings date
        (AS_OF, date(2026, 12, 1), False),  # Window ends on the earnings date
        (date(2026, 12, 2), DEC_EXPIRY, True),
        (AS_OF, date(2026, 11, 30), True),
    ],
)
def test_earnings_window_is_inclusive(calendar, as_of, expiry, allowed):
    assert calendar.check_entry("XBI.ARCA", as_of=as_of, expiry_date=expiry).allowed is allowed


def test_in_the_money_short_call_with_thin_extrinsic_blocks_over_ex_dividend(calendar):
    result = calendar.check_entry(
        "SPY.ARCA", as_of=AS_OF, expiry_date=DEC_EXPIRY,
        has_short_call=True, short_call_strike=690.0, spot_price=700.0, short_call_extrinsic=0.5,
    )
    assert not result.allowed
    assert result.reasons == ["EX_DIVIDEND_ASSIGNMENT_RISK(2026-12-18,1.78)"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"has_short_call": False},
        {"has_short_call": True, "short_call_strike": 720.0, "spot_price": 700.0, "short_call_extrinsic": 0.5},  # OTM
        {"has_short_call": True, "short_call_strike": 690.0, "spot_price": 700.0, "short_call_extrinsic": 2.5},  # Rich
    ],
)
def test_ex_dividend_passes_without_plausible_early_assignment(calendar, kwargs):
    assert calendar.check_entry("SPY.ARCA", as_of=AS_OF, expiry_date=DEC_EXPIRY, **kwargs).allowed


def test_ex_dividend_after_expiry_is_ignored(calendar):
    result = calendar.check_entry(
        "SPY.ARCA", as_of=AS_OF, expiry_date=date(2026, 12, 17), has_short_call=True, short_call_extrinsic=0.0
    )
    assert result.allowed


@pytest.mark.parametrize(
    "strike, spot, extrinsic, plausible",
    [
        (690.0, 700.0, 0.5, True),
        (None, 700.0, 0.5, True),  # Unknown moneyness fails closed
        (690.0, None, 0.5, True),
        (690.0, 700.0, None, True),  # Unknown extrinsic fails closed
        (700.0, 700.0, 0.5, False),  # At the money is not in the money
        (690.0, 700.0, 1.78, False),  # Extrinsic equal to the dividend
    ],
)
def test_early_assignment_rule(strike, spot, extrinsic, plausible):
    assert early_assignment_plausible(1.78, strike, spot, extrinsic) is plausible


@pytest.mark.parametrize(
    "as_of, expected",
    [
        (date(2026, 12, 2), "2026-12-04 US Non-Farm Payrolls"),
        (date(2026, 12, 4), "2026-12-04 US Non-Farm Payrolls"),
        (date(2026, 12, 7), "2026-12-09 FOMC Rate Decision"),
        (date(2026, 12, 1), None),  # Three days ahead: outside the window
        (date(2026, 12, 5), None),  # Already past, next one too far
    ],
)
def test_macro_release_window(calendar, as_of, expected):
    result = calendar.check_entry("SPY.ARCA", as_of=as_of, expiry_date=as_of)
    assert result.macro_event_within_2d == expected
    assert result.allowed  # Informational only


def test_symbols_match_exactly_then_by_root(calendar):
    for symbol in ("SPY.ARCA", "SPY", "SPY.SMART"):
        assert calendar.events_for(symbol) is calendar.corporate["SPY.ARCA"]
    assert calendar.events_for("IWM.ARCA") is None


def test_unknown_symbol_has_no_corporate_events(calendar):
    result = calendar.check_entry("IWM.ARCA", as_of=date(2026, 12, 3), expiry_date=DEC_EXPIRY, has_short_call=True)
    assert result.allowed and result.reasons == []
    assert result.macro_event_within_2d == "2026-12-04 US Non-Farm Payrolls"


def test_macro_window_is_configurable():
    cal = EventCalendar.from_dict(DATA, macro_window_days=0)
    assert cal.check_entry("SPY", as_of=date(2026, 12, 3), expiry_date=DEC_EXPIRY).macro_event_within_2d is None


def test_repository_events_file_loads():
    pytest.importorskip("yaml")
    cal = EventCalendar()

    assert DEFAULT_EVENTS_PATH.name == "events.yaml"
    assert cal.events_for("SPY.ARCA").ex_dividends == (DividendEvent(date(2026, 12, 18), 1.78),)
    assert [ev.name for ev in cal.macro_events][0] == "US Non-Farm Payrolls"  # Sorted by date


def test_yaml_native_dates_are_accepted(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "events.yaml"
    path.write_text(
        "macro_events:\n"
        "  - {date: 2026-12-09, name: FOMC}\n"
        "corporate_events:\n"
        "  ABC.ARCA:\n"
        "    earnings: [2026-12-01]\n",
        encoding="utf-8",
    )
    cal = EventCalendar(path=path)
    result = cal.check_entry("ABC.ARCA", as_of=date(2026, 12, 1), expiry_date=date(2026, 12, 9))
    assert result.reasons == ["EARNINGS_IN_WINDOW(2026-12-01)"]


def test_default_gate_calendar_is_the_file_backed_calendar():
    pytest.importorskip("yaml")
    from src.risk.calendar import default_event_calendar

    assert isinstance(default_event_calendar(), EventCalendar)
def test_yaml_event_provider_earnings(calendar):
    pytest.importorskip("yaml")
    from src.data.event_calendar import YamlEventProvider

    provider = YamlEventProvider(calendar)
    assert "2026-12-01" in [d.isoformat() for d in provider.earnings_dates("XBI.ARCA")]


def test_cached_event_provider_earnings_and_dividends():
    from src.data.event_calendar import CachedEventProvider

    data = {
        "SPY.ARCA": {
            "earnings": [],
            "ex_dividends": [{"date": "2026-12-18", "amount": 1.78}],
            "htb_fee_annual": 0.08,
        },
        "XBI.ARCA": {
            "earnings": ["2026-12-01", "2027-01-15"],
            "ex_dividends": [],
            "htb_fee_annual": 0.0,
        },
    }
    provider = CachedEventProvider(data)
    assert provider.earnings_dates("XBI.ARCA") == (date(2026, 12, 1), date(2027, 1, 15))
    divs = provider.ex_dividends("SPY.ARCA")
    assert len(divs) == 1 and divs[0].amount == 1.78
    assert provider.earnings_dates("UNKNOWN") == ()


def test_cached_provider_htb_rate():
    from src.data.event_calendar import CachedEventProvider

    provider = CachedEventProvider({
        "HTB.ARCA": {"htb_fee_annual": 0.25, "earnings": [], "ex_dividends": []},
    })
    assert provider.htb_borrow_rate("HTB.ARCA") == 0.25
    assert provider.htb_borrow_rate("NONE.ARCA") == 0.0


def test_htb_blocks_short_call_above_threshold():
    from src.data.event_calendar import (
        CachedEventProvider,
        htb_blocks_entry,
    )

    provider = CachedEventProvider({
        "HTB.ARCA": {"htb_fee_annual": 0.20, "earnings": [], "ex_dividends": []},
        "CHEAP.ARCA": {"htb_fee_annual": 0.08, "earnings": [], "ex_dividends": []},
    })
    assert htb_blocks_entry(provider, "HTB.ARCA", has_short_call=True, max_htb_rate=0.15)
    assert not htb_blocks_entry(provider, "CHEAP.ARCA", has_short_call=True, max_htb_rate=0.15)
    assert not htb_blocks_entry(provider, "HTB.ARCA", has_short_call=False, max_htb_rate=0.15)


def test_cached_provider_update_symbol_merges():
    from src.data.event_calendar import CachedEventProvider

    provider = CachedEventProvider()
    provider.update_symbol("NEW.ARCA", {"earnings": ["2026-12-15"], "ex_dividends": [], "htb_fee_annual": 0.0})
    assert provider.earnings_dates("NEW.ARCA") == (date(2026, 12, 15),)
