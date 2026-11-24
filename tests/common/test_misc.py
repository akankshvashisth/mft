import json

import pytest

from src.common.flags import is_truthy_flag
from src.common.jsonl import JsonlSink
from src.common.market import BarObs, LatestValue, QuoteObs, symbol_of
from src.common.numeric import finite_or_none, is_number, round_or_none
from tests.factories import NOW_NS, ago


@pytest.mark.parametrize("raw", [b"1", "TRUE ", "yes", b"On"])
def test_truthy_flags(raw):
    assert is_truthy_flag(raw) is True


@pytest.mark.parametrize("raw", [None, b"", b"0", "false", "off", "2"])
def test_non_truthy_flags(raw):
    assert is_truthy_flag(raw) is False


def test_jsonl_sink_creates_its_directory_on_first_write(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    sink = JsonlSink(path)
    assert not path.parent.exists()

    sink.write({"n": 1})
    sink.write({"n": 2})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"n": 1}, {"n": 2}]


def test_latest_value_keeps_the_newest_observation():
    level = LatestValue()

    assert level.update(10.0, 100) is True
    assert level.update(11.0, 100) is True   # Same timestamp still replaces by default
    assert level.update(9.0, 99) is False
    assert (level.value, level.ts_ns) == (11.0, 100)


def test_latest_value_strictly_newer_rejects_equal_timestamps():
    level = LatestValue()

    assert level.update(1.0, 5, strictly_newer=True) is True
    assert level.update(2.0, 5, strictly_newer=True) is False
    assert level.value == 1.0


def test_latest_value_fresh():
    level = LatestValue(20.0, ago(30))

    assert level.fresh(NOW_NS, 60.0) == 20.0
    assert level.fresh(NOW_NS, 10.0) is None
    assert LatestValue().fresh(NOW_NS, 60.0) is None


def test_quote_mid_and_bar_helpers():
    assert QuoteObs("X", 1.0, 1.2, 0).mid == pytest.approx(1.1)
    assert QuoteObs("X", 1.2, 1.0, 0).mid is None

    bar = BarObs("X", 1.0, 2.0, 0.5, 1.5, 10.0, 7, aggregation="DAY")
    assert bar.is_daily is True
    assert bar.ohlcv_row() == {"ts_ns": 7, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
    assert BarObs("X", 1, 1, 1, 1, 0, 0).is_daily is False


def test_symbol_of():
    assert symbol_of("SPY.ARCA") == "SPY"
    assert symbol_of("^VIX.CBOE") == "^VIX"
    assert symbol_of("BRK.B.NYSE") == "BRK.B"
    assert symbol_of("SPY") == "SPY"


def test_numeric_helpers():
    assert is_number(1) and is_number(1.5)
    assert not is_number(True) and not is_number("1") and not is_number(None)
    assert round_or_none(1.23456, 2) == 1.23
    assert round_or_none(None, 2) is None


def test_finite_or_none():
    assert finite_or_none(2) == 2.0 and isinstance(finite_or_none(2), float)
    assert finite_or_none(-1.5) == -1.5
    for bad in (float("nan"), float("inf"), float("-inf"), 10**400, True, "1", None):
        assert finite_or_none(bad) is None
