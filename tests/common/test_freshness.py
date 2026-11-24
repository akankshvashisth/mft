import pytest

from src.common.constants import NS_PER_SEC
from src.common.freshness import (
    DAEMON_TS_FIELDS,
    DISPLAY_TS_FIELDS,
    is_fresh,
    payload_age_secs,
    payload_timestamp_ns,
)
from tests.factories import NOW_NS, ago


def test_timestamp_uses_the_first_valid_field_in_priority_order():
    payload = {"ts_event": ago(10), "ts_ns": ago(30)}

    assert payload_timestamp_ns(payload, DAEMON_TS_FIELDS) == float(ago(30))
    assert payload_timestamp_ns({"event_ts_ns": ago(5)}, DISPLAY_TS_FIELDS) == float(ago(5))


def test_timestamp_ignores_values_that_are_not_epoch_nanoseconds():
    assert payload_timestamp_ns({"ts_ns": 12_345, "ts_event": ago(10)}, DAEMON_TS_FIELDS) == float(ago(10))
    assert payload_timestamp_ns({"ts_ns": True}) is None
    assert payload_timestamp_ns({"ts_ns": "123"}) is None
    assert payload_timestamp_ns({}) is None
    assert payload_timestamp_ns(None) is None


def test_age_is_measured_in_seconds():
    assert payload_age_secs({"ts_ns": ago(30)}, NOW_NS) == pytest.approx(30.0)


def test_small_future_skew_reads_as_zero_age_but_far_future_is_unusable():
    assert payload_age_secs({"ts_ns": NOW_NS + 100 * NS_PER_SEC}, NOW_NS) == 0.0
    assert payload_age_secs({"ts_ns": NOW_NS + 900 * NS_PER_SEC}, NOW_NS) == 0.0
    assert payload_age_secs({"ts_ns": NOW_NS + 901 * NS_PER_SEC}, NOW_NS) is None


def test_missing_timestamp_has_no_age():
    assert payload_age_secs({"spot": 1.0}, NOW_NS) is None


def test_is_fresh_includes_the_age_limit():
    assert is_fresh({"ts_ns": ago(120)}, NOW_NS, 120.0) is True
    assert is_fresh({"ts_ns": ago(121)}, NOW_NS, 120.0) is False
    assert is_fresh({}, NOW_NS, 120.0) is False
    assert is_fresh({"ts_ns": NOW_NS + 2_000 * NS_PER_SEC}, NOW_NS, 120.0) is False
