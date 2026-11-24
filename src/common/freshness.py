"""Payload timestamp and age helpers (payloads carry epoch-nanosecond timestamps)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.common.constants import MIN_EPOCH_NS, NS_PER_SEC
from src.common.numeric import is_number

# Field sets per consumer, in priority order
WALL_TS_FIELDS: tuple[str, ...] = ("ts_ns",)
DAEMON_TS_FIELDS: tuple[str, ...] = ("ts_ns", "ts_event", "ts_init", "updated_at_ns")
DISPLAY_TS_FIELDS: tuple[str, ...] = ("ts_ns", "event_ts_ns", "ts_event")

# Clock skew tolerated for timestamps slightly in the future; further ahead the timestamp is unusable
MAX_FUTURE_SKEW_SECS: float = 900.0


def payload_timestamp_ns(
    payload: Mapping[str, Any] | None, fields: Sequence[str] = WALL_TS_FIELDS
) -> float | None:
    """First field holding an epoch-ns timestamp; None when there is none."""
    if not payload:
        return None
    for field in fields:
        value = payload.get(field)
        if is_number(value) and value > MIN_EPOCH_NS:
            return float(value)
    return None


def payload_age_secs(
    payload: Mapping[str, Any] | None,
    now_ns: float,
    fields: Sequence[str] = WALL_TS_FIELDS,
    *,
    max_future_secs: float = MAX_FUTURE_SKEW_SECS,
) -> float | None:
    """Payload age in seconds.

    None when no field holds an epoch-ns timestamp or the timestamp is more than max_future_secs
    ahead of now_ns; smaller future skew reads as age 0.
    """
    ts_ns = payload_timestamp_ns(payload, fields)
    if ts_ns is None:
        return None
    age = (now_ns - ts_ns) / NS_PER_SEC
    if age < 0.0:
        return 0.0 if -age <= max_future_secs else None
    return age


def is_fresh(
    payload: Mapping[str, Any] | None,
    now_ns: float,
    max_age_secs: float,
    fields: Sequence[str] = WALL_TS_FIELDS,
) -> bool:
    age = payload_age_secs(payload, now_ns, fields)
    return age is not None and age <= max_age_secs
