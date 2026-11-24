"""Small helpers for numeric values read from JSON payloads."""
from __future__ import annotations

import math
from typing import Any


def is_number(value: Any) -> bool:
    """True for int/float values; bool is excluded even though it subclasses int."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def finite_or_none(value: Any) -> float | None:
    """value as a float when it is a finite number; NaN, infinities, bools and non-numbers give None."""
    if not is_number(value):
        return None
    try:
        number = float(value)
    except OverflowError:  # Integers too large for a float
        return None
    return number if math.isfinite(number) else None


def round_or_none(value: float | None, ndigits: int) -> float | None:
    return round(value, ndigits) if value is not None else None
