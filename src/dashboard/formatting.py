"""Display formatting of payload values (actors publish null when a value is unavailable)."""
from __future__ import annotations

from typing import Any

from src.common.numeric import is_number


def fmt(val: Any, spec: str, na: str = "N/A") -> str:
    """Formats numeric payload fields; anything else (None, bool, text) shows `na`."""
    return format(val, spec) if is_number(val) else na


def pct_or_none(val: Any, digits: int = 3) -> float | None:
    """Fraction -> percent rounded to `digits`, or None for non-numeric values."""
    return round(float(val) * 100, digits) if is_number(val) else None
