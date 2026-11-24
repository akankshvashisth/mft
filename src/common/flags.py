"""Parsing of on/off flags stored as text (e.g. the manual kill switch)."""
from __future__ import annotations

from typing import Any

TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def is_truthy_flag(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    return str(raw).strip().lower() in TRUTHY_VALUES
