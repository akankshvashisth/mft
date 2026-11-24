"""Append-only JSON-lines record sinks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class RecordSink(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...


class JsonlSink:
    """Appends one JSON document per line; the parent directory is created on first write."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
