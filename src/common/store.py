"""JSON-over-Valkey access shared by actors, services, risk components and the dashboard.

JsonStore wraps any redis-py compatible client (get/set/delete/lpush/ltrim/lrange/llen/scan_iter/pipeline).
"""
from __future__ import annotations

import json
from typing import Any


def safe_json_loads(raw: Any) -> Any | None:
    """Decodes a JSON document from bytes or text; None when missing or not valid JSON."""
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def safe_json_dict(raw: Any) -> dict[str, Any] | None:
    """Like safe_json_loads, but only JSON objects are accepted."""
    value = safe_json_loads(raw)
    return value if isinstance(value, dict) else None


class _JsonWriter:
    """Write operations shared by the store and its pipelined batches."""

    _target: Any

    def set_text(self, key: str, text: str, ttl_ms: int | None = None) -> None:
        if ttl_ms is None:
            self._target.set(key, text)
        else:
            self._target.set(key, text, px=int(ttl_ms))

    def set_json(self, key: str, value: Any, ttl_ms: int | None = None) -> None:
        self.set_text(key, json.dumps(value), ttl_ms)

    def push_capped(self, key: str, value: Any, max_len: int) -> None:
        """LPUSHes one entry (JSON-encoded unless already text) and trims the list to max_len entries."""
        self._target.lpush(key, value if isinstance(value, str) else json.dumps(value))
        self._target.ltrim(key, 0, max_len - 1)

    def delete(self, key: str) -> None:
        self._target.delete(key)


class JsonBatch(_JsonWriter):
    """Writes buffered in a pipeline until execute()."""

    def __init__(self, pipeline: Any) -> None:
        self._target = pipeline

    def execute(self) -> Any:
        return self._target.execute()


class JsonStore(_JsonWriter):
    connected = True

    def __init__(self, client: Any) -> None:
        self.client = client
        self._target = client

    def get_text(self, key: str) -> str | None:
        raw = self.client.get(key)
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return bytes(raw).decode("utf-8", errors="replace")
        return str(raw)

    def get_json_value(self, key: str) -> Any | None:
        return safe_json_loads(self.client.get(key))

    def get_json_or_none(self, key: str) -> dict[str, Any] | None:
        return safe_json_dict(self.client.get(key))

    def get_json(self, key: str) -> dict[str, Any]:
        """JSON object stored at key; {} when missing, invalid or not an object."""
        value = self.get_json_or_none(key)
        return value if value is not None else {}

    def get_json_list(self, key: str, count: int) -> list[dict[str, Any]]:
        """Up to count JSON objects from the head of a list (newest first); invalid entries are skipped."""
        if count <= 0:
            return []
        parsed = (safe_json_dict(item) for item in self.client.lrange(key, 0, count - 1))
        return [obj for obj in parsed if obj is not None]

    def list_len(self, key: str) -> int:
        return int(self.client.llen(key))

    def list_keys(self, pattern: str) -> list[str]:
        """Sorted key names matching a glob pattern, found with SCAN (never the blocking KEYS)."""
        found = {
            raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            for raw in self.client.scan_iter(match=pattern)
        }
        return sorted(found)

    def scan_json(self, pattern: str) -> dict[str, dict[str, Any]]:
        """Non-empty JSON objects stored under keys matching pattern."""
        out: dict[str, dict[str, Any]] = {}
        for key in self.list_keys(pattern):
            payload = self.get_json_or_none(key)
            if payload:
                out[key] = payload
        return out

    def batch(self) -> JsonBatch:
        return JsonBatch(self.client.pipeline())


class _NullClient:
    """Client and pipeline stand-in for NullStore: nothing is stored."""

    def get(self, key: str) -> None:
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:
        return None

    def delete(self, *keys: str) -> int:
        return 0

    def lpush(self, *args: Any) -> int:
        return 0

    def ltrim(self, *args: Any) -> bool:
        return True

    def lrange(self, *args: Any) -> list:
        return []

    def llen(self, key: str) -> int:
        return 0

    def scan_iter(self, match: str | None = None, count: int | None = None):
        return iter(())

    def pipeline(self, *args: Any, **kwargs: Any) -> "_NullClient":
        return self

    def execute(self) -> list:
        return []


class NullStore(JsonStore):
    """Store used when no Valkey URL is configured: reads are empty and writes are dropped."""

    connected = False

    def __init__(self) -> None:
        super().__init__(_NullClient())


def make_store(url: str | None) -> JsonStore:
    """JsonStore for a redis:// URL, or a NullStore when url is empty."""
    if not url:
        return NullStore()
    import redis  # Lazy: pure modules and their tests never need the client library

    return JsonStore(redis.from_url(url))
