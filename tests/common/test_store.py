import json
from unittest import mock

from src.common.store import JsonStore, NullStore, make_store, safe_json_dict, safe_json_loads


def test_safe_json_loads_handles_bytes_text_and_garbage():
    assert safe_json_loads(b'{"a": 1}') == {"a": 1}
    assert safe_json_loads("[1, 2]") == [1, 2]
    assert safe_json_loads(None) is None
    assert safe_json_loads(b"not json") is None
    assert safe_json_loads(b"\xff\xfe") is None


def test_safe_json_dict_only_accepts_objects():
    assert safe_json_dict(b'{"a": 1}') == {"a": 1}
    assert safe_json_dict(b"[1, 2]") is None
    assert safe_json_dict(b"3") is None


def test_reads_return_empty_values_for_missing_or_invalid_keys(store, fake_valkey):
    fake_valkey.set("obj", b'{"x": 1}')
    fake_valkey.set("list", b"[1]")
    fake_valkey.set("bad", b"{")

    assert store.get_json("obj") == {"x": 1}
    assert store.get_json("missing") == {}
    assert store.get_json("list") == {}
    assert store.get_json_or_none("bad") is None
    assert store.get_json_value("list") == [1]
    assert store.get_text("obj") == '{"x": 1}'
    assert store.get_text("missing") is None


def test_set_json_sets_a_ttl_only_when_given(store, fake_valkey):
    store.set_json("plain", {"v": 1})
    store.set_json("expiring", {"v": 2}, ttl_ms=5_000)

    assert json.loads(fake_valkey.get("plain")) == {"v": 1}
    assert fake_valkey.pttl("plain") == -1  # No expiry
    assert 4_000 < fake_valkey.pttl("expiring") <= 5_000


def test_push_capped_keeps_newest_first_and_trims(store):
    for i in range(5):
        store.push_capped("hist", {"i": i}, max_len=3)

    assert [row["i"] for row in store.get_json_list("hist", 10)] == [4, 3, 2]
    assert store.list_len("hist") == 3


def test_push_capped_stores_text_unchanged(store, fake_valkey):
    store.push_capped("hist", '{"raw": true}', max_len=2)

    assert fake_valkey.lrange("hist", 0, -1) == [b'{"raw": true}']


def test_get_json_list_skips_invalid_entries_and_respects_count(store, fake_valkey):
    fake_valkey.lpush("hist", b'{"n": 1}')
    fake_valkey.lpush("hist", b"broken")
    fake_valkey.lpush("hist", b'{"n": 3}')

    assert store.get_json_list("hist", 10) == [{"n": 3}, {"n": 1}]
    assert store.get_json_list("hist", 1) == [{"n": 3}]
    assert store.get_json_list("hist", 0) == []


def test_list_keys_uses_scan_and_sorts(store, fake_valkey):
    for key in ("mft:surface:QQQ.NASDAQ", "mft:surface:SPY.ARCA", "mft:rv:SPY.ARCA"):
        fake_valkey.set(key, b"{}")

    with mock.patch.object(fake_valkey, "keys", side_effect=AssertionError("KEYS blocks Valkey")):
        assert store.list_keys("mft:surface:*") == ["mft:surface:QQQ.NASDAQ", "mft:surface:SPY.ARCA"]


def test_scan_json_skips_empty_and_invalid_payloads(store, fake_valkey):
    fake_valkey.set("mft:surface:A", b'{"spot": 1}')
    fake_valkey.set("mft:surface:B", b"{}")
    fake_valkey.set("mft:surface:C", b"oops")

    assert store.scan_json("mft:surface:*") == {"mft:surface:A": {"spot": 1}}


def test_batch_writes_nothing_until_execute(store, fake_valkey):
    batch = store.batch()
    batch.set_text("flag", "1", ttl_ms=1_000)
    batch.set_json("status", {"ok": True})
    batch.push_capped("hist", {"n": 1}, max_len=5)

    assert fake_valkey.get("flag") is None
    batch.execute()

    assert fake_valkey.get("flag") == b"1"
    assert 0 < fake_valkey.pttl("flag") <= 1_000
    assert store.get_json("status") == {"ok": True}
    assert store.get_json_list("hist", 5) == [{"n": 1}]


def test_delete_removes_values(store, fake_valkey):
    store.set_text("k", "v")
    store.delete("k")

    assert fake_valkey.get("k") is None


def test_make_store_without_url_returns_a_null_store():
    store = make_store("")

    assert isinstance(store, NullStore)
    assert store.connected is False
    store.set_json("k", {"v": 1})
    store.push_capped("l", {"v": 1}, max_len=3)
    store.batch().execute()
    assert store.get_json("k") == {}
    assert store.get_text("k") is None
    assert store.get_json_list("l", 5) == []
    assert store.list_keys("*") == []


def test_connected_store_reports_connected(fake_valkey):
    assert JsonStore(fake_valkey).connected is True
