import logging
from unittest import mock

import pytest

from src.common.store import JsonStore
from tests.factories import NOW_NS, seed_healthy_inputs
from tests.fakes import recording_sink, valkey_client


@pytest.fixture
def fake_valkey():
    return valkey_client()


@pytest.fixture
def store(fake_valkey) -> JsonStore:
    return JsonStore(fake_valkey)


@pytest.fixture
def healthy_store(store: JsonStore) -> JsonStore:
    return seed_healthy_inputs(store)


@pytest.fixture
def now_ns() -> int:
    return NOW_NS


@pytest.fixture
def wall_clock() -> mock.Mock:
    """Epoch-ns clock reading NOW_NS; tests move it through return_value."""
    return mock.Mock(return_value=NOW_NS)


@pytest.fixture
def mono_clock() -> mock.Mock:
    """Monotonic clock (seconds); tests move it through return_value."""
    return mock.Mock(return_value=1_000.0)


@pytest.fixture
def mem_sink() -> mock.NonCallableMagicMock:
    return recording_sink()


@pytest.fixture
def recording_log() -> mock.NonCallableMagicMock:
    """Injected `log=` double with the stdlib Logger interface; assertions read its call arguments."""
    return mock.create_autospec(logging.Logger, instance=True)
