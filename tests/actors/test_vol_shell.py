import math

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")
pytestmark = pytest.mark.nautilus

import src.actors.vol_actor as vol_module  # noqa: E402
from src.common import keys  # noqa: E402
from src.quant.svi import svi_implied_vol  # noqa: E402
from tests.factories import NOW_NS, SVI_PARAMS  # noqa: E402
from tests.fakes import FakeActorPorts  # noqa: E402
from tests.nautilus_objects import iid, quote_tick, timer_event  # noqa: E402

STRIKES = [650.0 + 5 * i for i in range(11)] + [705.0 + 5 * i for i in range(10)]
IS_CALLS = [False] * 11 + [True] * 10
OPTION_IDS = [f"SPY 261224{'C' if c else 'P'}{int(k * 1000):08d}.SMART" for k, c in zip(STRIKES, IS_CALLS)]


def smile_solver(prices, S, K, t, r, flags):
    forward = S[0] * math.exp(r[0] * t[0])
    return svi_implied_vol(np.log(K / forward), t[0], SVI_PARAMS)


def exact_fitter(log_m, ivs, weights, ttm, x0_warm=None):
    return svi_implied_vol(log_m, ttm, SVI_PARAMS), np.array(SVI_PARAMS), 0.0


@pytest.fixture
def actor(monkeypatch, store):
    monkeypatch.setattr(vol_module, "make_store", lambda url: store)
    actor = vol_module.FastVolActor(
        vol_module.FastVolActorConfig(
            underlying_id=iid("SPY.ARCA"),
            option_ids=tuple(iid(o) for o in OPTION_IDS),
            strikes=tuple(STRIKES),
            is_calls=tuple(IS_CALLS),
            strike_min=640.0,
            strike_max=760.0,
        )
    )
    actor.core.iv_solver = smile_solver
    actor.core.svi_fitter = exact_fitter
    actor.core.arb_validator = lambda log_m, ttm, ivs: True
    actor.ports = FakeActorPorts(now_ns=NOW_NS + 7)
    actor.wall_clock_ns = lambda: NOW_NS
    return actor


def test_start_keeps_the_configured_slice_without_cached_options(actor):
    actor.on_start()
    assert actor.ports.quote_subscriptions == ["SPY.ARCA", *OPTION_IDS]
    assert actor.ports.timers == [("fast_vol_fit", 15, actor.on_timer)]
    assert actor.ports.infos == ["FastVolActor watching SPY.ARCA + 21 standard 100-mult options"]


def test_quotes_to_published_surface(actor, store):
    actor.on_start()
    actor.on_quote(quote_tick("SPY.ARCA", 699.95, 700.05, NOW_NS - 1_000_000_000))
    for option_id in OPTION_IDS:
        actor.on_quote(quote_tick(option_id, 1.95, 2.05, NOW_NS - 1_000_000_000))

    actor.on_timer(timer_event(NOW_NS))

    payload = store.get_json_value(keys.surface_key("SPY.ARCA"))
    assert (payload["quotes"], payload["arb_free"], payload["ts_ns"], payload["event_ts_ns"]) == (21, True, NOW_NS, NOW_NS)
    ((name, snapshot),) = actor.ports.published
    assert name == "SviSliceParams"
    assert (snapshot.underlying_id, snapshot.quote_count, snapshot.arb_free) == (iid("SPY.ARCA"), 21, True)
    assert snapshot.atm_iv == pytest.approx(payload["atm_iv"], abs=1e-4)
    assert (snapshot.ts_event, snapshot.ts_init) == (NOW_NS, NOW_NS + 7)


def test_stale_spot_publishes_nothing(actor):
    actor.on_timer(timer_event(NOW_NS))
    assert actor.ports.published == []
    assert actor.ports.warnings == ["FastVolActor underlying stale/missing (spot=None, age=9999.0s)"]
