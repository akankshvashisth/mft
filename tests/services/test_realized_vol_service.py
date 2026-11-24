import numpy as np
import pandas as pd
import pytest

from src.common import keys
from src.common.market import BarObs, QuoteObs
from src.services.realized_vol import DailyHistoryBook, RealizedVolService, RealizedVolSettings
from tests.factories import NOW_NS, daily_ohlc_frame, utc_ns

SPY, QQQ, VIXY = "SPY.ARCA", "QQQ.NASDAQ", "VIXY.ARCA"
# 30 daily rows ending Monday 2026-11-23 16:00 New York, the session before NOW
FIRST_TS = utc_ns(2026, 10, 25, 21)


def history(n=30, seed=7, scale=1.0):
    df = daily_ohlc_frame(n=n, seed=seed, first_ts_ns=FIRST_TS)
    df[["open", "high", "low", "close"]] *= scale
    return df


def inverse(df):
    """Prices 1/x: every log return flips sign."""
    out = df.copy()
    out["open"], out["close"] = 1e4 / df["open"], 1e4 / df["close"]
    out["high"], out["low"] = 1e4 / df["low"], 1e4 / df["high"]
    return out


@pytest.fixture
def make_service(store, recording_log):
    def make(universe=(SPY, QQQ, VIXY), frames=None):
        service = RealizedVolService(
            RealizedVolSettings(primary_id=SPY, universe_ids=universe), store=store, log=recording_log
        )
        frames = frames if frames is not None else {}
        service.warm_start(frames.get)
        return service

    return make


def test_warm_start_loads_available_histories(make_service):
    service = make_service(frames={SPY: history(), QQQ: pd.DataFrame()})
    assert list(service.book.histories) == [SPY]


def test_evaluate_publishes_each_instrument_and_the_universe(make_service, store, fake_valkey, recording_log):
    spy = history(n=20)
    service = make_service(frames={SPY: spy, QQQ: history(n=5)})

    snap = service.evaluate(event_ts_ns=NOW_NS - 3, wall_ns=NOW_NS)

    assert snap.instrument_id == SPY
    assert snap.spot == pytest.approx(spy["close"].iloc[-1])
    assert snap.rv_percentile == 50.0  # Fewer than 21 completed rows: no RV20 history yet
    payload = store.get_json_value(keys.rv_key(SPY))
    assert payload["symbol"] == SPY
    assert (payload["ts_ns"], payload["event_ts_ns"]) == (NOW_NS, NOW_NS - 3)
    assert payload["rv_20d"] == round(snap.rv_20d, 6)
    assert not fake_valkey.exists(keys.rv_key(QQQ))  # Fewer than 10 rows
    assert list(store.get_json_value(keys.RV_UNIVERSE)) == [SPY]
    assert recording_log.info.call_args_list[0].args[0].startswith(f"RealizedVol {SPY} S=")


def test_nothing_is_published_without_history(make_service, fake_valkey):
    assert make_service().evaluate(NOW_NS, NOW_NS) is None
    assert fake_valkey.dbsize() == 0


def test_correlation_uses_dates_and_excludes_vol_products(make_service):
    spy = history()
    service = make_service(frames={SPY: spy, QQQ: history(scale=2.0), VIXY: inverse(spy)})
    assert service.avg_pair_corr() == pytest.approx(1.0)  # VIXY (corr -1) is excluded

    with_vixy_as_equity = make_service(universe=(SPY, QQQ, "UVXY.ARCA"), frames={SPY: spy, QQQ: history(scale=2.0), "UVXY.ARCA": inverse(spy)})
    assert with_vixy_as_equity.avg_pair_corr() == pytest.approx(-1.0 / 3.0)


def test_rv20_percentile_history_is_cached_until_new_bars(make_service):
    service = make_service(frames={SPY: history(n=40)})
    first = service.book.historical_rv20(SPY, 252)
    assert len(first) == 20  # End points 21..40
    assert service.book.historical_rv20(SPY, 252) is first

    service.on_bar(BarObs(SPY, 100.0, 101.0, 99.0, 100.5, 1_000.0, utc_ns(2026, 12, 4, 21), "DAY"))  # After the last row
    assert len(service.book.historical_rv20(SPY, 252)) == 21


# --- Live regular-session bar ----------------------------------------------------------------------


def test_regular_session_prices_build_a_provisional_daily_bar(make_service):
    service = make_service(frames={SPY: history()})
    service.on_quote(QuoteObs(SPY, 99.9, 100.1, NOW_NS))  # 10:00 New York
    service.on_bar(BarObs(SPY, 0, 0, 0, 101.0, 0, NOW_NS + 60_000_000_000))  # Minute bar close
    service.on_quote(QuoteObs(SPY, 98.9, 99.1, NOW_NS + 120_000_000_000))

    frame = service.book.effective_frame(SPY)
    assert len(frame) == 31
    last = frame.iloc[-1]
    assert (last["open"], last["high"], last["low"], last["close"]) == (100.0, 101.0, 99.0, 99.0)


def test_extended_hours_and_weekend_prices_are_ignored():
    book = DailyHistoryBook(history_cap=300)
    book.update_intraday(SPY, 100.0, utc_ns(2026, 11, 24, 22, 0))  # 17:00 New York
    book.update_intraday(SPY, 100.0, utc_ns(2026, 11, 28, 15, 0))  # Saturday
    book.update_intraday(SPY, -1.0, NOW_NS)
    assert book.live_sessions == {} and book.last_tick_ns == {}


def test_next_session_rolls_the_previous_one_into_history(make_service):
    service = make_service(frames={SPY: history()})
    service.on_quote(QuoteObs(SPY, 99.9, 100.1, NOW_NS))
    next_morning = utc_ns(2026, 11, 25, 15, 0)
    service.on_quote(QuoteObs(SPY, 101.9, 102.1, next_morning))

    hist = service.book.histories[SPY]
    assert len(hist) == 31
    assert (int(hist["ts_ns"].iloc[-1]), hist["close"].iloc[-1]) == (NOW_NS, 100.0)
    assert service.book.live_sessions[SPY]["session"] == "2026-11-25"


def test_completed_daily_bar_replaces_the_live_session(make_service):
    service = make_service(frames={SPY: history()})
    service.on_quote(QuoteObs(SPY, 99.9, 100.1, NOW_NS))
    close_ts = utc_ns(2026, 11, 24, 21, 0)
    service.on_bar(BarObs(SPY, 99.0, 102.0, 98.0, 101.5, 5_000.0, close_ts, "DAY"))

    assert SPY not in service.book.live_sessions
    hist = service.book.histories[SPY]
    assert (len(hist), hist["close"].iloc[-1], hist["volume"].iloc[-1]) == (31, 101.5, 5_000.0)


def test_history_is_capped():
    book = DailyHistoryBook(history_cap=3)
    for i in range(5):
        book.append_daily_row(SPY, {"ts_ns": i, "open": 1.0, "high": 1.0, "low": 1.0, "close": float(i), "volume": 0.0})
    assert book.histories[SPY]["close"].tolist() == [2.0, 3.0, 4.0]
    assert np.all(np.diff(book.histories[SPY]["ts_ns"].to_numpy()) > 0)
