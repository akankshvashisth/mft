import math

import pytest

from src.common import keys
from src.common.market import BarObs, QuoteObs, TradeObs
from src.services.futures_basis import (
    FuturesBasis,
    FuturesBasisService,
    FuturesSettings,
    SameMinuteRatio,
    compute_futures_basis,
)
from tests.factories import DAY_NS, NOW_NS, ago, surface_payload

SPY, MES, SPX, MNQ = "SPY.ARCA", "MESZ6.CME", "^SPX.CBOE", "MNQZ6.CME"
SETTINGS = FuturesSettings(spy_id=SPY, mes_id=MES, spx_id=SPX, mnq_id=MNQ)
R, Q = 0.0425, 0.0125


# --- Pure carry maths ------------------------------------------------------------------------------


def test_basis_round_trips_the_carry_of_a_consistent_futures_price():
    fwd_30d = 700.0 * math.exp((R - Q) * 30 / 365.25)
    parity_fwd = 700.0 * math.exp((R - Q) * 24 / 365.25)

    basis = compute_futures_basis(700.0, 10.0 * fwd_30d, 10.0, 30.0, R, parity_fwd, 700.0, Q, 24.0)

    assert basis.mes_scaled_fwd == pytest.approx(fwd_30d)
    assert basis.basis_usd == pytest.approx(fwd_30d - 700.0)
    assert basis.basis_pct == pytest.approx((fwd_30d - 700.0) / 700.0)
    assert basis.q_fut == pytest.approx(Q, abs=1e-12)
    assert basis.fut_implied_opt_fwd == pytest.approx(parity_fwd)
    assert basis.fwd_diff_bps == pytest.approx(0.0, abs=1e-8)
    assert basis.q_diff_bps == pytest.approx(0.0, abs=1e-8)


def test_basis_needs_a_ratio_and_a_futures_expiry():
    assert compute_futures_basis(700.0, 7020.0, None, 30.0, R) == FuturesBasis()
    assert compute_futures_basis(700.0, 7020.0, 10.0, None, R) == FuturesBasis()


def test_parity_comparison_needs_the_surface():
    basis = compute_futures_basis(700.0, 7020.0, 10.0, 30.0, R)
    assert basis.q_fut is not None
    assert (basis.fut_implied_opt_fwd, basis.fwd_diff_bps, basis.q_diff_bps) == (None, None, None)

    no_q = compute_futures_basis(700.0, 7020.0, 10.0, 30.0, R, 701.0, 700.0, None, 24.0)
    assert no_q.fwd_diff_bps is not None and no_q.q_diff_bps is None


# --- Same-minute ratio -----------------------------------------------------------------------------


def test_ratio_needs_both_closes_of_the_same_minute():
    ratio = SameMinuteRatio(SPX, SPY)
    minute = NOW_NS - 60 * 1_000_000_000

    assert not ratio.on_close(SPX, 7000.0, minute, wall_ns=1)
    assert not ratio.on_close(SPY, 701.0, minute + 60 * 1_000_000_000, wall_ns=2)  # Different minute
    assert ratio.on_close(SPY, 700.0, minute, wall_ns=3)
    assert (ratio.ratio, ratio.bar_ts_ns, ratio.wall_ns) == (10.0, minute, 3)


def test_older_minutes_never_replace_a_newer_ratio():
    ratio = SameMinuteRatio(SPX, SPY)
    ratio.on_close(SPX, 7000.0, 200, 1)
    ratio.on_close(SPY, 700.0, 200, 1)
    ratio.on_close(SPX, 7100.0, 100, 2)
    assert not ratio.on_close(SPY, 700.0, 100, 2)
    assert ratio.ratio == 10.0


def test_ratio_ignores_other_instruments_bad_closes_and_bounds_memory():
    ratio = SameMinuteRatio(SPX, SPY, max_bars_kept=3)
    assert not ratio.on_close("QQQ.NASDAQ", 500.0, 1, 1)
    assert not ratio.on_close(SPX, 0.0, 1, 1)
    for ts in range(10):
        ratio.on_close(SPX, 7000.0 + ts, ts, 1)
    assert list(ratio._closes[SPX]) == [7, 8, 9]


# --- Service ---------------------------------------------------------------------------------------


@pytest.fixture
def service(store, recording_log):
    store.set_json(keys.surface_key(SPY), surface_payload(SPY))
    return FuturesBasisService(SETTINGS, store=store, log=recording_log)


def feed_market(service, quote_ts=ago(1.0), ratio_wall_ns=ago(30.0)):
    service.on_quote(QuoteObs(SPY, 699.99, 700.01, quote_ts))
    service.on_quote(QuoteObs(MES, 7019.75, 7020.25, quote_ts))
    minute = ago(60.0)
    service.on_bar(BarObs(SPX, 7000.0, 7000.0, 7000.0, 7000.0, 0.0, minute), wall_ns=ratio_wall_ns)
    service.on_bar(BarObs(SPY, 700.0, 700.0, 700.0, 700.0, 0.0, minute), wall_ns=ratio_wall_ns)


def test_waits_for_fresh_spy_and_mes_quotes(service, fake_valkey, recording_log):
    assert service.evaluate(NOW_NS, NOW_NS, NOW_NS + 30 * DAY_NS) is None
    recording_log.info.assert_called_once_with("FuturesContext waiting for fresh quotes (SPY=0.00, MES=0.00)")
    assert not fake_valkey.exists(keys.FUTURES_LATEST)

    service.on_quote(QuoteObs(SPY, 699.99, 700.01, ago(91.0)))  # Too old to count
    service.on_quote(QuoteObs(MES, 7019.75, 7020.25, ago(1.0)))
    assert service.evaluate(NOW_NS, NOW_NS, NOW_NS + 30 * DAY_NS) is None


def test_full_cross_check(service, store):
    feed_market(service)
    payload = service.evaluate(wall_ns=NOW_NS, event_ts_ns=NOW_NS - 7, mes_expiry_ns=NOW_NS + 30 * DAY_NS)

    assert store.get_json_value(keys.FUTURES_LATEST) == payload
    assert payload["cross_check_valid"] is True
    assert payload["ratio_source"] == "LIVE_SPX_SPY_1MIN_BARS"
    assert (payload["spx_spy_ratio"], payload["ratio_age_secs"], payload["spx_quote"]) == (10.0, 30.0, 7000.0)
    assert (payload["mes_quote"], payload["mes_scaled_fwd"], payload["cash_futures_basis_usd"]) == (7020.0, 702.0, 2.0)
    assert (payload["mes_ttm_days"], payload["mes_spy_equiv_shares"], payload["parity_fwd"]) == (30.0, 50.0, 701.0)
    assert (payload["mnq_quote"], payload["mnq_symbol"], payload["spx_symbol"]) == (None, MNQ, SPX)
    implied = 700.0 * math.exp((R - payload["futures_implied_q"]) * 30 / 365.25)
    assert implied == pytest.approx(702.0, rel=1e-6)
    assert payload["parity_vs_fut_diff_usd"] == pytest.approx(701.0 - payload["fut_implied_opt_fwd"], abs=2e-3)
    assert (payload["ts_ns"], payload["event_ts_ns"]) == (NOW_NS, NOW_NS - 7)


def test_stale_ratio_disables_the_cross_check(service):
    feed_market(service, ratio_wall_ns=ago(901.0))
    payload = service.evaluate(NOW_NS, NOW_NS, NOW_NS + 30 * DAY_NS)

    assert (payload["ratio_source"], payload["spx_spy_ratio"], payload["ratio_age_secs"]) == ("UNAVAILABLE", None, 901.0)
    assert payload["cross_check_valid"] is False
    assert (payload["futures_implied_q"], payload["mes_spy_equiv_shares"]) == (None, None)


def test_stale_surface_keeps_futures_carry_but_no_parity_comparison(service, store):
    store.set_json(keys.surface_key(SPY), surface_payload(SPY, now_ns=ago(121.0)))
    feed_market(service)
    payload = service.evaluate(NOW_NS, NOW_NS, NOW_NS + 30 * DAY_NS)

    assert payload["futures_implied_q"] is not None
    assert (payload["parity_fwd"], payload["parity_vs_fut_diff_bps"], payload["cross_check_valid"]) == (None, None, False)


def test_unknown_futures_expiry_disables_carry(service):
    feed_market(service)
    payload = service.evaluate(NOW_NS, NOW_NS, mes_expiry_ns=0)
    assert (payload["mes_ttm_days"], payload["futures_implied_q"], payload["cross_check_valid"]) == (None, None, False)


def test_without_an_index_feed_bars_are_ignored(store):
    service = FuturesBasisService(FuturesSettings(spy_id=SPY, mes_id=MES), store=store)
    service.on_bar(BarObs(SPY, 700.0, 700.0, 700.0, 700.0, 0.0, NOW_NS), wall_ns=NOW_NS)
    assert service.index_ratio is None
    assert service.live_ratio(NOW_NS) == (None, None, "UNAVAILABLE")
    assert service.quote_ids() == [SPY, MES]


def test_trades_update_mids_and_bad_prints_are_ignored(service):
    service.on_trade(TradeObs(MES, 7021.0, ago(1.0)))
    service.on_trade(TradeObs(MES, 0.0, NOW_NS))
    assert service.fresh_mid(MES, NOW_NS) == 7021.0
    assert service.fresh_mid(None, NOW_NS) is None
    assert service.quote_ids() == [SPY, MES, MNQ]
