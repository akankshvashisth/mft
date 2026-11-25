"""Offline tests for ``src.data.opra_chain_builder`` -- no network calls, no databento."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.opra_chain_builder import (
    CANONICAL_COLUMNS,
    build_opra_backtest_chain,
    compute_daily_rv_table,
    extract_daily_nbbo_snapshot,
    parse_opra_definitions,
)


class TestParseOpraDefinitions:
    def test_filters_non_standard_multipliers(self):
        def_df = pd.DataFrame([
            {"instrument_id": "A1", "raw_symbol": "DECK", "strike_price": 100.0,
             "expiration": "2026-06-19", "instrument_class": "P", "contract_multiplier": 100},
            {"instrument_id": "A2", "raw_symbol": "DECK", "strike_price": 100.0,
             "expiration": "2026-06-19", "instrument_class": "P", "contract_multiplier": 10},
            {"instrument_id": "A3", "raw_symbol": "DECK", "strike_price": 100.0,
             "expiration": "2026-06-19", "instrument_class": "C", "contract_multiplier": 100},
            {"instrument_id": "A4", "raw_symbol": "DECK", "strike_price": 100.0,
             "expiration": "2026-06-19", "instrument_class": "X", "contract_multiplier": 100},
            {"instrument_id": "A5", "raw_symbol": "DECK", "strike_price": 0.0,
             "expiration": "2026-06-19", "instrument_class": "P", "contract_multiplier": 100},
            {"instrument_id": "A6", "raw_symbol": "DECK", "strike_price": 100.0,
             "expiration": "invalid-date", "instrument_class": "P", "contract_multiplier": 100},
        ])
        parsed = parse_opra_definitions(def_df)
        assert len(parsed) == 2
        assert set(parsed["right"]) == {"P", "C"}
        assert (parsed["multiplier"] == 100).all()
        assert (parsed["strike"] == 100.0).all()

    def test_empty_input_returns_empty(self):
        assert parse_opra_definitions(pd.DataFrame()).empty


class TestExtractDailyNbboSnapshot:
    def test_picks_last_valid_before_1545_et(self):
        def _ny_epoch(h: int, m: int, s: int = 0) -> int:
            utc_h = h + 4
            if utc_h >= 24:
                utc_h -= 24
            return pd.Timestamp(f"2026-06-01 {utc_h:02d}:{m:02d}:{s:02d}", tz="UTC").value

        rows = [
            {"ts_event": _ny_epoch(15, 20), "instrument_id": "I1",
             "bid_px_00": 1.00, "ask_px_00": 1.10, "bid_sz_00": 10, "ask_sz_00": 15},
            {"ts_event": _ny_epoch(15, 44), "instrument_id": "I1",
             "bid_px_00": 1.02, "ask_px_00": 1.12, "bid_sz_00": 12, "ask_sz_00": 17},
            {"ts_event": _ny_epoch(15, 45, 0), "instrument_id": "I1",
             "bid_px_00": 1.03, "ask_px_00": 1.13, "bid_sz_00": 14, "ask_sz_00": 19},
            {"ts_event": _ny_epoch(15, 45, 30), "instrument_id": "I1",
             "bid_px_00": 1.20, "ask_px_00": 1.05, "bid_sz_00": 5, "ask_sz_00": 5},
            {"ts_event": _ny_epoch(15, 59), "instrument_id": "I1",
             "bid_px_00": 1.04, "ask_px_00": 1.14, "bid_sz_00": 16, "ask_sz_00": 20},
        ]
        cbbo = pd.DataFrame(rows)
        result = extract_daily_nbbo_snapshot(cbbo)
        assert len(result) == 1
        assert result.iloc[0]["bid"] == pytest.approx(1.03, abs=1e-4)
        assert result.iloc[0]["ask"] == pytest.approx(1.13, abs=1e-4)
        assert result.iloc[0]["mid"] == pytest.approx(1.08, abs=1e-4)
        assert result.iloc[0]["bid_size"] == 14
        assert result.iloc[0]["ask_size"] == 19
        assert result.iloc[0]["date"] == "2026-06-01"

    def test_empty_input_returns_empty(self):
        assert extract_daily_nbbo_snapshot(pd.DataFrame()).empty


class TestComputeDailyRvTable:
    def test_has_no_lookahead(self):
        n = 80
        rng = np.random.default_rng(42)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        opens = np.concatenate([[100.0], closes[:-1]]) * np.exp(rng.normal(0.0, 0.005, n))
        highs = np.maximum(opens, closes) * (1.0 + rng.uniform(0.0, 0.01, n))
        lows = np.minimum(opens, closes) * (1.0 - rng.uniform(0.0, 0.01, n))
        dates = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
        df = pd.DataFrame({
            "date": dates, "open": opens, "high": highs, "low": lows, "close": closes,
        })
        rv_table = compute_daily_rv_table(df, window=20)
        assert len(rv_table) == n - 20

        rv_day_39 = rv_table.iloc[19]["rv_20d"]
        df_shock = df.copy()
        df_shock.loc[40, "close"] = df_shock.loc[40, "close"] * 10.0
        df_shock.loc[40, "high"] = max(df_shock.loc[40, "high"], df_shock.loc[40, "close"])
        rv_shock = compute_daily_rv_table(df_shock, window=20)
        rv_shock_day_39 = rv_shock.iloc[19]["rv_20d"]
        assert abs(rv_shock_day_39 - rv_day_39) < 1e-6

    def test_empty_input_returns_empty(self):
        assert compute_daily_rv_table(pd.DataFrame()).empty


class TestBuildOpraBacktestChain:
    def test_round_trips_bsm_iv(self):
        from src.quant.black_scholes import bsm_prices

        known_iv = 0.35
        risk_free = 0.0425

        rng = np.random.default_rng(7)
        bar_rows = []
        prev = 100.0
        for i in range(75):
            d = (date(2026, 3, 28) + timedelta(days=i)).isoformat()
            if i == 0:
                bar_rows.append({"date": d, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
            else:
                ret = rng.normal(0.0, known_iv / 16.0)
                nc = prev * np.exp(ret)
                bar_rows.append({"date": d, "open": prev,
                                 "high": max(prev, nc) * 1.005, "low": min(prev, nc) * 0.995,
                                 "close": nc})
                prev = nc
        bars_df = pd.DataFrame(bar_rows)
        spot = round(float(bars_df[bars_df["date"] == "2026-06-01"]["close"].iloc[0]), 2)

        dte_vals = [25, 30, 35]
        strikes = [spot * 0.9, spot * 0.95, spot * 1.0, spot * 1.05, spot * 1.1]
        rights = ["P", "C"]

        def_rows, cbbo_rows = [], []
        date_str = "2026-06-01"
        expiry_base = date(2026, 6, 1)

        for exp_dte in dte_vals:
            expiry_str = (expiry_base + timedelta(days=exp_dte)).isoformat()
            for s in strikes:
                for r in rights:
                    iid = f"DECK_{expiry_str}_{r}{int(s)}"
                    def_rows.append({"instrument_id": iid, "raw_symbol": "DECK",
                                     "strike_price": s, "expiration": expiry_str,
                                     "instrument_class": r, "contract_multiplier": 100})
                    tte = exp_dte / 365.25
                    p = float(np.asarray(bsm_prices(spot, s, tte, risk_free, known_iv, r, 0.0)).flat[0])
                    bid = max(p * 0.95, 0.05)
                    ask = p * 1.05
                    if ask < bid:
                        ask = bid + 0.01
                    cbbo_rows.append({"ts_event": pd.Timestamp(f"{date_str} 19:45:00", tz="UTC").value,
                                      "instrument_id": iid,
                                      "bid_px_00": round(bid, 4), "ask_px_00": round(ask, 4),
                                      "bid_sz_00": 10, "ask_sz_00": 10})

        def_df = pd.DataFrame(def_rows)
        cbbo_df = pd.DataFrame(cbbo_rows)

        chain = build_opra_backtest_chain(def_df, cbbo_df, bars_df,
                                          risk_free_rate=risk_free, dividend_yield=0.0,
                                          min_dte=1, max_dte=90)

        assert not chain.empty
        for col in CANONICAL_COLUMNS:
            assert col in chain.columns, f"Missing column: {col}"

        median_iv = chain["iv"].median()
        assert abs(median_iv - known_iv) < 1e-2, f"Median IV {median_iv:.4f} deviates from known {known_iv}"

        assert chain["atm_iv"].notna().all()
        median_atm = chain["atm_iv"].median()
        assert abs(median_atm - known_iv) < 1e-1

        for i in range(len(chain) - 1):
            a = tuple(chain.iloc[i][c] for c in ("date", "expiry", "right", "strike"))
            b = tuple(chain.iloc[i + 1][c] for c in ("date", "expiry", "right", "strike"))
            assert a <= b, f"Row {i} unsorted: {a} > {b}"
