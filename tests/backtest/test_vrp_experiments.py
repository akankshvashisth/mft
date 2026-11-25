"""Offline test for ``run_ablation_suite`` — verifies all 4 experiment tables."""
from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.run_vrp_experiments import run_ablation_suite
from tests.factories import risk_assessment
from tests.fakes import FakePricer, recording_sink, stub_calendar


def _build_synthetic_chain(days: int = 25) -> pd.DataFrame:
    """Multi-expiry daily chain for experiment testing (25 days, 2 expiries)."""
    start = date(2026, 1, 1)
    expiry_a = (start + timedelta(days=21)).isoformat()
    expiry_b = (start + timedelta(days=30)).isoformat()

    rows = []
    for d in range(days):
        dt = (start + timedelta(days=d)).isoformat()
        spot = 100.0
        for expiry, mul in [(expiry_a, 1.0), (expiry_b, 1.2)]:
            for strike, (bid, ask) in {
                80.0: (0.40 * mul, 0.44 * mul),  # spread 9.5% < 18%
                85.0: (0.80 * mul, 0.88 * mul),
                90.0: (1.60 * mul, 1.76 * mul),
                95.0: (2.40 * mul, 2.64 * mul),
            }.items():
                rows.append({
                    "date": dt, "expiry": expiry,
                    "dte": (date.fromisoformat(expiry) - date.fromisoformat(dt)).days,
                    "spot": spot, "rv_20d": 0.15, "atm_iv": 0.22,
                    "right": "P", "strike": strike, "bid": bid, "ask": ask,
                    "mid": round(0.5 * (bid + ask), 4), "iv": 0.22,
                    "bid_size": 10, "ask_size": 10,
                })
    return pd.DataFrame(rows)


class TestRunAblationSuite:
    def test_run_ablation_suite_returns_all_four_tables(self):
        """Verifies all 4 experiment keys have expected row counts and valid metrics."""
        df = _build_synthetic_chain(days=25)
        pricer = FakePricer(lambda combo: risk_assessment(
            combo_name=combo.name, recommended_contracts=2, approved=True,
        ))
        cal = stub_calendar()
        results = run_ablation_suite(df, symbol="TEST", pricer=pricer, calendar=cal)

        # Check all 4 keys exist
        expected_keys = {"friction_frontier", "exit_rules", "vertical_selection", "vrp_threshold"}
        assert expected_keys.issubset(results.keys()), f"Missing keys: {expected_keys - results.keys()}"

        # Friction Frontier: 3 fill_probs × 4 slip_fracs = 12
        ff = results["friction_frontier"]
        assert len(ff) == 12, f"Expected 12 friction frontier rows, got {len(ff)}"
        for r in ff:
            assert "passive_touch_fill_prob" in r
            assert "exit_slippage_frac" in r
            assert "completed_trades" in r

        # Exit Rules: 4 TP fractions × 3 min_exit_dte = 12
        er = results["exit_rules"]
        assert len(er) == 12, f"Expected 12 exit rules rows, got {len(er)}"
        for r in er:
            assert "profit_target_fraction" in r
            assert "min_exit_dte" in r
            assert "completed_trades" in r

        # Vertical Selection: 2 rows (adjacent + delta_targeted)
        vs = results["vertical_selection"]
        assert len(vs) == 2, f"Expected 2 vertical selection rows, got {len(vs)}"
        modes = [r.get("mode") for r in vs]
        assert "adjacent" in modes
        assert "delta_targeted" in modes
        for r in vs:
            if "error" not in r:
                assert "completed_trades" in r

        # VRP Threshold: 4 values
        vt = results["vrp_threshold"]
        assert len(vt) == 4, f"Expected 4 VRP threshold rows, got {len(vt)}"
        for r in vt:
            assert "min_vrp" in r
            assert "completed_trades" in r

        # At least some trades should complete (positive VRP at entry)
        any_trades = any(
            r.get("completed_trades", 0) > 0
            for exp_list in results.values()
            for r in exp_list if isinstance(r, dict)
        )
        assert any_trades, "Expected at least one experiment to produce trades"