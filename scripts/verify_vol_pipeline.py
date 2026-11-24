"""End-to-end check of the vol pipeline on a synthetic chain (needs fast_vollib, nautilus_trader and Valkey).

Hygiene mask -> fast_vollib IV inversion -> IVSurface arbitrage check -> SviSliceParams custom data ->
ScenarioPricer iron condor -> mft:verify:vol_pipeline in Valkey.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import fast_vollib  # noqa: E402
from fast_vollib.surface import IVSurface  # noqa: E402
from nautilus_trader.model import InstrumentId  # noqa: E402

from src.common import keys  # noqa: E402
from src.common.constants import DEFAULT_VALKEY_URL  # noqa: E402
from src.common.store import make_store  # noqa: E402
from src.models.vol_surface import SviSliceParams  # noqa: E402
from src.quant.quote_hygiene import clean_quote_mask  # noqa: E402
from src.quant.surface_metrics import delta25_log_moneyness  # noqa: E402
from src.risk.scenario_pricer import ScenarioPricer  # noqa: E402
from src.scanner.combos import CandidateCombo, CandidateLeg  # noqa: E402

S0 = 100.0
R = 0.045
LOG_MONEYNESS = np.linspace(-0.35, 0.35, 41)
EXPIRIES = np.array([0.08, 0.25, 0.50])


def synthetic_smiles() -> np.ndarray:
    """IV grid (strikes x expiries): skewed, convex smiles flattening with maturity."""
    k = LOG_MONEYNESS
    return np.column_stack([
        0.24 + 0.18 * (k**2) - 0.06 * k,
        0.25 + 0.16 * (k**2) - 0.05 * k,
        0.26 + 0.14 * (k**2) - 0.04 * k,
    ])


def main() -> None:
    true_iv = synthetic_smiles()

    # 1. Synthetic option chain (41 strikes x 3 expiries), OTM-spliced, 2-cent wide markets
    K = S0 * np.exp(LOG_MONEYNESS)
    K_flat = np.repeat(K, len(EXPIRIES))
    T_flat = np.tile(EXPIRIES, len(K))
    S_flat = np.full_like(K_flat, S0)
    r_flat = np.full_like(K_flat, R)
    sig_flat = true_iv.ravel()
    flags_flat = np.where(K_flat < S0, "p", "c")

    mids = fast_vollib.fast_black_scholes(
        flag=flags_flat, S=S_flat, K=K_flat, t=T_flat, r=r_flat, sigma=sig_flat,
        backend="torch", return_as="numpy",
    )
    bids = np.maximum(mids - 0.01, 0.0)
    asks = mids + 0.01

    # 2. Hygiene filter + GPU fp64 IV inversion (ground truth matched to the surviving quotes)
    keep = clean_quote_mask(S_flat, K_flat, T_flat, r_flat, mids, flags_flat, bids=bids, asks=asks, max_rel_spread=0.25)
    iv_solved = fast_vollib.fast_implied_volatility(
        price=mids[keep], S=S_flat[keep], K=K_flat[keep], t=T_flat[keep], r=r_flat[keep], flag=flags_flat[keep],
        backend="torch", return_as="numpy",
    )
    max_err = float(np.max(np.abs(iv_solved - sig_flat[keep])))

    # 3. Surface arbitrage check
    report = IVSurface.from_logmoneyness(LOG_MONEYNESS, EXPIRIES, true_iv).validate()
    assert report.passed, "Expected arbitrage-free surface!"

    # 4. Real SviSliceParams custom data for the front slice (25-delta points as in the surface metrics)
    front = true_iv[:, 0]
    atm_iv = float(np.interp(0.0, LOG_MONEYNESS, front))
    k25 = delta25_log_moneyness(atm_iv, float(EXPIRIES[0]))
    snapshot = SviSliceParams(
        underlying_id=InstrumentId.from_str("XBI.ARCA"),
        expiry_ns=1_780_000_000_000_000_000,
        forward=S0,
        atm_iv=atm_iv,
        skew_25d=float(np.interp(-k25, LOG_MONEYNESS, front) - np.interp(k25, LOG_MONEYNESS, front)),
        arb_free=bool(report.passed),
        quote_count=int(np.count_nonzero(keep)),
        ts_event=1_770_000_000_000_000_000,
        ts_init=1_770_000_000_000_000_000,
    )

    # 5. 25k-path iron condor risk with the production pricer
    strikes = [(90.0, "P", +1, 0.48, 0.28), (95.0, "P", -1, 1.05, 0.25), (105.0, "C", -1, 0.92, 0.23), (110.0, "C", +1, 0.36, 0.25)]
    legs = [
        CandidateLeg(f"XBI {k:g}{right}", k, right, side, 1, mid - 0.02, mid + 0.02, mid, iv, iv, 0.0, 0.0)
        for k, right, side, mid, iv in strikes
    ]
    net_credit = -sum(leg.side * leg.mid for leg in legs)
    combo = CandidateCombo(
        name="XBI 90/95P-105/110C IC",
        strategy_type="IRON_CONDOR",
        underlying="XBI.ARCA",
        spot=S0,
        forward=S0,
        ttm_days=30.0,
        risk_free_rate=R,
        dividend_yield=0.0,
        legs=legs,
        net_credit=round(net_credit, 4),
        max_width=5.0,
    )
    risk = ScenarioPricer(n_scenarios=25_000).evaluate_combo(combo, realized_vol_p=0.22, horizon_days=7.0)

    store = make_store(os.environ.get("VALKEY_URL", DEFAULT_VALKEY_URL))
    store.set_json(
        keys.VERIFY_VOL_PIPELINE,
        {"atm_iv": snapshot.atm_iv, "arb_free": snapshot.arb_free, "max_iv_err": max_err, "risk": risk.to_dict()},
    )
    cached = store.get_json(keys.VERIFY_VOL_PIPELINE)

    print(f"CustomData OK  : {snapshot}")
    print(f"GPU IV Max Err : {max_err:.2e} ({int(np.count_nonzero(keep))} clean quotes)")
    print(f"Iron Condor    : EV=${risk.net_ev_usd:+.2f} ES99=${risk.es_99_usd:.2f} approved={risk.approved}")
    print(f"Valkey Cached  : {sorted(cached)} at {keys.VERIFY_VOL_PIPELINE}")


if __name__ == "__main__":
    main()
