import json
import numpy as np
import redis
import fast_vollib
from fast_vollib.surface import IVSurface
from nautilus_trader.model import InstrumentId

from src.models.vol_surface import VolSurfaceSnapshot
from src.actors.vol_actor import ComboLegSpec, evaluate_combo_risk, filter_clean_quotes


def main() -> None:
    # 1. Synthetic option chain (3 expiries x 41 strikes)
    S0 = 100.0
    r = 0.045
    k = np.linspace(-0.35, 0.35, 41)
    T = np.array([0.08, 0.25, 0.50])
    K = S0 * np.exp(k)

    true_iv = np.column_stack([
        0.24 + 0.18 * (k**2) - 0.06 * k,
        0.25 + 0.16 * (k**2) - 0.05 * k,
        0.26 + 0.14 * (k**2) - 0.04 * k,
    ])

    K_flat = np.repeat(K, len(T))
    T_flat = np.tile(T, len(K))
    S_flat = np.full_like(K_flat, S0)
    r_flat = np.full_like(K_flat, r)
    sig_flat = true_iv.ravel()
    flags_flat = np.where(K_flat < S0, "p", "c")

    mids = fast_vollib.fast_black_scholes(
        flag=flags_flat, S=S_flat, K=K_flat, t=T_flat, r=r_flat, sigma=sig_flat,
        backend="torch", return_as="numpy",
    )
    bids = np.maximum(mids - 0.01, 0.0)
    asks = mids + 0.01

    # 2. Hygiene filter + GPU fp64 IV inversion (match ground truth to surviving quotes)
    spreads = asks - bids
    disc = np.exp(-r_flat * T_flat)
    intrinsic = np.where(flags_flat == "c", np.maximum(S_flat - K_flat * disc, 0.0), np.maximum(K_flat * disc - S_flat, 0.0))
    keep = (
        np.isfinite(mids)
        & (bids > 0.0)
        & (mids > np.maximum(intrinsic + 1e-4, 1e-3))
        & (mids < S_flat - 1e-4)
        & ((spreads / np.maximum(mids, 1e-6)) <= 0.25)
        & (np.abs(np.log(K_flat / S_flat)) <= 0.45)
    )

    Sf, Kf, Tf, rf, mf, ff = filter_clean_quotes(S_flat, K_flat, T_flat, r_flat, bids, asks, flags_flat)
    iv_solved = fast_vollib.fast_implied_volatility(
        price=mf, S=Sf, K=Kf, t=Tf, r=rf, flag=ff,
        backend="torch", return_as="numpy",
    )
    max_err = float(np.max(np.abs(iv_solved - sig_flat[keep])))

    # 3. Surface arbitrage check
    report = IVSurface.from_logmoneyness(k, T, true_iv).validate()
    assert report.passed, "Expected arbitrage-free surface!"

    # 4. Instantiate Nautilus @customdataclass
    snap = VolSurfaceSnapshot(
        underlying_id="XBI.ARCA",
        front_expiry_ns=1_780_000_000_000_000_000,
        forward=S0,
        atm_iv=float(np.interp(0.0, k, true_iv[:, 0])),
        skew_25d=float(np.interp(-0.10, k, true_iv[:, 0]) - np.interp(0.10, k, true_iv[:, 0])),
        term_slope=float(np.interp(0.0, k, true_iv[:, -1]) - np.interp(0.0, k, true_iv[:, 0])),
        n_clean_quotes=len(mf),
        arb_free=bool(report.passed),
        compute_ms=1.8,
        ts_event=1_770_000_000_000_000_000,
        ts_init=1_770_000_000_000_000_000,
    )

    # 5. Evaluate 25k-path Iron Condor risk & write both to Valkey
    legs = [
        ComboLegSpec(90.0, "p", 1.0, 0.48),
        ComboLegSpec(95.0, "p", -1.0, 1.05),
        ComboLegSpec(105.0, "c", -1.0, 0.92),
        ComboLegSpec(110.0, "c", 1.0, 0.36),
    ]
    risk = evaluate_combo_risk(
        S0=S0, T0=30 / 365.0, horizon_days=7.0, r=r, realized_vol=0.22,
        legs=legs, leg_ivs=np.array([0.28, 0.25, 0.23, 0.25]),
    )

    vk = redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)
    vk.set("surface:XBI.ARCA", json.dumps({"atm_iv": snap.atm_iv, "arb_free": snap.arb_free, "risk": risk}))
    cached = json.loads(vk.get("surface:XBI.ARCA"))

    print(f"CustomData OK  : {snap}")
    print(f"GPU IV Max Err : {max_err:.2e} ({len(mf)} clean quotes)")
    print(f"Valkey Cached  : {cached}")


if __name__ == "__main__":
    main()
