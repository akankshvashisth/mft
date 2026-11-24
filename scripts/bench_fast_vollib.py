import os
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import time
import numpy as np
import torch
import fast_vollib
from fast_vollib.surface import IVSurface


def sync_if_cuda(backend: str) -> None:
    if backend == "torch" and torch.cuda.is_available():
        torch.cuda.synchronize()


def filter_clean_quotes(
    S: np.ndarray,
    K: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    prices: np.ndarray,
    flag: np.ndarray,
    true_sigma: np.ndarray,
    min_price: float = 1e-3,
    max_abs_k: float = 0.45,
) -> tuple[np.ndarray, ...]:
    """Production pre-solver hygiene gate: strips zero-bid/subnormal, intrinsic-violating, and extreme-wing quotes."""
    disc = np.exp(-r * t)
    intrinsic = np.where(flag == "c", np.maximum(S - K * disc, 0.0), np.maximum(K * disc - S, 0.0))
    log_m = np.abs(np.log(K / S))

    valid = (
        np.isfinite(prices)
        & (prices > np.maximum(intrinsic + 1e-4, min_price))
        & (prices < S - 1e-4)
        & (log_m <= max_abs_k)
    )
    return S[valid], K[valid], t[valid], r[valid], prices[valid], flag[valid], true_sigma[valid]


def bench_iv_inversion(n_raw: int, backend: str, dtype=np.float64, n_runs: int = 20) -> tuple[int, float, float]:
    rng = np.random.default_rng(42)
    S_raw = np.full(n_raw, 100.0, dtype=dtype)
    K_raw = rng.uniform(70.0, 130.0, size=n_raw).astype(dtype)
    t_raw = rng.uniform(0.05, 1.0, size=n_raw).astype(dtype)
    r_raw = np.full(n_raw, 0.045, dtype=dtype)
    sig_raw = rng.uniform(0.15, 0.65, size=n_raw).astype(dtype)
    flag_raw = np.where(K_raw < S_raw, "p", "c")  # OTM-only splicing

    prices_raw = fast_vollib.fast_black_scholes(
        flag=flag_raw, S=S_raw, K=K_raw, t=t_raw, r=r_raw, sigma=sig_raw,
        backend=backend, return_as="numpy",
    )

    # Apply production pre-solver hygiene filter
    S, K, t, r, prices, flag, true_sigma = filter_clean_quotes(
        S_raw, K_raw, t_raw, r_raw, prices_raw, flag_raw, sig_raw
    )
    n_clean = len(prices)

    # Warm-up CUDA kernels & caches
    for _ in range(3):
        _ = fast_vollib.fast_implied_volatility(
            price=prices, S=S, K=K, t=t, r=r, flag=flag,
            backend=backend, return_as="numpy",
        )
    sync_if_cuda(backend)

    times_ms = []
    for _ in range(n_runs):
        sync_if_cuda(backend)
        t0 = time.perf_counter()
        iv = fast_vollib.fast_implied_volatility(
            price=prices, S=S, K=K, t=t, r=r, flag=flag,
            backend=backend, return_as="numpy",
        )
        sync_if_cuda(backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    max_err = float(np.max(np.abs(iv - true_sigma)))
    return n_clean, float(np.median(times_ms)), max_err


def test_surface_arbitrage() -> tuple[bool, bool]:
    k = np.linspace(-0.35, 0.35, 41)
    T = np.array([0.08, 0.25, 0.50])

    clean_iv = np.column_stack([
        0.24 + 0.18 * (k ** 2) - 0.06 * k,
        0.25 + 0.16 * (k ** 2) - 0.05 * k,
        0.26 + 0.14 * (k ** 2) - 0.04 * k,
    ])
    clean_report = IVSurface.from_logmoneyness(k, T, clean_iv).validate()

    bad_iv = clean_iv.copy()
    bad_iv[20, 1] = 0.10
    bad_report = IVSurface.from_logmoneyness(k, T, bad_iv).validate()

    return bool(clean_report.passed), bool(bad_report.passed)


def bench_combo_risk(n_scenarios: int = 25_000, backend: str = "torch") -> dict:
    S0, T0, T1, r = 100.0, 30 / 365.0, 23 / 365.0, 0.045
    strikes = np.array([90.0, 95.0, 105.0, 110.0])
    flags = np.array(["p", "p", "c", "c"])
    qtys = np.array([1.0, -1.0, -1.0, 1.0])
    iv_entry = np.array([0.34, 0.30, 0.27, 0.29])

    entry_prices = fast_vollib.fast_black_scholes(
        flag=flags, S=S0, K=strikes, t=T0, r=r, sigma=iv_entry,
        backend="numpy", return_as="numpy",
    )
    entry_cost = float(np.sum(qtys * entry_prices)) * 100.0

    rng = np.random.default_rng(7)
    dt = 7 / 365.0
    realized_vol = 0.24
    z = rng.standard_normal(n_scenarios)
    S_H = S0 * np.exp((-0.5 * realized_vol**2) * dt + realized_vol * np.sqrt(dt) * z)

    S_grid = np.broadcast_to(S_H[None, :], (4, n_scenarios)).ravel()
    K_grid = np.repeat(strikes, n_scenarios)
    F_grid = np.repeat(flags, n_scenarios)
    sig_grid = np.repeat(iv_entry, n_scenarios)

    _ = fast_vollib.fast_black_scholes(
        flag=F_grid, S=S_grid, K=K_grid, t=T1, r=r, sigma=sig_grid,
        backend=backend, return_as="numpy",
    )
    sync_if_cuda(backend)

    t0 = time.perf_counter()
    vals_H = fast_vollib.fast_black_scholes(
        flag=F_grid, S=S_grid, K=K_grid, t=T1, r=r, sigma=sig_grid,
        backend=backend, return_as="numpy",
    ).reshape(4, n_scenarios)
    pnl = (np.sum(qtys[:, None] * vals_H, axis=0) * 100.0) - entry_cost
    var_95 = float(np.percentile(pnl, 5))
    es_95 = float(np.mean(pnl[pnl <= var_95]))
    sync_if_cuda(backend)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "ms": elapsed_ms,
        "ev": float(np.mean(pnl)),
        "pop": float(np.mean(pnl > 0) * 100.0),
        "var_95": var_95,
        "es_95": es_95,
    }


if __name__ == "__main__":
    print(f"fast-vollib version : {fast_vollib.__version__}")
    print(f"Auto backend        : {fast_vollib.get_backend()}")
    print(f"CUDA available      : {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    print("-" * 78)

    clean_ok, bad_ok = test_surface_arbitrage()
    print(f"Surface Arb Gate    : clean_passed={clean_ok} (expect True), bad_passed={bad_ok} (expect False)")
    assert clean_ok and not bad_ok

    print("-" * 78)
    print(f"{'Raw -> Clean Quotes':<26} | {'Backend':<14} | {'Median (ms)':>11} | {'Max IV Err':>11}")
    print("-" * 78)
    for n in (10_000, 100_000):
        for b_name, dt in [("numpy", np.float64), ("torch (fp64)", np.float64), ("torch (fp32)", np.float32)]:
            b = "numpy" if b_name == "numpy" else "torch"
            n_clean, ms, err = bench_iv_inversion(n, backend=b, dtype=dt)
            label = f"{n:,} -> {n_clean:,}"
            print(f"{label:<26} | {b_name:<14} | {ms:>9.2f} ms | {err:>10.2e}")

    print("-" * 78)
    risk = bench_combo_risk(25_000, backend="torch")
    print(
        f"25k-Path Iron Condor Risk  : {risk['ms']:.2f} ms | "
        f"EV=${risk['ev']:.2f} | PoP={risk['pop']:.1f}% | "
        f"VaR95=${risk['var_95']:.2f} | ES95=${risk['es_95']:.2f}"
    )
