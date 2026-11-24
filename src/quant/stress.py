"""Spot x vol stress grids."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def shock_mesh(spot_shocks: Sequence[float], vol_shocks: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (spot shock, vol shock) pairs in row-major grid order (rows = spot shocks)."""
    spot_grid, vol_grid = np.meshgrid(
        np.asarray(spot_shocks, dtype=float), np.asarray(vol_shocks, dtype=float), indexing="ij"
    )
    return spot_grid.ravel(), vol_grid.ravel()


def to_grid(values, n_rows: int, n_cols: int, decimals: int = 2) -> list[list[float]]:
    arr = np.asarray(values, dtype=float).reshape(n_rows, n_cols)
    return [[round(float(x), decimals) for x in row] for row in arr]


def shock_grid(
    spot_shocks: Sequence[float],
    vol_shocks: Sequence[float],
    pnl_fn: Callable[[np.ndarray, np.ndarray], Sequence[float]],
    decimals: int = 2,
) -> list[list[float]]:
    """P&L grid with rows = spot shocks and columns = vol shocks.

    pnl_fn receives the flattened spot and vol shocks (see shock_mesh) and returns P&L in the same order,
    so it can be vectorised.
    """
    spot_flat, vol_flat = shock_mesh(spot_shocks, vol_shocks)
    return to_grid(pnl_fn(spot_flat, vol_flat), len(spot_shocks), len(vol_shocks), decimals)
