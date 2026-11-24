"""Risk-neutral density implied by an SVI smile (Breeden-Litzenberger, via QuantLib's SmileSection.density)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import trapezoid

from src.quant.svi import svi_smile_section

DEFAULT_DENSITY_K_GRID = np.linspace(-0.18, 0.14, 240)
# Finite-difference strike step of QuantLib's density, as a fraction of the forward
DENSITY_GAP_FRACTION = 1e-4


def breeden_litzenberger_density(
    forward: float,
    ttm: float,
    svi_params: Sequence[float],
    k_grid=None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """(strikes, q(K)): second strike derivative of undiscounted SVI call prices, so the rate drops out.

    Values are clipped at zero and, with normalize, rescaled to integrate to 1 over the grid. Raises RuntimeError
    (QuantLib) for SVI parameters outside QuantLib's admissible set.
    """
    k = DEFAULT_DENSITY_K_GRID if k_grid is None else np.asarray(k_grid, dtype=float)
    strikes = forward * np.exp(k)
    section = svi_smile_section(svi_params, ttm, forward)
    gap = DENSITY_GAP_FRACTION * forward
    density = np.maximum([section.density(float(strike), 1.0, gap) for strike in strikes], 0.0)
    if normalize:
        area = density_area(strikes, density)
        if area > 0:
            density = density / area
    return strikes, density


def density_area(strikes: np.ndarray, density: np.ndarray) -> float:
    return float(trapezoid(density, strikes))
