import math

import numpy as np
import pytest
from scipy.stats import lognorm

from src.quant.density import breeden_litzenberger_density, density_area
from tests.factories import SVI_PARAMS

T = 30 / 365.25
FORWARD = 700.0
FLAT_VOL = 0.20
FLAT_SMILE = (FLAT_VOL ** 2 * T, 0.0, 0.0, 0.0, 0.1)  # b = 0: total variance a at every strike
WIDE_GRID = np.linspace(-0.5, 0.5, 801)


def test_flat_smile_density_is_the_black_lognormal():
    strikes, density = breeden_litzenberger_density(FORWARD, T, FLAT_SMILE, k_grid=WIDE_GRID, normalize=False)

    std_dev = FLAT_VOL * math.sqrt(T)
    expected = lognorm.pdf(strikes, std_dev, scale=FORWARD * math.exp(-0.5 * std_dev ** 2))
    np.testing.assert_allclose(density, expected, rtol=1e-4, atol=1e-9)


def test_density_is_non_negative_and_normalised_on_the_default_grid():
    strikes, density = breeden_litzenberger_density(FORWARD, T, SVI_PARAMS)

    assert strikes.shape == density.shape == (240,)
    assert np.all(np.diff(strikes) > 0)
    assert np.all(density >= 0.0)
    assert density_area(strikes, density) == pytest.approx(1.0, rel=1e-9)


def test_inadmissible_svi_parameters_are_rejected():
    with pytest.raises(RuntimeError):
        breeden_litzenberger_density(FORWARD, T, (-0.1, 0.01, 0.0, 0.0, 0.1))
