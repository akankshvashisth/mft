import numpy as np

from src.quant.stress import shock_grid, shock_mesh, to_grid


def test_mesh_is_row_major_with_spot_shocks_as_rows():
    spot, vol = shock_mesh([-1.0, 0.0, 1.0], [-2.0, 2.0])
    np.testing.assert_array_equal(spot, [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(vol, [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0])


def test_grid_rows_are_spot_shocks_and_columns_are_vol_shocks():
    grid = shock_grid([-1.0, 0.0, 1.0], [-2.0, 2.0], lambda s, v: 10.0 * s + v)
    assert grid == [[-12.0, -8.0], [-2.0, 2.0], [8.0, 12.0]]


def test_pnl_function_is_called_once_with_flat_vectors():
    calls = []

    def pnl(spot, vol):
        calls.append((spot.shape, vol.shape))
        return np.zeros_like(spot)

    shock_grid([1.0, 2.0], [3.0, 4.0, 5.0], pnl)
    assert calls == [((6,), (6,))]


def test_grid_values_are_rounded_plain_floats():
    grid = to_grid([1.0 / 3.0, 2.0 / 3.0], 1, 2)
    assert grid == [[0.33, 0.67]]
    assert all(type(x) is float for row in grid for x in row)
    assert to_grid([1.23456], 1, 1, decimals=3) == [[1.235]]
