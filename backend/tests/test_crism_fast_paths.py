"""Fast CRISM paths must give exactly the same output as the full-cube ones."""

import numpy as np

from api.crism.rgb import IGNORE_VALUE, make_rgb, nearest_band, stretch_rgb


def _cube(seed=0):
    rng = np.random.default_rng(seed)
    cube = rng.uniform(-0.05, 0.4, size=(40, 30, 20)).astype(np.float32)
    cube[:, 0, :] = IGNORE_VALUE          # an ignored line
    cube[5, 3, 4] = np.nan
    return cube, np.linspace(0.4, 3.9, 40).astype(np.float32)


def test_three_band_stretch_equals_full_cube_rgb():
    cube, wl = _cube()
    args = (2.53, 1.51, 1.08, 0.02, 0.25)
    full = make_rgb(cube, wl, *args)
    idx = [nearest_band(wl, um) for um in args[:3]]
    fast = stretch_rgb(cube[idx[0]], cube[idx[1]], cube[idx[2]], *args[3:])
    assert np.array_equal(full, fast, equal_nan=True)


def test_stretch_does_not_modify_inputs():
    cube, _ = _cube(1)
    band = cube[0].copy()
    stretch_rgb(cube[0], cube[1], cube[2], 0.02, 0.25)
    assert np.array_equal(cube[0], band, equal_nan=True)
