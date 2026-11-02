"""Tests for CRISM TRR3 sensor-geometry → north-up lat/lon georeferencing."""

import numpy as np
import pytest

from api.mineral_cnn.georef import GeoGrid, _regular_rows, warp_to_latlon


def _sensor_grid(rows=40, cols=60):
    # Like the real data: row 0 is the SOUTH edge, sample 0 is the EAST edge.
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    lat = 10.0 + r * 0.001
    lon = 20.0 - c * 0.001
    return GeoGrid(lat=lat.astype(float), lon=lon.astype(float))


def test_output_is_north_up_and_west_left():
    grid = _sensor_grid()
    rows, cols = grid.lat.shape
    # Brightness encodes position: bright in the north-west corner only.
    image = np.zeros((rows, cols), dtype=np.uint8)
    image[-5:, -5:] = 255  # last rows = north, last samples = west
    image[:-5, :] = 50
    image[-5:, :-5] = 50

    rgba, bounds = warp_to_latlon(image, np.ones_like(image, bool), grid)

    assert rgba[:3, :3, 0].mean() > 200                      # bright NW in top-left
    assert rgba[-3:, -3:, 0].mean() < 100                    # SE stays dark
    assert bounds["west"] == pytest.approx(20.0 - (cols - 1) * 0.001)
    assert bounds["east"] == pytest.approx(20.0)
    assert bounds["south"] == pytest.approx(10.0)
    assert bounds["north"] == pytest.approx(10.0 + (rows - 1) * 0.001)


def test_nodata_is_transparent_but_extent_unchanged():
    grid = _sensor_grid()
    image = np.full(grid.lat.shape, 128, dtype=np.uint8)
    valid = np.ones_like(image, bool)
    valid[:, :10] = False  # an invalid strip along one edge

    _, full = warp_to_latlon(image, np.ones_like(valid), grid)
    rgba, partial = warp_to_latlon(image, valid, grid)

    assert partial == full                       # box still matches the DDR extent
    assert (rgba[..., 3] == 0).any()             # invalid strip is transparent
    assert (rgba[..., 3] == 255).any()


def test_non_tracking_rows_are_excluded():
    grid = _sensor_grid(rows=50)
    # Last 8 rows jump 30x the normal along-track spacing (instrument stopped tracking).
    for i in range(42, 50):
        grid.lat[i, :] = grid.lat[41, :] + (i - 41) * 0.03

    mask = _regular_rows(grid)
    assert mask[:42].all() and not mask[42:].any()

    _, bounds = warp_to_latlon(np.full(grid.lat.shape, 100, np.uint8), np.ones(grid.lat.shape, bool), grid)
    assert bounds["north"] == pytest.approx(10.0 + 41 * 0.001)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        warp_to_latlon(np.zeros((5, 5), np.uint8), np.ones((5, 5), bool), _sensor_grid())
