from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


GRID_ROWS = 2400
GRID_COLS = 7200
LAT_MIN, LAT_MAX = -60.0, 60.0
LON_MIN, LON_MAX = -180.0, 180.0
CELL_SIZE = 0.05
MARS_EQC_CRS = (
    'PROJCS["Mars2000_equicylindrical_clon0",'
    'GEOGCS["GCS_Mars_2000_Sphere",'
    'DATUM["D_Mars_2000_Sphere",'
    'SPHEROID["Mars_2000_Sphere_IAU_IAG",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],'
    'PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)


def lat_lon_to_cell(lat: float, lon: float) -> tuple[int, int]:
    row = int((LAT_MAX - lat) / CELL_SIZE)
    col = int((lon - LON_MIN) / CELL_SIZE)
    return min(max(row, 0), GRID_ROWS - 1), min(max(col, 0), GRID_COLS - 1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float(np.median(v))
    cdf = np.cumsum(w) / w_sum
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def grid_measurements(
    lats: np.ndarray,
    lons: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = (
        np.isfinite(lats)
        & np.isfinite(lons)
        & np.isfinite(values)
        & np.isfinite(weights)
        & (lats >= LAT_MIN)
        & (lats <= LAT_MAX)
        & (lons >= LON_MIN)
        & (lons <= LON_MAX)
    )
    if not np.any(valid):
        return grid

    cell_data: dict[tuple[int, int], list[int]] = {}
    valid_idx = np.where(valid)[0]
    for i in valid_idx.tolist():
        rc = lat_lon_to_cell(float(lats[i]), float(lons[i]))
        cell_data.setdefault(rc, []).append(i)

    for (row, col), idxs in cell_data.items():
        idx = np.asarray(idxs, dtype=np.int32)
        cell_vals = values[idx]
        cell_w = np.maximum(weights[idx], 0.0)
        n = idx.size

        if n >= 3:
            agg = _weighted_median(cell_vals, cell_w)
        elif n >= 1:
            w_sum = float(np.sum(cell_w))
            if w_sum > 0:
                agg = float(np.sum(cell_vals * cell_w) / w_sum)
            else:
                agg = float(np.mean(cell_vals))
        else:
            agg = 0.0
        grid[row, col] = np.float32(agg)

    return grid


def write_geotiff(grid: np.ndarray, output_path: str, description: str = "") -> None:
    arr = np.asarray(grid, dtype=np.float32)
    if arr.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"Expected grid shape {(GRID_ROWS, GRID_COLS)}, got {arr.shape}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, GRID_COLS, GRID_ROWS)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=GRID_ROWS,
        width=GRID_COLS,
        count=1,
        dtype="float32",
        crs=MARS_EQC_CRS,
        transform=transform,
    ) as ds:
        ds.write(arr, 1)
        if description:
            ds.set_band_description(1, description)
