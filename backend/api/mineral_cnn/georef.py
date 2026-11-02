"""
Georeference CRISM TRR3 sensor-space images onto a north-up lat/lon grid.

TRR3 cubes (and the CNN mineral maps derived from them) are in sensor
geometry: rows run south→north and samples run east→west for these
observations, so draping the raw array onto a north-up rectangle shows it
rotated/mirrored relative to the terrain. The matching DDR file carries the
areocentric latitude/longitude of every pixel; we use it to resample the
image onto an equirectangular grid whose extent is exactly the DDR lat/lon
min/max — the same bounding box the frontend gets from the TRR3 index
(whose polygons are traced from the DDR edges).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .data_loader import parse_pds3_label

_DDR_LAT_BAND = 3  # "Latitude, areocentric, deg N"
_DDR_LON_BAND = 4  # "Longitude, areocentric, deg E"
_DDR_FILL = 65534.0
_MAX_OUTPUT_DIM = 2048


@dataclass
class GeoGrid:
    lat: np.ndarray  # (rows, cols) float64, NaN where invalid
    lon: np.ndarray  # (rows, cols) float64 in [-180, 180], NaN where invalid


def find_ddr(obs_dir: str) -> Optional[tuple[str, str]]:
    """Return (img, lbl) paths of the DDR1 product in an observation folder."""
    img = lbl = None
    for fname in os.listdir(obs_dir):
        upper = fname.upper()
        if upper.endswith("_DDR1.IMG"):
            img = os.path.join(obs_dir, fname)
        elif upper.endswith("_DDR1.LBL"):
            lbl = os.path.join(obs_dir, fname)
    return (img, lbl) if img and lbl else None


def load_ddr_latlon(ddr_img: str, ddr_lbl: str) -> GeoGrid:
    meta = parse_pds3_label(ddr_lbl)
    rows, cols, bands = int(meta["LINES"]), int(meta["LINE_SAMPLES"]), int(meta["BANDS"])
    raw = np.fromfile(ddr_img, dtype=np.float32)[: rows * cols * bands]
    cube = raw.reshape(bands, rows, cols).astype(np.float64)
    lat, lon = cube[_DDR_LAT_BAND], cube[_DDR_LON_BAND]
    bad = ~np.isfinite(lat) | ~np.isfinite(lon) | (lat >= _DDR_FILL) | (lon >= _DDR_FILL)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lat = np.where(bad, np.nan, lat)
    lon = np.where(bad, np.nan, lon)
    return GeoGrid(lat=lat, lon=lon)


def _regular_rows(grid: GeoGrid, tolerance: float = 5.0) -> np.ndarray:
    """
    Boolean mask of rows belonging to the longest run of regularly spaced rows.

    Some observations end with rows where the instrument stopped tracking the
    ground: consecutive rows jump ~30x the normal along-track spacing. Those
    rows stretch the footprint by tens of km and must not be georeferenced.
    """
    rows = grid.lat.shape[0]
    with np.errstate(all="ignore"):
        lat = np.nanmedian(grid.lat, axis=1)
        lon = np.nanmedian(grid.lon, axis=1)
    k = np.cos(np.radians(np.nanmedian(lat)))
    step = np.hypot(np.diff(lat), np.diff(lon) * k)
    finite = np.isfinite(step)
    if not finite.any():
        return np.isfinite(lat)
    typical = np.median(step[finite])
    good_step = finite & (step <= tolerance * max(typical, 1e-9))

    best_start, best_len, start = 0, 1, 0
    for i in range(rows - 1):
        if not good_step[i]:
            start = i + 1
        length = i + 2 - start
        if good_step[i] and length > best_len:
            best_start, best_len = start, length
    mask = np.zeros(rows, dtype=bool)
    mask[best_start:best_start + best_len] = True
    return mask & np.isfinite(lat)


def warp_to_latlon(image: np.ndarray, valid: np.ndarray, grid: GeoGrid) -> tuple[np.ndarray, dict]:
    """
    Resample a sensor-space image onto a north-up lat/lon grid.

    Parameters
    ----------
    image : (rows, cols) or (rows, cols, C) uint8 array in sensor geometry.
    valid : (rows, cols) bool mask of pixels that carry data.
    grid  : per-pixel lat/lon from the DDR (same rows/cols as ``image``).

    Returns
    -------
    rgba : (H, W, 4) uint8, row 0 = north, column 0 = west, alpha 0 = no data.
    bounds : {"west", "south", "east", "north"} in degrees.
    """
    if image.shape[:2] != grid.lat.shape:
        raise ValueError(f"image {image.shape[:2]} and DDR {grid.lat.shape} shapes differ")

    # Extent and lookup use every georeferenced pixel (so the output box equals
    # the DDR/index bounding box); image nodata just becomes transparent.
    ok = np.isfinite(grid.lat) & np.isfinite(grid.lon)
    ok &= _regular_rows(grid)[:, None]
    if ok.sum() < 4:
        raise ValueError("not enough georeferenced pixels")

    lat, lon = grid.lat[ok], grid.lon[ok]
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    if east - west > 180.0:
        raise ValueError("observation crosses the antimeridian; not supported")

    # Planar coordinates (degrees, longitude scaled by cos(lat)) so distances
    # are roughly isotropic for the nearest-neighbour lookup.
    k = np.cos(np.radians(0.5 * (south + north)))
    src = np.column_stack([lon * k, lat])
    tree = cKDTree(src)

    # Typical source pixel spacing → output resolution (and the max distance
    # at which an output pixel still counts as covered).
    spacing = float(np.median(tree.query(src[:: max(1, len(src) // 2000)], k=2)[0][:, 1]))
    spacing = max(spacing, 1e-7)
    width = int(np.clip(round((east - west) * k / spacing), 2, _MAX_OUTPUT_DIM))
    height = int(np.clip(round((north - south) / spacing), 2, _MAX_OUTPUT_DIM))

    # Output pixel centres.
    xs = west + (np.arange(width) + 0.5) * (east - west) / width
    ys = north - (np.arange(height) + 0.5) * (north - south) / height
    gx, gy = np.meshgrid(xs * k, ys)
    dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    covered = dist <= 1.5 * spacing

    src_vals = image[ok]
    if src_vals.ndim == 1:  # grayscale → RGBA
        src_rgba = np.stack([src_vals, src_vals, src_vals, np.full_like(src_vals, 255)], axis=-1)
    elif src_vals.shape[1] == 3:
        src_rgba = np.concatenate([src_vals, np.full((len(src_vals), 1), 255, np.uint8)], axis=1)
    else:
        src_rgba = src_vals

    src_rgba = src_rgba.copy()
    src_rgba[~valid[ok], 3] = 0

    out = np.zeros((height * width, 4), dtype=np.uint8)
    out[covered] = src_rgba[idx[covered]]
    return out.reshape(height, width, 4), {"west": west, "south": south, "east": east, "north": north}


class PixelLocator:
    """Map lat/lon to the nearest sensor (line, sample) of an observation."""

    def __init__(self, grid: GeoGrid):
        ok = np.isfinite(grid.lat) & np.isfinite(grid.lon)
        ok &= _regular_rows(grid)[:, None]
        if ok.sum() < 4:
            raise ValueError("not enough georeferenced pixels")
        self._lines, self._samples = np.nonzero(ok)
        lat, lon = grid.lat[ok], grid.lon[ok]
        self._k = float(np.cos(np.radians(np.median(lat))))
        self._tree = cKDTree(np.column_stack([lon * self._k, lat]))
        sample_idx = slice(None, None, max(1, len(lat) // 2000))
        self._spacing = float(np.median(self._tree.query(self._tree.data[sample_idx], k=2)[0][:, 1]))

    def locate(self, lat: float, lon: float) -> dict:
        lon = lon - 360.0 if lon > 180.0 else lon
        dist, i = self._tree.query([lon * self._k, lat], k=1)
        mars_km_per_deg = 3389.5 * np.pi / 180.0
        return {
            "line": int(self._lines[i]),
            "sample": int(self._samples[i]),
            "distance_m": round(float(dist) * mars_km_per_deg * 1000.0, 1),
            "inside": bool(dist <= 1.5 * self._spacing),
        }
