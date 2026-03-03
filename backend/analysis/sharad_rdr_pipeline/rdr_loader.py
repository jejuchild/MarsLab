from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ..shared.constants import N_SHARAD_RANGE_BINS


ROW_BYTES = 5822
ECHO_REAL_OFFSET = 194
ECHO_IMAG_OFFSET = 2862
ECHO_BYTES = N_SHARAD_RANGE_BINS * 4
LON_OFFSET = 5637
LAT_OFFSET = 5645
ALT_OFFSET = 5629

_BACKEND_DIR = Path(__file__).resolve().parents[2]
SHARAD_HR_DIR = _BACKEND_DIR / "sharad_highres"


@dataclass
class TrackData:
    product_id: str
    power: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    surface_bins: np.ndarray
    n_traces: int


def _lon_to_180(lon: np.ndarray) -> np.ndarray:
    return np.where(lon > 180.0, lon - 360.0, lon)


def _cache_dir(product_id: str) -> Path:
    return SHARAD_HR_DIR / ".cache" / product_id.upper()


def _resolve_dat_path(product_id: str) -> Path:
    pid = product_id.upper()
    direct = SHARAD_HR_DIR / f"{pid.lower()}.dat"
    if direct.exists():
        return direct

    pattern = re.compile(re.escape(pid), re.IGNORECASE)
    for path in sorted(SHARAD_HR_DIR.glob("*.dat")):
        if pattern.search(path.stem):
            return path
    raise FileNotFoundError(f"No .dat file found for product_id={pid}")


def _parse_and_cache_product(product_id: str, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dat_path = _resolve_dat_path(product_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(dat_path, dtype=np.uint8)
    n_traces = raw.size // ROW_BYTES
    if n_traces <= 0:
        raise ValueError(f"No rows parsed from {dat_path}")
    raw = raw[: n_traces * ROW_BYTES].reshape(n_traces, ROW_BYTES)

    real = np.frombuffer(
        raw[:, ECHO_REAL_OFFSET : ECHO_REAL_OFFSET + ECHO_BYTES].tobytes(),
        dtype="<f4",
    ).reshape(n_traces, N_SHARAD_RANGE_BINS)
    imag = np.frombuffer(
        raw[:, ECHO_IMAG_OFFSET : ECHO_IMAG_OFFSET + ECHO_BYTES].tobytes(),
        dtype="<f4",
    ).reshape(n_traces, N_SHARAD_RANGE_BINS)
    power = (real**2 + imag**2).astype(np.float32)
    if not np.isfinite(power).all():
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    lon = np.frombuffer(raw[:, LON_OFFSET : LON_OFFSET + 8].tobytes(), dtype="<f8").copy()
    lat = np.frombuffer(raw[:, LAT_OFFSET : LAT_OFFSET + 8].tobytes(), dtype="<f8").copy()
    alt_m = np.frombuffer(raw[:, ALT_OFFSET : ALT_OFFSET + 8].tobytes(), dtype="<f8").copy()

    np.save(cache_dir / "power.npy", power)
    np.savez_compressed(cache_dir / "geometry.npz", lat=lat, lon=lon, alt=alt_m)
    return power, lat, lon, alt_m


def _load_power_geometry(product_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_dir = _cache_dir(product_id)
    power_path = cache_dir / "power.npy"
    geom_path = cache_dir / "geometry.npz"

    if power_path.exists() and geom_path.exists():
        power = np.load(power_path, mmap_mode="r")
        geom = np.load(geom_path)
        lat = np.asarray(geom["lat"], dtype=np.float64)
        lon = np.asarray(geom["lon"], dtype=np.float64)
        alt_m = np.asarray(geom["alt"], dtype=np.float64)
        return power, lat, lon, alt_m

    return _parse_and_cache_product(product_id, cache_dir)


def _fallback_pick_surface(power: np.ndarray) -> np.ndarray:
    n_traces, _ = power.shape
    coarse = np.argmax(power[:, :120], axis=1)
    surface = np.full(n_traces, -1, dtype=np.int32)

    for i in range(n_traces):
        c = int(coarse[i])
        lo = min(c + 15, N_SHARAD_RANGE_BINS - 1)
        hi = min(c + 250, N_SHARAD_RANGE_BINS)
        if hi <= lo:
            continue

        band = np.asarray(power[i, lo:hi], dtype=np.float64)
        noise = float(np.median(band)) + 1e-12
        peak = int(np.argmax(band))
        if band[peak] / noise >= 3.0:
            surface[i] = lo + peak

    return surface


def list_available_products() -> list[str]:
    if not SHARAD_HR_DIR.exists():
        return []
    return sorted(path.stem.upper() for path in SHARAD_HR_DIR.glob("*.dat"))


def load_track_data(product_id: str) -> TrackData | None:
    pid = product_id.upper()
    try:
        power, lat, lon, alt_m = _load_power_geometry(pid)
    except FileNotFoundError:
        return None

    cache_dir = _cache_dir(pid)
    surface_path = cache_dir / "surface_v3.npy"
    if surface_path.exists():
        surface_bins = np.load(surface_path).astype(np.int32, copy=False)
    else:
        surface_bins = _fallback_pick_surface(np.asarray(power, dtype=np.float32))
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(surface_path, surface_bins)

    n_traces = min(
        int(power.shape[0]),
        int(lat.shape[0]),
        int(lon.shape[0]),
        int(alt_m.shape[0]),
        int(surface_bins.shape[0]),
    )

    lon180 = _lon_to_180(np.asarray(lon[:n_traces], dtype=np.float64))
    alt_km = np.asarray(alt_m[:n_traces], dtype=np.float64) / 1000.0

    return TrackData(
        product_id=pid,
        power=np.asarray(power[:n_traces]),
        lat=np.asarray(lat[:n_traces], dtype=np.float64),
        lon=lon180,
        alt=alt_km,
        surface_bins=np.asarray(surface_bins[:n_traces], dtype=np.int32),
        n_traces=n_traces,
    )
