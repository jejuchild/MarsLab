from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..shared.constants import N_SHARAD_RANGE_BINS
from .rdr_loader import TrackData


@dataclass
class SurfacePowerResult:
    lat: np.ndarray
    lon: np.ndarray
    p_corr_db: np.ndarray
    snr: np.ndarray
    product_id: str


def compute_surface_power(track: TrackData) -> SurfacePowerResult | None:
    valid = np.where(track.surface_bins >= 0)[0]
    if valid.size == 0:
        return None

    lat_out: list[float] = []
    lon_out: list[float] = []
    p_corr_out: list[float] = []
    snr_out: list[float] = []
    h_ref = 300.0

    for t in valid.tolist():
        sb = int(track.surface_bins[t])
        lo = max(0, sb - 2)
        hi = min(N_SHARAD_RANGE_BINS, sb + 3)
        if hi <= lo:
            continue

        p_surf = float(np.max(track.power[t, lo:hi]))
        n_lo = min(N_SHARAD_RANGE_BINS, sb + 100)
        n_hi = min(N_SHARAD_RANGE_BINS, sb + 200)
        if n_hi <= n_lo:
            continue
        p_noise = float(np.median(track.power[t, n_lo:n_hi]))
        snr = p_surf / (p_noise + 1e-20)
        if not np.isfinite(snr) or snr < 5.0:
            continue

        alt_km = float(track.alt[t])
        if not np.isfinite(alt_km) or alt_km <= 0:
            continue

        p_db = 10.0 * np.log10(p_surf + 1e-20)
        p_corr_db = p_db + 20.0 * np.log10(alt_km / h_ref)
        if not np.isfinite(p_corr_db):
            continue

        lat_out.append(float(track.lat[t]))
        lon_out.append(float(track.lon[t]))
        p_corr_out.append(float(p_corr_db))
        snr_out.append(float(snr))

    if not lat_out:
        return None

    return SurfacePowerResult(
        lat=np.asarray(lat_out, dtype=np.float64),
        lon=np.asarray(lon_out, dtype=np.float64),
        p_corr_db=np.asarray(p_corr_out, dtype=np.float64),
        snr=np.asarray(snr_out, dtype=np.float64),
        product_id=track.product_id,
    )


def build_latitude_reference(results: list[SurfacePowerResult]) -> Callable[[np.ndarray | float], np.ndarray | float]:
    if not results:
        return lambda lat: np.zeros_like(np.asarray(lat, dtype=np.float64))

    all_lat = np.concatenate([r.lat for r in results if r.lat.size > 0])
    all_power = np.concatenate([r.p_corr_db for r in results if r.p_corr_db.size > 0])
    if all_lat.size == 0:
        return lambda lat: np.zeros_like(np.asarray(lat, dtype=np.float64))

    band_values = np.full(36, np.nan, dtype=np.float64)
    band_idx = np.floor((all_lat + 90.0) / 5.0).astype(int)
    band_idx = np.clip(band_idx, 0, 35)
    for i in range(36):
        m = band_idx == i
        if np.any(m):
            band_values[i] = float(np.median(all_power[m]))

    global_median = float(np.median(all_power))

    def ref(lat: np.ndarray | float) -> np.ndarray | float:
        lat_arr = np.asarray(lat, dtype=np.float64)
        idx = np.floor((lat_arr + 90.0) / 5.0).astype(int)
        idx = np.clip(idx, 0, 35)
        out = band_values[idx]
        out = np.where(np.isfinite(out), out, global_median)
        if np.isscalar(lat):
            return float(out)
        return out

    return ref


def score_surface_consistency(
    result: SurfacePowerResult,
    ref_func: Callable[[np.ndarray | float], np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_vals = np.asarray(ref_func(result.lat), dtype=np.float64)
    excess_db = result.p_corr_db - ref_vals
    consistency = np.clip(excess_db / 3.0, -1.0, 1.0)
    return result.lat, result.lon, consistency.astype(np.float32), result.snr
