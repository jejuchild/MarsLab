"""
SHARAD subsurface interface time pick aligned to crater location.

Picks the subsurface reflector two-way travel time from the SHARAD radargram
at the trace(s) nearest to a terraced crater, and provides uncertainty.
"""

import os
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# SHARAD timing constants
SHARAD_SAMPLE_INTERVAL_US = 3.0 / 80.0  # 0.0375 µs per range bin
SPEED_OF_LIGHT = 299792458.0  # m/s

# Mars ellipsoid
_MARS_AB2 = (3396190.0 / 3376200.0) ** 2


@dataclass
class SharadPick:
    """A subsurface interface pick from SHARAD."""
    product_id: str
    trace_idx: int
    lat: float
    lon: float
    surface_bin: int
    interface_bin: int
    delta_bins: int
    twt_us: float          # Two-way travel time in microseconds
    twt_unc_us: float      # Uncertainty in twt
    snr: float             # Signal-to-noise ratio of the pick
    along_track_km: float  # Distance along track from start
    distance_to_crater_km: float


@dataclass
class SharadPickResult:
    """Result of SHARAD interface picking near a crater."""
    crater_lat: float
    crater_lon: float
    picks: List[SharadPick] = field(default_factory=list)
    median_twt_us: float = 0.0
    twt_unc_us: float = 0.0
    error: Optional[str] = None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance on Mars in km."""
    R = 3389.5  # Mars mean radius in km
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def _centric_to_graphic(lat_centric: float) -> float:
    return math.degrees(math.atan(_MARS_AB2 * math.tan(math.radians(lat_centric))))


def pick_subsurface_interface(
    sharad_product_id: str,
    crater_lat: float,
    crater_lon: float,
    window_km: float = 5.0,
    min_snr: float = 3.5,
    search_lo_bins: int = 20,
    search_hi_bins: int = 200,
) -> SharadPickResult:
    """
    Pick subsurface interface two-way travel time from SHARAD radargram
    at traces nearest to a crater location.

    Args:
        sharad_product_id: SHARAD high-res product ID
        crater_lat/lon: Crater center in geographic coordinates (-180 to 180 lon)
        window_km: Search window ±km along track centered on nearest trace
        min_snr: Minimum SNR for a valid subsurface pick
        search_lo_bins: Min bins below surface to search
        search_hi_bins: Max bins below surface to search

    Returns:
        SharadPickResult with picks and median/uncertainty
    """
    result = SharadPickResult(crater_lat=crater_lat, crater_lon=crater_lon)

    try:
        from backend.api.sharad_highres_router import (
            _get_power, _get_geometry, _pick_surface, _lon_to_180,
        )
    except ImportError:
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from api.sharad_highres_router import (
                _get_power, _get_geometry, _pick_surface, _lon_to_180,
            )
        except ImportError as e:
            result.error = f"Cannot import SHARAD router: {e}"
            return result

    try:
        power, n_traces = _get_power(sharad_product_id)
        geom, _ = _get_geometry(sharad_product_id)
        surface = _pick_surface(sharad_product_id, power)
    except FileNotFoundError as e:
        result.error = f"SHARAD data not found: {e}"
        return result

    lons180 = _lon_to_180(geom["lon"])
    lats_graphic = np.array([
        _centric_to_graphic(float(lat)) for lat in geom["lat"]
    ])

    # Compute along-track cumulative distance
    cum_dist_km = np.zeros(n_traces)
    for i in range(1, n_traces):
        cum_dist_km[i] = cum_dist_km[i - 1] + _haversine_km(
            float(lats_graphic[i - 1]), float(lons180[i - 1]),
            float(lats_graphic[i]), float(lons180[i]),
        )

    # Find traces within window_km of the crater
    dists = np.array([
        _haversine_km(crater_lat, crater_lon, float(lats_graphic[i]), float(lons180[i]))
        for i in range(n_traces)
    ])
    nearest_idx = int(np.argmin(dists))
    nearest_dist_km = float(dists[nearest_idx])

    # Select traces within window
    half_window_km = window_km
    near_along_track = np.abs(cum_dist_km - cum_dist_km[nearest_idx]) <= half_window_km
    # Also require within reasonable lateral distance
    near_mask = near_along_track & (dists <= half_window_km * 2)

    candidate_indices = np.where(near_mask & (surface >= 0))[0]

    if len(candidate_indices) == 0:
        result.error = f"No valid surface traces within {window_km}km of crater (nearest={nearest_dist_km:.1f}km)"
        return result

    n_bins = power.shape[1]
    picks = []

    for idx in candidate_indices:
        s_bin = int(surface[idx])
        lo = min(s_bin + search_lo_bins, n_bins - 1)
        hi = min(s_bin + search_hi_bins, n_bins)
        if hi <= lo:
            continue

        band = power[idx, lo:hi].astype(np.float64)
        noise = float(np.median(band)) + 1e-12
        peak_pos = int(np.argmax(band))
        peak_val = float(band[peak_pos])
        snr = peak_val / noise

        if snr >= min_snr:
            iface_bin = lo + peak_pos
            delta_bins = iface_bin - s_bin
            twt_us = delta_bins * SHARAD_SAMPLE_INTERVAL_US

            picks.append(SharadPick(
                product_id=sharad_product_id,
                trace_idx=int(idx),
                lat=float(lats_graphic[idx]),
                lon=float(lons180[idx]),
                surface_bin=s_bin,
                interface_bin=iface_bin,
                delta_bins=delta_bins,
                twt_us=twt_us,
                twt_unc_us=SHARAD_SAMPLE_INTERVAL_US,  # ±1 bin as minimum
                snr=round(snr, 2),
                along_track_km=round(float(cum_dist_km[idx]), 2),
                distance_to_crater_km=round(float(dists[idx]), 2),
            ))

    if not picks:
        result.error = f"No subsurface reflectors found (SNR>{min_snr}) near crater"
        return result

    # Coherence filter: keep picks with consistent twt
    twt_values = np.array([p.twt_us for p in picks])
    if len(twt_values) >= 3:
        med = float(np.median(twt_values))
        mad = float(np.median(np.abs(twt_values - med)))
        # Keep within 3 MAD of median
        threshold = max(mad * 3, SHARAD_SAMPLE_INTERVAL_US * 3)
        coherent = [p for p in picks if abs(p.twt_us - med) <= threshold]
        if len(coherent) >= 2:
            picks = coherent

    result.picks = picks
    twt_arr = np.array([p.twt_us for p in picks])
    result.median_twt_us = round(float(np.median(twt_arr)), 4)
    # Uncertainty: MAD or std, whichever is larger
    if len(twt_arr) >= 3:
        mad = float(np.median(np.abs(twt_arr - np.median(twt_arr))))
        std = float(np.std(twt_arr))
        result.twt_unc_us = round(max(mad, std, SHARAD_SAMPLE_INTERVAL_US), 4)
    else:
        result.twt_unc_us = round(SHARAD_SAMPLE_INTERVAL_US * 2, 4)

    return result
