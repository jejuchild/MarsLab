from __future__ import annotations

from dataclasses import dataclass

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


def build_global_power_stats(
    results: list[SurfacePowerResult],
) -> tuple[float, float]:
    """Compute global mean and std of corrected power across ALL tracks.

    SWIM2 RS methodology (swim.psi.edu/SWIM2Products.php):
    Consistency values are assigned according to the **global** power
    distribution expressed in sigma (standard deviation) units.

    Returns (mean_dB, std_dB) of the pooled corrected-power distribution.
    """
    if not results:
        return 0.0, 1.0
    all_power = np.concatenate(
        [r.p_corr_db for r in results if r.p_corr_db.size > 0]
    )
    if all_power.size == 0:
        return 0.0, 1.0
    mu = float(np.mean(all_power))
    sigma = float(np.std(all_power))
    if sigma <= 0:
        sigma = 1.0
    return mu, sigma


def score_surface_consistency(
    result: SurfacePowerResult,
    mu: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign SWIM2 discrete RS consistency scores.

    SWIM2 rubric (swim.psi.edu/SWIM2Products.php):
      z < -1 sigma         ->  +1.0  (very low power, consistent with ice)
      -1 sigma <= z < -0.5 sigma  ->  +0.5
      -0.5 sigma <= z < 0.5 sigma ->   0.0  (inconclusive)
      0.5 sigma <= z < 1 sigma    ->  -0.5
      z >= 1 sigma         ->  -1.0  (very high power, inconsistent with ice)

    Note: the sign is *inverted* vs. power: low power = ice = positive.
    """
    z = (result.p_corr_db - mu) / sigma

    # SWIM2 5-level discrete rubric
    consistency = np.zeros(z.size, dtype=np.float32)
    consistency[z < -1.0] = 1.0
    consistency[(z >= -1.0) & (z < -0.5)] = 0.5
    # -0.5 <= z < 0.5 -> 0.0 (already initialised)
    consistency[(z >= 0.5) & (z < 1.0)] = -0.5
    consistency[z >= 1.0] = -1.0

    return result.lat, result.lon, consistency, result.snr
