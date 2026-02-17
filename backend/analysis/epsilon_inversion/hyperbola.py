"""Hyperbola fitting for εr estimation via NMO (Normal Moveout) analysis."""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares

from analysis.shared.constants import SPEED_OF_LIGHT, SHARAD_SAMPLE_INTERVAL_US

logger = logging.getLogger(__name__)


@dataclass
class HyperbolaFitResult:
    """Result of hyperbola fitting."""

    epsilon_r_est: Optional[float]  # Estimated εr, None if fit failed
    epsilon_r_std: Optional[float]  # Uncertainty in estimate
    rmse_bins: float  # RMS error of residuals (in bins)
    inlier_ratio: float  # Fraction of picks that are inliers (0–1)
    n_inliers: int  # Number of inlier picks used
    quality_note: str  # Human-readable fit quality
    diagnostics: Dict  # Curve parameters, residuals, etc.


class HyperbolaFitter:
    """Fit hyperbolic NMO curve to reflector picks and estimate εr."""

    def fit(
        self,
        product_id: str,
        trace_idx0: int,
        bin_idx0: int,
        window_traces: int = 150,
        window_bins: int = 120,
    ) -> HyperbolaFitResult:
        """Fit NMO hyperbola and estimate εr from velocity.

        Algorithm (simplified):
        1. Load SHARAD power data around (trace_idx0, bin_idx0)
        2. Define trace offsets relative to center: x_i = (i - idx0) * trace_spacing_km
        3. For each trace, search for peak near predicted hyperbola: t(x) = sqrt(t0^2 + (x/v)^2)
        4. Collect picks (x_i, t_i)
        5. Non-linear LS fit: minimize Σ(t_i - t(x_i; v, t0))^2
        6. Outlier rejection: remove picks with residual > 2*std
        7. Refit on inliers
        8. Compute covariance → εr_std
        9. Convert v → εr = (c / v)^2

        Args:
            product_id: SHARAD product ID
            trace_idx0: Center trace index
            bin_idx0: Center bin index
            window_traces: Traces to include (±window_traces/2 around center)
            window_bins: Bins to include (±window_bins/2 around center)

        Returns:
            HyperbolaFitResult with epsilon_r_est and quality metrics
        """
        try:
            # Late import to avoid circular deps
            from api.sharad_highres_router import _get_power, _get_geometry

            power, n_traces = _get_power(product_id)
            geom, _ = _get_geometry(product_id)

            # Bounds check
            if trace_idx0 < window_traces // 2 or trace_idx0 >= n_traces - window_traces // 2:
                return HyperbolaFitResult(
                    epsilon_r_est=None,
                    epsilon_r_std=None,
                    rmse_bins=float("nan"),
                    inlier_ratio=0.0,
                    n_inliers=0,
                    quality_note="Center trace too close to track edge",
                    diagnostics={},
                )

            # Extract local window
            trace_start = max(0, trace_idx0 - window_traces // 2)
            trace_end = min(n_traces, trace_idx0 + window_traces // 2)
            bin_start = max(0, bin_idx0 - window_bins // 2)
            bin_end = min(power.shape[1], bin_idx0 + window_bins // 2)

            window = power[trace_start:trace_end, bin_start:bin_end]

            if window.size == 0:
                return HyperbolaFitResult(
                    epsilon_r_est=None,
                    epsilon_r_std=None,
                    rmse_bins=float("nan"),
                    inlier_ratio=0.0,
                    n_inliers=0,
                    quality_note="Window extraction failed",
                    diagnostics={},
                )

            # Pick reflector points: find max per trace in window
            picks_x = []  # Trace offsets (relative)
            picks_t_bins = []  # Time picks (in bins)
            picks_confidence = []  # SNR or peak power

            for i in range(window.shape[0]):
                peak_idx = np.argmax(window[i, :])
                peak_power = window[i, peak_idx]

                # Skip very weak peaks
                if peak_power < 0.1 * np.max(window[i, :]):
                    continue

                picks_x.append(i - window_traces // 2)
                picks_t_bins.append(peak_idx)
                picks_confidence.append(peak_power)

            if len(picks_x) < 5:
                return HyperbolaFitResult(
                    epsilon_r_est=None,
                    epsilon_r_std=None,
                    rmse_bins=float("nan"),
                    inlier_ratio=0.0,
                    n_inliers=0,
                    quality_note=f"Insufficient picks ({len(picks_x)} < 5)",
                    diagnostics={},
                )

            picks_x = np.array(picks_x, dtype=np.float64)
            picks_t_bins = np.array(picks_t_bins, dtype=np.float64)
            picks_confidence = np.array(picks_confidence, dtype=np.float64)

            # Convert trace offset to distance (assumption: ~300m per trace)
            x_km = picks_x * 0.3 / 1000.0  # 300m/trace → km

            # Convert bin indices to time (microseconds)
            t_us = picks_t_bins * SHARAD_SAMPLE_INTERVAL_US

            # Initial parameter guess: v~40 km/s (reasonable for regolith)
            v0 = 40_000.0  # m/s
            t0_guess = np.median(t_us)

            def hyperbola_model(params, x_km, t_us):
                """NMO hyperbola: t(x) = sqrt(t0^2 + (x/v)^2), t in us, x in km."""
                v_ms, t0_us = params
                v_km_us = v_ms / 1e6  # Convert m/s to km/μs
                predicted_t = np.sqrt(t0_us ** 2 + (x_km / v_km_us) ** 2)
                return predicted_t - t_us

            # Fit hyperbola
            try:
                result_fit = least_squares(
                    hyperbola_model,
                    x0=[v0, t0_guess],
                    args=(x_km, t_us),
                    bounds=([10_000.0, 0.0], [100_000.0, 1000.0]),
                )
                v_fit_ms, t0_fit_us = result_fit.x
                residuals = result_fit.fun

            except Exception as e:
                logger.warning(f"Hyperbola fit failed: {e}")
                return HyperbolaFitResult(
                    epsilon_r_est=None,
                    epsilon_r_std=None,
                    rmse_bins=float("nan"),
                    inlier_ratio=0.0,
                    n_inliers=0,
                    quality_note=f"Fit error: {str(e)}",
                    diagnostics={},
                )

            # Outlier rejection: remove picks with large residuals
            std_residuals = np.std(residuals)
            inlier_mask = np.abs(residuals) < 2.0 * std_residuals
            n_inliers = inlier_mask.sum()
            inlier_ratio = n_inliers / len(picks_x)

            if n_inliers < 5:
                return HyperbolaFitResult(
                    epsilon_r_est=None,
                    epsilon_r_std=None,
                    rmse_bins=np.sqrt(np.mean(residuals ** 2)),
                    inlier_ratio=inlier_ratio,
                    n_inliers=n_inliers,
                    quality_note=f"Poor fit: {inlier_ratio:.1%} inliers",
                    diagnostics={"v_ms": v_fit_ms, "t0_us": t0_fit_us},
                )

            # Refit on inliers only
            x_inliers = x_km[inlier_mask]
            t_inliers = t_us[inlier_mask]

            try:
                result_inliers = least_squares(
                    hyperbola_model,
                    x0=[v_fit_ms, t0_fit_us],
                    args=(x_inliers, t_inliers),
                    bounds=([10_000.0, 0.0], [100_000.0, 1000.0]),
                )
                v_final_ms, t0_final_us = result_inliers.x
                residuals_final = result_inliers.fun

            except Exception:
                v_final_ms = v_fit_ms
                t0_final_us = t0_fit_us
                residuals_final = residuals[inlier_mask]

            rmse_bins = np.sqrt(np.mean(residuals_final ** 2))

            # Convert v → εr = (c / v)^2
            epsilon_r_est = (SPEED_OF_LIGHT / v_final_ms) ** 2

            # Rough std estimate from fit covariance
            if result_inliers.jac is not None:
                try:
                    cov_matrix = np.linalg.inv(result_inliers.jac.T @ result_inliers.jac)
                    v_std = np.sqrt(cov_matrix[0, 0])
                    # εr ≈ (c/v)^2, so dεr/dv = -2c^2/v^3
                    depsilon_dv = -2 * SPEED_OF_LIGHT ** 2 / (v_final_ms ** 3)
                    epsilon_std = abs(depsilon_dv * v_std)
                except Exception:
                    epsilon_std = None
            else:
                epsilon_std = None

            quality_note = f"Fit OK: εr={epsilon_r_est:.2f}, {inlier_ratio:.1%} inliers, rmse={rmse_bins:.2f} bins"

            return HyperbolaFitResult(
                epsilon_r_est=epsilon_r_est,
                epsilon_r_std=epsilon_std,
                rmse_bins=rmse_bins,
                inlier_ratio=inlier_ratio,
                n_inliers=n_inliers,
                quality_note=quality_note,
                diagnostics={
                    "v_ms": round(v_final_ms, 0),
                    "t0_us": round(t0_final_us, 3),
                    "n_picks": len(picks_x),
                    "std_residuals": round(std_residuals, 3),
                },
            )

        except Exception as e:
            logger.exception("Hyperbola fitting exception")
            return HyperbolaFitResult(
                epsilon_r_est=None,
                epsilon_r_std=None,
                rmse_bins=float("nan"),
                inlier_ratio=0.0,
                n_inliers=0,
                quality_note=f"Exception: {str(e)}",
                diagnostics={},
            )
