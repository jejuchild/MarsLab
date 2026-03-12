"""
COSI-Corr-style phase correlation for sub-pixel displacement measurement.

Based on Leprince et al. (2007) IEEE TGRS 45(6):1529-1558 with band-pass
frequency masking for illumination robustness. Sub-pixel refinement via
Guizar-Sicairos et al. (2008) upsampled DFT method.

At HiRISE 25 cm/px with upsample_factor=100: precision ~2.5 mm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a single chip-pair phase correlation."""

    row_shift: float  # pixels (positive = downward)
    col_shift: float  # pixels (positive = rightward)
    snr: float  # peak-to-background ratio
    error: float  # RMS correlation error (0=perfect)
    valid: bool  # passes SNR + error thresholds


@dataclass
class DisplacementField:
    """Dense displacement field from sliding-window correlation."""

    row_centers: np.ndarray  # pixel coordinates of window centers
    col_centers: np.ndarray
    row_disp: np.ndarray  # displacement in pixels (2D array)
    col_disp: np.ndarray
    magnitude: np.ndarray  # sqrt(row² + col²)
    snr: np.ndarray
    error: np.ndarray
    valid_mask: np.ndarray  # boolean
    pixel_scale_m: float  # meters per pixel (e.g. 0.25 for HiRISE)

    @property
    def row_disp_m(self) -> np.ndarray:
        return self.row_disp * self.pixel_scale_m

    @property
    def col_disp_m(self) -> np.ndarray:
        return self.col_disp * self.pixel_scale_m

    @property
    def magnitude_m(self) -> np.ndarray:
        return self.magnitude * self.pixel_scale_m

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid_mask))


def _upsampled_dft(
    data: np.ndarray,
    region_size: int,
    upsample_factor: int,
    offsets: np.ndarray,
) -> np.ndarray:
    """
    Compute upsampled DFT in a small region via matrix multiplication.

    Guizar-Sicairos et al. (2008) Optics Letters 33:156-158.
    O(N*M) where M = region_size, vs O(N*upsample_factor^2) for zero-pad.
    """
    rows, cols = data.shape
    row_kern = np.exp(
        -1j * 2 * np.pi
        * (np.arange(region_size)[:, None] - offsets[0])
        * fftfreq(rows)[None, :]
        / upsample_factor
    )
    col_kern = np.exp(
        -1j * 2 * np.pi
        * fftfreq(cols)[:, None]
        * (np.arange(region_size)[None, :] - offsets[1])
        / upsample_factor
    )
    return row_kern @ data @ col_kern


def cosicorr_phase_correlation(
    chip1: np.ndarray,
    chip2: np.ndarray,
    upsample_factor: int = 100,
    freq_low: float = 0.02,
    freq_high: float = 0.80,
    snr_threshold: float = 3.0,
    use_window: bool = False,
) -> CorrelationResult:
    """
    COSI-Corr-style phase correlation with band-pass frequency masking.

    Parameters
    ----------
    chip1, chip2 : 2D float arrays, same shape (power-of-2 recommended)
        Reference and secondary image chips.
    upsample_factor : int
        Sub-pixel precision = 1/upsample_factor pixels.
        100 -> 0.01 px = 2.5 mm at HiRISE 25cm/px.
    freq_low : float
        Low-frequency cutoff (fraction of max freq).
    freq_high : float
        High-frequency cutoff.
    snr_threshold : float
        Minimum SNR for valid measurement.
    use_window : bool
        Apply Tukey window (reduces edge artifacts, slight accuracy cost).

    Returns
    -------
    CorrelationResult with displacement (row, col), SNR, error, validity.
    Sign convention: positive row_shift = content moved downward in img2.
    """
    if chip1.shape != chip2.shape:
        raise ValueError(f"Chip shapes must match: {chip1.shape} vs {chip2.shape}")
    if chip1.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got {chip1.ndim}D")

    rows, cols = chip1.shape
    c1 = chip1.astype(np.float64)
    c2 = chip2.astype(np.float64)

    if use_window:
        from scipy.signal import windows
        wy = windows.tukey(rows, alpha=0.2)
        wx = windows.tukey(cols, alpha=0.2)
        window = np.outer(wy, wx)
        c1 = c1 * window
        c2 = c2 * window

    F1 = fft2(c1)
    F2 = fft2(c2)
    cross = F1 * np.conj(F2)
    eps = np.finfo(np.float64).eps

    if freq_low > 0 or freq_high < 0.5:
        u = fftfreq(rows)
        v = fftfreq(cols)
        U, V = np.meshgrid(u, v, indexing="ij")
        freq_r = np.sqrt(U**2 + V**2)
        band_mask = (freq_r >= freq_low) & (freq_r <= freq_high)
        cross *= band_mask

    cross /= np.maximum(np.abs(cross), 100 * eps)

    cc = ifft2(cross)
    cc_abs = np.abs(cc)
    peak_idx = np.unravel_index(np.argmax(cc_abs), cc_abs.shape)
    peak_val = float(cc_abs[peak_idx])

    bg = cc_abs.copy()
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            bg[(peak_idx[0] + dr) % rows, (peak_idx[1] + dc) % cols] = 0
    bg_mean = float(bg[bg > 0].mean()) if np.any(bg > 0) else eps
    snr = peak_val / (bg_mean + eps)

    shape = np.array([rows, cols], dtype=np.float64)
    midpoint = shape / 2.0
    shift = np.array(peak_idx, dtype=np.float64)
    shift[shift > midpoint] -= shape[shift > midpoint]

    if upsample_factor > 1:
        region_size = int(np.ceil(upsample_factor * 1.5))
        dftshift = np.trunc(region_size / 2.0)
        offsets = np.array([
            dftshift - shift[0] * upsample_factor,
            dftshift - shift[1] * upsample_factor,
        ])

        cc_up = np.conj(
            _upsampled_dft(np.conj(cross), region_size, upsample_factor, offsets)
        )
        up_abs = np.abs(cc_up)
        up_peak = np.unravel_index(np.argmax(up_abs), up_abs.shape)
        shift[0] += (up_peak[0] - dftshift) / upsample_factor
        shift[1] += (up_peak[1] - dftshift) / upsample_factor

    # Negate: cross-corr gives alignment shift, we want displacement
    displacement = -shift
    error_val = max(0.0, 1.0 - peak_val)

    valid = snr >= snr_threshold and peak_val >= 0.05

    return CorrelationResult(
        row_shift=float(displacement[0]),
        col_shift=float(displacement[1]),
        snr=float(snr),
        error=float(error_val),
        valid=valid,
    )


def sliding_window_correlation(
    img1: np.ndarray,
    img2: np.ndarray,
    chip_size: int = 64,
    step_size: int = 16,
    upsample_factor: int = 100,
    snr_threshold: float = 3.0,
    pixel_scale_m: float = 0.25,
    stable_mask: Optional[np.ndarray] = None,
    remove_bulk_offset: bool = True,
    detrend_order: int = 0,
) -> DisplacementField:
    """
    Dense displacement field via sliding-window phase correlation.

    Parameters
    ----------
    img1, img2 : 2D arrays (float)
        Co-registered reference and secondary images.
    chip_size : int
        Window size in pixels. 64 px @ 25cm/px = 16m.
        Must be >= 4× expected max displacement.
    step_size : int
        Stride between windows. Overlap = (chip_size - step_size) / chip_size.
        16 on 64 → 75% overlap.
    upsample_factor : int
        Sub-pixel precision. 100 → 0.01 px = 2.5 mm @ HiRISE.
    snr_threshold : float
        Reject windows with SNR below this.
    pixel_scale_m : float
        Ground sampling distance in meters. 0.25 for HiRISE RED.
    stable_mask : optional 2D bool array, same shape as img1
        True = stable terrain (use for offset removal / detrending).
        If None, all valid measurements used for bulk offset.
    remove_bulk_offset : bool
        Whether to subtract median displacement from stable terrain
        to remove residual co-registration error. Ignored if detrend_order > 0.
    detrend_order : int
        Polynomial detrending order for spatially-varying distortion removal.
        0 = median subtraction only (legacy behavior).
        1 = linear (affine-like, 6 parameters).
        2 = quadratic (12 parameters) — recommended for HiRISE temporal pairs.
        3 = cubic (20 parameters) — use only with >60 stable points.
        Polynomial is fit to stable terrain displacements only, then
        subtracted from ALL measurements. This preserves real displacement
        at scarps while removing systematic co-registration distortion.

    Returns
    -------
    DisplacementField with displacement maps, SNR, validity mask.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")

    H, W = img1.shape
    half = chip_size // 2

    row_centers = np.arange(half, H - half, step_size)
    col_centers = np.arange(half, W - half, step_size)
    nr, nc = len(row_centers), len(col_centers)

    if nr == 0 or nc == 0:
        raise ValueError(
            f"Image too small ({H}×{W}) for chip_size={chip_size}. "
            f"Need at least {chip_size} pixels in each dimension."
        )

    row_disp = np.full((nr, nc), np.nan, dtype=np.float64)
    col_disp = np.full_like(row_disp, np.nan)
    snr_map = np.full_like(row_disp, np.nan)
    err_map = np.full_like(row_disp, np.nan)

    total = nr * nc
    logged_pct = -1

    for i, r in enumerate(row_centers):
        for j, c in enumerate(col_centers):
            r0, r1 = r - half, r + half
            c0, c1 = c - half, c + half

            chip1 = img1[r0:r1, c0:c1].astype(np.float64)
            chip2 = img2[r0:r1, c0:c1].astype(np.float64)

            # Skip invalid chips
            if not np.all(np.isfinite(chip1)) or not np.all(np.isfinite(chip2)):
                continue
            if chip1.std() < 1e-6 or chip2.std() < 1e-6:
                continue

            result = cosicorr_phase_correlation(
                chip1,
                chip2,
                upsample_factor=upsample_factor,
                snr_threshold=snr_threshold,
            )

            if result.valid:
                row_disp[i, j] = result.row_shift
                col_disp[i, j] = result.col_shift
                snr_map[i, j] = result.snr
                err_map[i, j] = result.error

        # Progress logging
        pct = int(100 * (i + 1) / nr)
        if pct >= logged_pct + 10:
            logged_pct = pct
            logger.info(f"Phase correlation: {pct}% ({i+1}/{nr} rows)")

    valid_mask = np.isfinite(row_disp)

    # Remove systematic co-registration error using stable terrain
    if (remove_bulk_offset or detrend_order > 0) and np.any(valid_mask):
        if stable_mask is not None:
            # Build stable terrain indicator at window centers
            stable_at_centers = np.zeros((nr, nc), dtype=bool)
            for i, r in enumerate(row_centers):
                for j, c in enumerate(col_centers):
                    r0, r1 = r - half, r + half
                    c0, c1 = c - half, c + half
                    patch = stable_mask[r0:r1, c0:c1]
                    stable_at_centers[i, j] = patch.mean() > 0.8
            use_for_offset = valid_mask & stable_at_centers
        else:
            use_for_offset = valid_mask

        n_stable = int(np.sum(use_for_offset))

        if detrend_order > 0 and n_stable >= _min_points_for_order(detrend_order):
            # Polynomial detrending: fit 2D polynomial to stable terrain
            # displacements, then subtract from all measurements
            row_disp, col_disp = _polynomial_detrend(
                row_disp, col_disp, row_centers, col_centers,
                use_for_offset, valid_mask, detrend_order, pixel_scale_m,
            )
        elif n_stable >= 10:
            # Fallback: median subtraction (legacy behavior)
            bulk_row = np.nanmedian(row_disp[use_for_offset])
            bulk_col = np.nanmedian(col_disp[use_for_offset])
            row_disp[valid_mask] -= bulk_row
            col_disp[valid_mask] -= bulk_col
            logger.info(
                f"Removed bulk offset: Δrow={bulk_row:.3f} px, Δcol={bulk_col:.3f} px "
                f"({bulk_row * pixel_scale_m:.4f} m, {bulk_col * pixel_scale_m:.4f} m) "
                f"from {n_stable} stable points"
            )
        else:
            logger.warning(
                f"Only {n_stable} stable points — "
                "skipping offset removal / detrending"
            )

    magnitude = np.sqrt(row_disp**2 + col_disp**2)

    return DisplacementField(
        row_centers=row_centers,
        col_centers=col_centers,
        row_disp=row_disp,
        col_disp=col_disp,
        magnitude=magnitude,
        snr=snr_map,
        error=err_map,
        valid_mask=valid_mask,
        pixel_scale_m=pixel_scale_m,
    )


# ---------------------------------------------------------------------------
# Polynomial detrending helpers
# ---------------------------------------------------------------------------


def _min_points_for_order(order: int) -> int:
    """Minimum stable terrain points required for polynomial fit."""
    n_coeffs = (order + 1) * (order + 2) // 2  # 2D polynomial terms
    return max(n_coeffs * 3, 10)  # At least 3× overdetermined


def _build_polynomial_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """
    Build Vandermonde-style design matrix for 2D polynomial fit.

    For order=2: columns are [1, x, y, x², xy, y²]
    For order=3: columns are [1, x, y, x², xy, y², x³, x²y, xy², y³]
    """
    terms = []
    for total_deg in range(order + 1):
        for iy in range(total_deg + 1):
            ix = total_deg - iy
            terms.append(x**ix * y**iy)
    return np.column_stack(terms)


def _polynomial_detrend(
    row_disp: np.ndarray,
    col_disp: np.ndarray,
    row_centers: np.ndarray,
    col_centers: np.ndarray,
    stable_mask: np.ndarray,
    valid_mask: np.ndarray,
    order: int,
    pixel_scale_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit and subtract a 2D polynomial from displacement fields.

    The polynomial is fit using only stable terrain measurements,
    then evaluated and subtracted from ALL valid measurements.
    This removes spatially-varying co-registration distortion
    while preserving real displacement at scarps.

    Parameters
    ----------
    row_disp, col_disp : 2D arrays
        Displacement fields (nan where invalid).
    row_centers, col_centers : 1D arrays
        Pixel coordinates of window centers.
    stable_mask : 2D bool array
        True at stable terrain window positions.
    valid_mask : 2D bool array
        True at positions with valid measurements.
    order : int
        Polynomial order (1=linear, 2=quadratic, 3=cubic).
    pixel_scale_m : float
        Meters per pixel.

    Returns
    -------
    Detrended (row_disp, col_disp) arrays.
    """
    nr, nc = row_disp.shape

    # Build coordinate grids at chip centers
    cc, rr = np.meshgrid(col_centers, row_centers)  # pixel coords

    # Normalize coordinates to [-1, 1] for numerical stability
    r_mean, r_std = rr.mean(), max(rr.std(), 1.0)
    c_mean, c_std = cc.mean(), max(cc.std(), 1.0)
    rr_norm = (rr - r_mean) / r_std
    cc_norm = (cc - c_mean) / c_std

    # Extract stable terrain data
    s_x = cc_norm[stable_mask].ravel()
    s_y = rr_norm[stable_mask].ravel()
    s_row_disp = row_disp[stable_mask].ravel()
    s_col_disp = col_disp[stable_mask].ravel()

    n_stable = len(s_x)
    n_coeffs = (order + 1) * (order + 2) // 2

    logger.info(
        f"Polynomial detrend (order={order}): {n_stable} stable points, "
        f"{n_coeffs} coefficients"
    )

    # Build design matrix for stable terrain
    A_stable = _build_polynomial_matrix(s_x, s_y, order)

    # Robust fit using iteratively reweighted least squares (IRLS)
    # to handle outliers among stable terrain chips
    row_coeffs = _robust_polyfit(A_stable, s_row_disp)
    col_coeffs = _robust_polyfit(A_stable, s_col_disp)

    # Evaluate polynomial model at ALL valid positions
    all_x = cc_norm[valid_mask].ravel()
    all_y = rr_norm[valid_mask].ravel()
    A_all = _build_polynomial_matrix(all_x, all_y, order)

    row_model = A_all @ row_coeffs
    col_model = A_all @ col_coeffs

    # Compute residuals on stable terrain for quality assessment
    stable_row_model = A_stable @ row_coeffs
    stable_col_model = A_stable @ col_coeffs
    stable_row_residual = s_row_disp - stable_row_model
    stable_col_residual = s_col_disp - stable_col_model
    stable_mag_residual = np.sqrt(stable_row_residual**2 + stable_col_residual**2)

    before_row = np.sqrt(np.mean(s_row_disp**2))
    before_col = np.sqrt(np.mean(s_col_disp**2))
    after_row = np.sqrt(np.mean(stable_row_residual**2))
    after_col = np.sqrt(np.mean(stable_col_residual**2))

    logger.info(
        f"  Stable terrain RMS (before): row={before_row * pixel_scale_m:.4f}m, "
        f"col={before_col * pixel_scale_m:.4f}m"
    )
    logger.info(
        f"  Stable terrain RMS (after):  row={after_row * pixel_scale_m:.4f}m, "
        f"col={after_col * pixel_scale_m:.4f}m"
    )
    logger.info(
        f"  Improvement: row={before_row / max(after_row, 1e-10):.1f}×, "
        f"col={before_col / max(after_col, 1e-10):.1f}×"
    )
    logger.info(
        f"  Stable terrain mag (after): "
        f"mean={np.mean(stable_mag_residual) * pixel_scale_m:.4f}m, "
        f"median={np.median(stable_mag_residual) * pixel_scale_m:.4f}m, "
        f"std={np.std(stable_mag_residual) * pixel_scale_m:.4f}m"
    )

    # Subtract polynomial model from all valid measurements
    row_out = row_disp.copy()
    col_out = col_disp.copy()
    row_out[valid_mask] -= row_model
    col_out[valid_mask] -= col_model

    return row_out, col_out


def _robust_polyfit(
    A: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 3,
    clip_sigma: float = 2.5,
) -> np.ndarray:
    """
    Iteratively reweighted least squares for robust polynomial fitting.

    Downweights outliers (>clip_sigma standard deviations from fit)
    in each iteration.
    """
    weights = np.ones(len(y))

    for iteration in range(n_iterations):
        W = np.diag(weights)
        try:
            # Weighted least squares: (A^T W A)^{-1} A^T W y
            AW = A.T @ W
            coeffs = np.linalg.solve(AW @ A, AW @ y)
        except np.linalg.LinAlgError:
            # Fall back to unweighted least squares
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            break

        # Compute residuals and update weights
        residuals = y - A @ coeffs
        sigma = np.std(residuals[weights > 0.5])
        if sigma < 1e-10:
            break

        # Tukey bisquare weights
        u = residuals / (clip_sigma * sigma)
        weights = np.where(np.abs(u) < 1, (1 - u**2)**2, 0.0)

        n_inliers = int(np.sum(weights > 0.5))
        if iteration < n_iterations - 1:
            logger.debug(
                f"  IRLS iter {iteration}: sigma={sigma:.4f}, "
                f"inliers={n_inliers}/{len(y)}"
            )

    return coeffs
