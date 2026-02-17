"""Generic colormap interpolation and overlay segment builders.

Used by analysis modules to generate colored polyline overlays for
Cesium map rendering. Each module defines its own colormap stops;
this module provides the interpolation and segment-building logic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Predefined colormaps ─────────────────────────────────────────

# Regolith thickness: thin (blue) → thick (red)
CMAP_THICKNESS: List[Tuple[float, List[int]]] = [
    (0,   [70, 130, 230, 220]),    # blue  (thin, 0 m)
    (20,  [70, 200, 220, 220]),    # cyan  (20 m)
    (50,  [240, 220, 80, 220]),    # yellow (50 m)
    (80,  [230, 130, 50, 220]),    # orange (80 m)
    (150, [210, 50, 50, 220]),     # red   (≥150 m)
]

NO_DATA_COLOR = [120, 120, 120, 120]  # gray for no detection


# ── Colormap interpolation ───────────────────────────────────────

def interpolate_colormap(
    value: Optional[float],
    stops: List[Tuple[float, List[int]]],
    no_data_color: List[int] = NO_DATA_COLOR,
) -> List[int]:
    """Map a scalar value to RGBA (0-255) using piecewise-linear colormap stops.

    Args:
        value: The measurement value (or None for no-data)
        stops: List of (threshold, [r, g, b, a]) sorted by threshold ascending
        no_data_color: Color for None values

    Returns:
        [r, g, b, a] each in 0-255
    """
    if value is None:
        return list(no_data_color)

    t = max(0.0, value)
    for i in range(len(stops) - 1):
        lo_t, lo_c = stops[i]
        hi_t, hi_c = stops[i + 1]
        if t <= hi_t:
            frac = (t - lo_t) / max(hi_t - lo_t, 1e-6)
            return [int(lo_c[j] + frac * (hi_c[j] - lo_c[j])) for j in range(4)]
    # Clamp to max color
    return list(stops[-1][1])


# ── Overlay segment builder ──────────────────────────────────────

def build_overlay_segments(
    lats: np.ndarray,
    lons: np.ndarray,
    detected: np.ndarray,
    values: np.ndarray,
    colormap_stops: List[Tuple[float, List[int]]],
    max_segments: int = 500,
    no_data_color: List[int] = NO_DATA_COLOR,
) -> List[Dict]:
    """Build color-coded track overlay segments, downsampled to max_segments.

    Args:
        lats: Latitude array (planetographic, for display)
        lons: Longitude array (-180..180)
        detected: Boolean mask — True where measurement is valid
        values: Measurement values (e.g., thickness_m); only used where detected=True
        colormap_stops: Piecewise-linear colormap stops
        max_segments: Maximum number of output segments
        no_data_color: RGBA for segments with no valid detections

    Returns:
        List of dicts with start_lat, start_lon, end_lat, end_lon,
        value (float|None), color ([r,g,b,a]).
    """
    n = len(lats)
    if n < 2:
        return []

    step = max(1, n // max_segments)
    segments: List[Dict] = []

    for i in range(0, n - step, step):
        j = min(i + step, n - 1)

        # Average value over segment
        seg_det = detected[i:j + 1]
        if seg_det.any():
            seg_val = float(np.mean(values[i:j + 1][seg_det]))
        else:
            seg_val = None

        segments.append({
            "start_lat": round(float(lats[i]), 5),
            "start_lon": round(float(lons[i]), 5),
            "end_lat": round(float(lats[j]), 5),
            "end_lon": round(float(lons[j]), 5),
            "value": round(seg_val, 1) if seg_val is not None else None,
            "color": interpolate_colormap(seg_val, colormap_stops, no_data_color),
        })

    return segments
