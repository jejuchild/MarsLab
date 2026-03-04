"""
Mars Ice Accessibility Algorithm.

Computes a composite accessibility score (0-1) from four sub-scores:
1. Ice Presence — SWIM consistency + landform classification
2. Ice Depth — SWIM depth products + thermal inertia proxy
3. Excavation Feasibility — thermal inertia + dust cover + slope
4. Landing & Traversability — elevation + slope + roughness

Higher score = more accessible ice for future missions.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

# Default weights for the composite score
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ice_presence": 0.35,
    "ice_depth": 0.25,
    "excavation": 0.20,
    "landing": 0.20,
}

# Landform bonus for ice presence
LANDFORM_BONUS: Dict[str, float] = {
    "LDA": 1.0,
    "LVF": 0.8,
    "CCF": 0.6,
    "OTHER": 0.0,
}


@dataclass
class AccessibilityResult:
    """Result of accessibility computation at a single point."""

    lat: float
    lon: float
    score: float  # Composite 0-1

    # Sub-scores (0-1 each)
    ice_presence: float
    ice_depth: float
    excavation: float
    landing: float

    # Weights used
    weights: Dict[str, float]

    # Raw input values
    inputs: Dict[str, object] = field(default_factory=dict)

    # Data quality
    layers_available: int = 0
    layers_total: int = 8
    confidence: Literal["high", "medium", "low", "insufficient"] = "insufficient"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _safe(v: Optional[float]) -> Optional[float]:
    """Return None for NaN / inf values."""
    if v is None:
        return None
    if not np.isfinite(v):
        return None
    return v


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in w.items()}


# ---------------------------------------------------------------------------
# Sub-score functions (scalar)
# ---------------------------------------------------------------------------

def compute_ice_presence(
    swim_consistency: Optional[float],
    landform: Optional[str] = None,
) -> Optional[float]:
    """
    Ice Presence Score (0-1).
    SWIM consistency (-1 to +1) normalised to 0-1, combined with landform bonus.
    """
    parts: list[float] = []
    weights: list[float] = []

    if swim_consistency is not None:
        parts.append(_clamp((swim_consistency + 1.0) / 2.0))
        weights.append(0.7)

    if landform is not None and landform in LANDFORM_BONUS:
        parts.append(LANDFORM_BONUS[landform])
        weights.append(0.3)

    if not parts:
        return None

    total_w = sum(weights)
    return _clamp(sum(p * w for p, w in zip(parts, weights)) / total_w)


def compute_ice_depth(
    swim_0_1m: Optional[float],
    swim_1_5m: Optional[float],
    swim_5m_plus: Optional[float],
    thermal_inertia: Optional[float],
) -> Optional[float]:
    """
    Ice Depth Score (0-1). Shallower ice = higher score.
    Low TI → fine regolith → shallower ice table.
    """
    parts: list[float] = []
    weights: list[float] = []

    swim_vals: list[tuple[float, float]] = []
    if swim_0_1m is not None:
        swim_vals.append((_clamp((swim_0_1m + 1) / 2), 1.0))
    if swim_1_5m is not None:
        swim_vals.append((_clamp((swim_1_5m + 1) / 2), 0.6))
    if swim_5m_plus is not None:
        swim_vals.append((_clamp((swim_5m_plus + 1) / 2), 0.3))

    if swim_vals:
        swim_depth = sum(v * w for v, w in swim_vals) / sum(w for _, w in swim_vals)
        parts.append(swim_depth)
        weights.append(0.6)

    # TI < 150 TIU → fine regolith → shallow ice → 1.0
    # TI 150-300 → mixed → linear decay
    # TI > 300 → consolidated → deep/no ice → 0.0
    if thermal_inertia is not None and thermal_inertia > 0:
        parts.append(_clamp(1.0 - (thermal_inertia - 150.0) / 150.0))
        weights.append(0.4)

    if not parts:
        return None

    return _clamp(sum(p * w for p, w in zip(parts, weights)) / sum(weights))


def compute_excavation(
    thermal_inertia: Optional[float],
    dci: Optional[float],
    slope: Optional[float],
) -> Optional[float]:
    """
    Excavation Feasibility Score (0-1). Easier digging = higher score.
    """
    parts: list[float] = []
    weights: list[float] = []

    # TI < 100 → very soft → 1.0;  TI > 500 → very hard → 0.0
    if thermal_inertia is not None and thermal_inertia > 0:
        parts.append(_clamp(1.0 - (thermal_inertia - 100.0) / 400.0))
        weights.append(0.5)

    # DCI ~0.94 (dusty) → easy surface → 1.0;  DCI ~1.0 (rocky) → 0.0
    if dci is not None and 0.9 <= dci <= 1.0:
        parts.append(_clamp((1.0 - dci) / 0.06))
        weights.append(0.25)

    # Slope < 5° → 1.0;  > 15° → 0.0
    if slope is not None and slope >= 0:
        parts.append(_clamp(1.0 - slope / 15.0))
        weights.append(0.25)

    if not parts:
        return None

    return _clamp(sum(p * w for p, w in zip(parts, weights)) / sum(weights))


def compute_landing(
    elevation: Optional[float],
    slope: Optional[float],
    tri: Optional[float],
) -> Optional[float]:
    """
    Landing & Traversability Score (0-1). Safer = higher score.
    """
    parts: list[float] = []
    weights: list[float] = []

    # Elevation < 0 m → ideal; 0-2000 m → linear decay; > 2000 m → 0
    if elevation is not None:
        elev_score = 1.0 if elevation < 0 else _clamp(1.0 - elevation / 2000.0)
        parts.append(elev_score)
        weights.append(0.4)

    # Slope < 5° → 1.0;  > 15° → 0.0
    if slope is not None and slope >= 0:
        parts.append(_clamp(1.0 - slope / 15.0))
        weights.append(0.35)

    # TRI < 50m → smooth plains → 1.0;  50-500m → linear decay;  > 500m → rough → 0.0
    # (at 3km resolution, TRI is in metres; mid-lat median ~48m, p95 ~470m)
    if tri is not None and tri >= 0:
        parts.append(_clamp(1.0 - (tri - 50.0) / 450.0))
        weights.append(0.25)

    if not parts:
        return None

    return _clamp(sum(p * w for p, w in zip(parts, weights)) / sum(weights))


# ---------------------------------------------------------------------------
# Point-level composite
# ---------------------------------------------------------------------------

def compute_accessibility(
    swim_consistency: Optional[float] = None,
    swim_0_1m: Optional[float] = None,
    swim_1_5m: Optional[float] = None,
    swim_5m_plus: Optional[float] = None,
    thermal_inertia: Optional[float] = None,
    dci: Optional[float] = None,
    elevation: Optional[float] = None,
    slope: Optional[float] = None,
    tri: Optional[float] = None,
    landform: Optional[str] = None,
    lat: float = 0.0,
    lon: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> AccessibilityResult:
    """Compute full accessibility score at a single point."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    w = _normalize_weights(w)

    # Sanitise
    swim_consistency = _safe(swim_consistency)
    swim_0_1m = _safe(swim_0_1m)
    swim_1_5m = _safe(swim_1_5m)
    swim_5m_plus = _safe(swim_5m_plus)
    thermal_inertia = _safe(thermal_inertia)
    dci = _safe(dci)
    elevation = _safe(elevation)
    slope = _safe(slope)
    tri = _safe(tri)

    # Sub-scores
    ip = compute_ice_presence(swim_consistency, landform)
    id_ = compute_ice_depth(swim_0_1m, swim_1_5m, swim_5m_plus, thermal_inertia)
    ex = compute_excavation(thermal_inertia, dci, slope)
    la = compute_landing(elevation, slope, tri)

    layers_available = sum(
        1 for v in [
            swim_consistency, thermal_inertia, dci,
            elevation, slope, tri, swim_0_1m, swim_1_5m,
        ] if v is not None
    )

    sub = {"ice_presence": ip, "ice_depth": id_, "excavation": ex, "landing": la}
    avail = {k: v for k, v in sub.items() if v is not None}

    if not avail:
        composite = 0.0
        confidence: Literal["high", "medium", "low", "insufficient"] = "insufficient"
    else:
        avail_w = {k: w[k] for k in avail}
        tw = sum(avail_w.values())
        composite = sum(avail[k] * avail_w[k] for k in avail) / tw if tw > 0 else 0.0
        n = len(avail)
        confidence = "high" if n == 4 else "medium" if n >= 3 else "low" if n >= 1 else "insufficient"

    inputs: Dict[str, object] = {
        "swim_consistency": swim_consistency,
        "swim_0_1m": swim_0_1m,
        "swim_1_5m": swim_1_5m,
        "swim_5m_plus": swim_5m_plus,
        "thermal_inertia": thermal_inertia,
        "dci": dci,
        "elevation": elevation,
        "slope": slope,
        "tri": tri,
        "landform": landform,
    }
    inputs = {k: round(v, 4) if isinstance(v, float) else v for k, v in inputs.items()}

    return AccessibilityResult(
        lat=lat,
        lon=lon,
        score=round(composite, 4),
        ice_presence=round(ip, 4) if ip is not None else 0.0,
        ice_depth=round(id_, 4) if id_ is not None else 0.0,
        excavation=round(ex, 4) if ex is not None else 0.0,
        landing=round(la, 4) if la is not None else 0.0,
        weights=w,
        inputs=inputs,
        layers_available=layers_available,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Grid-level composite (vectorised for tile rendering)
# ---------------------------------------------------------------------------

def _weighted_mean_parts(
    parts: list[tuple[np.ndarray, float]],
    shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Weighted mean of (array, weight) pairs, handling NaN."""
    if not parts:
        return None
    h, w = shape
    numer = np.zeros((h, w), dtype=np.float32)
    denom = np.zeros((h, w), dtype=np.float32)
    for arr, wt in parts:
        valid = np.isfinite(arr)
        numer[valid] += arr[valid] * wt
        denom[valid] += wt
    result = np.full((h, w), np.nan, dtype=np.float32)
    ok = denom > 0
    result[ok] = np.clip(numer[ok] / denom[ok], 0.0, 1.0)
    return result


def compute_accessibility_grid(
    swim_consistency: Optional[np.ndarray],
    swim_0_1m: Optional[np.ndarray],
    swim_1_5m: Optional[np.ndarray],
    swim_5m_plus: Optional[np.ndarray],
    thermal_inertia: Optional[np.ndarray],
    dci: Optional[np.ndarray],
    elevation: Optional[np.ndarray],
    slope: Optional[np.ndarray],
    tri: Optional[np.ndarray],
    shape: tuple[int, int],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Vectorised accessibility for a 2D grid. Returns float32 (0-1), NaN = no data."""
    h, w = shape
    wd = dict(DEFAULT_WEIGHTS)
    if weights:
        wd.update(weights)
    wd = _normalize_weights(wd)

    # --- Ice Presence ---
    ip_parts: list[tuple[np.ndarray, float]] = []
    if swim_consistency is not None:
        valid = np.isfinite(swim_consistency)
        arr = np.full((h, w), np.nan, dtype=np.float32)
        arr[valid] = np.clip((swim_consistency[valid] + 1.0) / 2.0, 0, 1)
        ip_parts.append((arr, 0.7))
    ice_pres = _weighted_mean_parts(ip_parts, shape)

    # --- Ice Depth ---
    id_parts: list[tuple[np.ndarray, float]] = []
    for arr_in, arr_w in [(swim_0_1m, 1.0), (swim_1_5m, 0.6), (swim_5m_plus, 0.3)]:
        if arr_in is not None:
            valid = np.isfinite(arr_in)
            a = np.full((h, w), np.nan, dtype=np.float32)
            a[valid] = np.clip((arr_in[valid] + 1.0) / 2.0, 0, 1)
            id_parts.append((a, arr_w))
    if thermal_inertia is not None:
        valid = np.isfinite(thermal_inertia) & (thermal_inertia > 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - (thermal_inertia[valid] - 150.0) / 150.0, 0, 1)
        id_parts.append((a, 0.4))
    ice_depth = _weighted_mean_parts(id_parts, shape)

    # --- Excavation ---
    ex_parts: list[tuple[np.ndarray, float]] = []
    if thermal_inertia is not None:
        valid = np.isfinite(thermal_inertia) & (thermal_inertia > 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - (thermal_inertia[valid] - 100.0) / 400.0, 0, 1)
        ex_parts.append((a, 0.5))
    if dci is not None:
        valid = np.isfinite(dci) & (dci >= 0.9) & (dci <= 1.0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip((1.0 - dci[valid]) / 0.06, 0, 1)
        ex_parts.append((a, 0.25))
    if slope is not None:
        valid = np.isfinite(slope) & (slope >= 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - slope[valid] / 15.0, 0, 1)
        ex_parts.append((a, 0.25))
    excavation = _weighted_mean_parts(ex_parts, shape)

    # --- Landing ---
    la_parts: list[tuple[np.ndarray, float]] = []
    if elevation is not None:
        valid = np.isfinite(elevation)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.where(
            elevation[valid] < 0, 1.0,
            np.clip(1.0 - elevation[valid] / 2000.0, 0, 1),
        )
        la_parts.append((a, 0.4))
    if slope is not None:
        valid = np.isfinite(slope) & (slope >= 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - slope[valid] / 15.0, 0, 1)
        la_parts.append((a, 0.35))
    if tri is not None:
        valid = np.isfinite(tri) & (tri >= 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - (tri[valid] - 50.0) / 450.0, 0, 1)
        la_parts.append((a, 0.25))
    landing = _weighted_mean_parts(la_parts, shape)

    # --- Composite ---
    result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for name, grid in [
        ("ice_presence", ice_pres),
        ("ice_depth", ice_depth),
        ("excavation", excavation),
        ("landing", landing),
    ]:
        if grid is not None:
            valid = np.isfinite(grid)
            result[valid] += grid[valid] * wd[name]
            weight_sum[valid] += wd[name]

    ok = weight_sum > 0
    result[ok] /= weight_sum[ok]
    result[~ok] = np.nan

    return np.clip(result, 0, 1)
