"""
Mars ISRU Accessibility Algorithm.

Computes a composite accessibility score (0-1) from five sub-scores:
1. Ice-Related Landform — HiRISE classification (LDA/LVF/CCF/OTHER)
2. Water-Related Mineral — CRISM mineral tier scoring
3. Surface Ice Signal — CRISM H2O Ice detection
4. Excavation Feasibility — thermal inertia + dust cover + slope
5. Landing & Traversability — elevation + slope + roughness

Higher score = more accessible for future ISRU missions.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

# Default weights for the composite score (5 sub-scores)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ice_landform": 0.25,
    "water_mineral": 0.20,
    "surface_ice": 0.15,
    "excavation": 0.20,
    "landing": 0.20,
}

# Landform → ice indicator score (from PRD)
LANDFORM_SCORE: Dict[str, float] = {
    "LDA": 1.0,       # Lobate Debris Apron — strongest ice indicator
    "LVF": 0.8,       # Lineated Valley Fill
    "CCF": 0.6,       # Concentric Crater Fill
    "SCT": 0.9,       # Scalloped Terrain — thermokarst from ice sublimation, strong ice indicator
    "OTHER": 0.0,
    "Uncertain": 0.0,
}


@dataclass
class AccessibilityResult:
    """Result of ISRU accessibility computation at a single point."""

    lat: float
    lon: float
    score: float  # Composite 0-1

    # Sub-scores (0-1 each)
    ice_landform: float
    water_mineral: float
    surface_ice: float
    excavation: float
    landing: float

    # Weights used
    weights: Dict[str, float]

    # Raw input values
    inputs: Dict[str, object] = field(default_factory=dict)

    # CRISM metadata (optional)
    crism_obs_id: str = ""
    crism_minerals: Dict[str, float] = field(default_factory=dict)

    # Data quality
    layers_available: int = 0
    layers_total: int = 7  # TES TI, elevation, slope, TRI, landform, CRISM×2
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

def compute_ice_landform(
    landform: Optional[str] = None,
    landform_confidence: float = 1.0,
) -> Optional[float]:
    """
    Ice-Related Landform Score (0-1).
    Based on HiRISE classification result.
    """
    if landform is None:
        return None
    base = LANDFORM_SCORE.get(landform, 0.0)
    return _clamp(base * landform_confidence)


def compute_water_mineral(
    water_mineral_score: Optional[float] = None,
) -> Optional[float]:
    """
    Water-Related Mineral Score (0-1).
    Directly from CRISM bridge pre-computed score.
    """
    if water_mineral_score is None:
        return None
    return _clamp(water_mineral_score)


def compute_surface_ice(
    surface_ice_score: Optional[float] = None,
) -> Optional[float]:
    """
    Surface Ice Signal Score (0-1).
    Directly from CRISM bridge H2O Ice detection.
    """
    if surface_ice_score is None:
        return None
    return _clamp(surface_ice_score)


def compute_excavation(
    thermal_inertia: Optional[float],
) -> Optional[float]:
    """
    Excavation Feasibility Score (0-1). Easier digging = higher score.

    Based solely on TES Thermal Inertia (Putzig & Mellon 2007):
      TI ≤ 200   → dust/sand, easy excavation → 1.0
      200 < TI < 2000 → linear decay (duricrust ~889 → 0.62)
      TI ≥ 2000  → consolidated rock → 0.0
    """
    if thermal_inertia is None or thermal_inertia <= 0:
        return None
    return _clamp(1.0 - (thermal_inertia - 200.0) / 1800.0)


def compute_landing(
    elevation: Optional[float],
    slope: Optional[float],
) -> Optional[float]:
    """
    Landing & Traversability Score (0-1). Safer = higher score.
    Based on elevation and slope only.
    """
    parts: list[float] = []
    weights: list[float] = []

    # Elevation < 0 m → ideal; 0-2000 m → linear decay; > 2000 m → 0
    if elevation is not None:
        elev_score = 1.0 if elevation < 0 else _clamp(1.0 - elevation / 2000.0)
        parts.append(elev_score)
        weights.append(0.55)

    # Slope < 5° → 1.0;  > 15° → 0.0
    if slope is not None and slope >= 0:
        parts.append(_clamp(1.0 - slope / 15.0))
        weights.append(0.45)

    if not parts:
        return None

    return _clamp(sum(p * w for p, w in zip(parts, weights)) / sum(weights))


# ---------------------------------------------------------------------------
# Point-level composite
# ---------------------------------------------------------------------------

def compute_accessibility(
    thermal_inertia: Optional[float] = None,
    elevation: Optional[float] = None,
    slope: Optional[float] = None,
    tri: Optional[float] = None,
    lat: float = 0.0,
    lon: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    # New ISRU params
    landform: Optional[str] = None,
    landform_confidence: float = 1.0,
    water_mineral_score: Optional[float] = None,
    surface_ice_score: Optional[float] = None,
    crism_obs_id: str = "",
    crism_minerals: Optional[Dict[str, float]] = None,
) -> AccessibilityResult:
    """Compute full ISRU accessibility score at a single point."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    w = _normalize_weights(w)

    # Sanitise
    thermal_inertia = _safe(thermal_inertia)
    elevation = _safe(elevation)
    slope = _safe(slope)
    # tri removed from landing scoring

    # Sub-scores
    il = compute_ice_landform(landform, landform_confidence)
    wm = compute_water_mineral(water_mineral_score)
    si = compute_surface_ice(surface_ice_score)
    ex = compute_excavation(thermal_inertia)
    la = compute_landing(elevation, slope)

    layers_available = sum(
        1 for v in [thermal_inertia, elevation, slope]
        if v is not None
    )
    if landform is not None:
        layers_available += 1
    if water_mineral_score is not None:
        layers_available += 1
    if surface_ice_score is not None:
        layers_available += 1

    sub = {
        "ice_landform": il,
        "water_mineral": wm,
        "surface_ice": si,
        "excavation": ex,
        "landing": la,
    }
    avail = {k: v for k, v in sub.items() if v is not None}

    if not avail:
        composite = 0.0
        confidence: Literal["high", "medium", "low", "insufficient"] = "insufficient"
    else:
        avail_w = {k: w.get(k, 0) for k in avail}
        tw = sum(avail_w.values())
        composite = sum(avail[k] * avail_w[k] for k in avail) / tw if tw > 0 else 0.0
        n = len(avail)
        if n >= 4:
            confidence = "high"
        elif n >= 2:
            confidence = "medium"
        else:
            confidence = "low"

    inputs: Dict[str, object] = {
        "thermal_inertia": thermal_inertia,
        "elevation": elevation,
        "slope": slope,
        "tri": tri,
    }
    inputs = {k: round(v, 4) if isinstance(v, float) else v for k, v in inputs.items()}

    return AccessibilityResult(
        lat=lat,
        lon=lon,
        score=round(composite, 4),
        ice_landform=round(il, 4) if il is not None else 0.0,
        water_mineral=round(wm, 4) if wm is not None else 0.0,
        surface_ice=round(si, 4) if si is not None else 0.0,
        excavation=round(ex, 4) if ex is not None else 0.0,
        landing=round(la, 4) if la is not None else 0.0,
        weights=w,
        inputs=inputs,
        crism_obs_id=crism_obs_id,
        crism_minerals=crism_minerals or {},
        layers_available=layers_available,
        layers_total=5,  # TES TI, elevation, slope, landform, CRISM
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
    thermal_inertia: Optional[np.ndarray],
    elevation: Optional[np.ndarray],
    slope: Optional[np.ndarray],
    tri: Optional[np.ndarray],
    shape: tuple[int, int],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Vectorised accessibility for a 2D grid. Returns float32 (0-1), NaN = no data.

    Note: Grid computation only uses excavation+landing (TES/MOLA data).
    ice_landform, water_mineral, surface_ice are point-query only (per-observation).
    """
    h, w = shape
    wd = dict(DEFAULT_WEIGHTS)
    if weights:
        wd.update(weights)
    wd = _normalize_weights(wd)

    # For grid rendering, only excavation + landing are available
    # Renormalize weights to just these two
    grid_keys = ["excavation", "landing"]
    grid_w = {k: wd.get(k, 0) for k in grid_keys}
    tw = sum(grid_w.values())
    if tw > 0:
        grid_w = {k: v / tw for k, v in grid_w.items()}

    # --- Excavation (TI only) ---
    excavation: Optional[np.ndarray] = None
    if thermal_inertia is not None:
        valid = np.isfinite(thermal_inertia) & (thermal_inertia > 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - (thermal_inertia[valid] - 200.0) / 1800.0, 0, 1)
        excavation = a

    # --- Landing ---
    la_parts: list[tuple[np.ndarray, float]] = []
    if elevation is not None:
        valid = np.isfinite(elevation)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.where(
            elevation[valid] < 0, 1.0,
            np.clip(1.0 - elevation[valid] / 2000.0, 0, 1),
        )
        la_parts.append((a, 0.55))
    if slope is not None:
        valid = np.isfinite(slope) & (slope >= 0)
        a = np.full((h, w), np.nan, dtype=np.float32)
        a[valid] = np.clip(1.0 - slope[valid] / 15.0, 0, 1)
        la_parts.append((a, 0.45))
    # TRI removed from landing scoring
    landing = _weighted_mean_parts(la_parts, shape)

    # --- Composite ---
    result = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for name, grid in [
        ("excavation", excavation),
        ("landing", landing),
    ]:
        if grid is not None:
            valid = np.isfinite(grid)
            result[valid] += grid[valid] * grid_w[name]
            weight_sum[valid] += grid_w[name]

    ok = weight_sum > 0
    result[ok] /= weight_sum[ok]
    result[~ok] = np.nan

    return np.clip(result, 0, 1)
