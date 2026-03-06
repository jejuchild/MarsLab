"""
D. Seismic Risk + Surface Feature Correlation.

Correlates PINNs-derived interior velocity anomalies with surface
geological features detected by MOLA and HiRISE to produce a
seismic-geological risk assessment.

Physics:
  - Low crustal Vp anomalies → potential thermal anomaly, weaker crust
  - Velocity discontinuities with depth → layered structure, fault zones
  - Surface grabens → extensional tectonics, potentially active faulting
  - Volcanic constructs → magmatic conduits, higher heat flow
  - LDA/LVF features → ice-rich masses, slope instability

The module answers: "What is the seismic and geological hazard at this
location, and how do surface features correlate with interior structure?"

Integration points:
  - PINNs Interior → crustal Vp profile, anomaly detection
  - MOLA detect → craters, grabens, ridges, volcanic constructs
  - HiRISE landforms → LDA/LVF/CCF classification
  - Terrain router → slope statistics
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference seismic velocity model (Knapmeyer-Endrun et al. 2021, InSight)
# ---------------------------------------------------------------------------

# Expected Vp at various depths (km/s) based on InSight SEIS data
REFERENCE_VP = {
    10: 3.5,    # Shallow crust: fractured basalt
    30: 4.5,    # Mid-crust: consolidated basalt
    50: 5.5,    # Lower crust: denser rock
    100: 6.5,   # Upper mantle transition
    200: 7.5,   # Mantle
    500: 8.0,   # Deep mantle
}

# Anomaly thresholds
ANOMALY_THRESHOLDS = {
    "low": 5.0,       # <5% anomaly = normal variation
    "moderate": 10.0,  # 5-10% = noteworthy
    "elevated": 20.0,  # 10-20% = significant
    "high": 20.0,      # >20% = major anomaly
}

# Surface feature seismic relevance weights
FEATURE_SEISMIC_WEIGHTS = {
    "graben": 0.9,           # Extensional faulting — direct tectonic indicator
    "volcanic": 0.8,         # Magmatic activity — heat flow anomaly
    "terraced_crater": 0.4,  # Complex structure but not tectonic
    "crater": 0.2,           # Impact, not endogenic
    "ridge": 0.6,            # Compressional tectonics — wrinkle ridges
    "channel": 0.3,          # Fluvial/erosional, not seismic
    "LDA": 0.5,              # Ice-rich, potential instability
    "LVF": 0.5,              # Lineated Valley Fill — ice flow
    "CCF": 0.3,              # Concentric Crater Fill
    "OTHER": 0.1,            # Generic
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VelocityAnomaly:
    """Velocity anomaly at a specific depth."""
    depth_km: float
    vp_predicted_km_s: float
    vp_reference_km_s: float
    anomaly_pct: float
    interpretation: str


@dataclass
class SeismicProfile:
    """Complete seismic velocity profile with anomalies."""
    anomalies: List[VelocityAnomaly]
    max_anomaly_pct: float
    mean_anomaly_pct: float
    risk_level: str            # "low", "moderate", "elevated", "high"
    profile_source: str        # "pinns" or "default"
    interpretation: str


@dataclass
class SurfaceFeature:
    """A detected surface geological feature."""
    feature_type: str
    count: int
    seismic_relevance: float   # 0–1
    details: str


@dataclass
class CorrelationResult:
    """Correlation between seismic anomalies and surface features."""
    correlation_score: float   # 0–1, higher = stronger correlation
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


@dataclass
class SeismicSurfaceResult:
    """Complete seismic risk + surface feature assessment."""
    lat: float
    lon: float
    seismic_profile: SeismicProfile
    surface_features: List[SurfaceFeature]
    terrain_context: Dict
    correlation: CorrelationResult
    overall_risk_score: float  # 0–1
    risk_grade: str            # "Low", "Moderate", "Elevated", "High"
    summary: str


# ---------------------------------------------------------------------------
# PINNs seismic analysis
# ---------------------------------------------------------------------------

def _query_pinns_profile(sample_depths_km: Optional[List[float]] = None,
                         ) -> Tuple[List[VelocityAnomaly], str]:
    """Query PINNs for Vp at multiple depths and compute anomalies.

    Returns (anomalies, source).
    """
    if sample_depths_km is None:
        sample_depths_km = [10, 30, 50, 100, 200, 500]

    try:
        from pinns_interior.predictor import get_predictor, is_model_trained
        if not is_model_trained():
            return _default_profile(sample_depths_km), "default"

        pred = get_predictor()
        anomalies = []

        for depth in sample_depths_km:
            vp = float(pred.predict_velocity(depth_km=float(depth)))
            ref = _get_reference_vp(depth)
            anom_pct = abs(vp - ref) / ref * 100.0 if ref > 0 else 0.0

            interp = _interpret_anomaly(depth, vp, ref, anom_pct)
            anomalies.append(VelocityAnomaly(
                depth_km=depth,
                vp_predicted_km_s=round(vp, 3),
                vp_reference_km_s=round(ref, 3),
                anomaly_pct=round(anom_pct, 1),
                interpretation=interp,
            ))

        return anomalies, "pinns"

    except Exception as exc:
        logger.debug("PINNs unavailable: %s", exc)
        return _default_profile(sample_depths_km), "default"


def _default_profile(depths: List[float]) -> List[VelocityAnomaly]:
    """Return a default (no-anomaly) profile when PINNs is unavailable."""
    anomalies = []
    for depth in depths:
        ref = _get_reference_vp(depth)
        anomalies.append(VelocityAnomaly(
            depth_km=depth,
            vp_predicted_km_s=round(ref, 3),
            vp_reference_km_s=round(ref, 3),
            anomaly_pct=0.0,
            interpretation=f"Default reference Vp at {depth}km depth",
        ))
    return anomalies


def _get_reference_vp(depth_km: float) -> float:
    """Interpolate reference Vp at arbitrary depth."""
    ref_depths = sorted(REFERENCE_VP.keys())
    ref_vps = [REFERENCE_VP[d] for d in ref_depths]

    if depth_km <= ref_depths[0]:
        return ref_vps[0]
    if depth_km >= ref_depths[-1]:
        return ref_vps[-1]

    # Linear interpolation
    for i in range(len(ref_depths) - 1):
        if ref_depths[i] <= depth_km <= ref_depths[i + 1]:
            frac = (depth_km - ref_depths[i]) / (ref_depths[i + 1] - ref_depths[i])
            return ref_vps[i] + frac * (ref_vps[i + 1] - ref_vps[i])

    return ref_vps[-1]


def _interpret_anomaly(depth_km: float, vp: float, ref: float, anom_pct: float) -> str:
    """Generate human-readable interpretation of velocity anomaly."""
    if anom_pct < ANOMALY_THRESHOLDS["low"]:
        return f"Normal velocity at {depth_km}km: Vp={vp:.2f} km/s (ref={ref:.2f})"

    direction = "low" if vp < ref else "high"

    if depth_km <= 30:
        layer = "shallow crust"
        if direction == "low":
            cause = "fractured/porous rock, possible thermal anomaly"
        else:
            cause = "dense intrusive body, well-consolidated crust"
    elif depth_km <= 100:
        layer = "lower crust"
        if direction == "low":
            cause = "elevated temperature, possible partial melt"
        else:
            cause = "mafic/ultramafic cumulates, cold stable crust"
    else:
        layer = "mantle"
        if direction == "low":
            cause = "thermal plume, elevated mantle temperature"
        else:
            cause = "cold mantle root, depleted lithosphere"

    return (
        f"{'Anomalous' if anom_pct > ANOMALY_THRESHOLDS['moderate'] else 'Slightly anomalous'} "
        f"Vp at {depth_km}km ({layer}): {vp:.2f} km/s vs ref {ref:.2f} km/s "
        f"({direction} by {anom_pct:.1f}%) — suggests {cause}"
    )


def _classify_seismic_risk(anomalies: List[VelocityAnomaly]) -> Tuple[str, float, float]:
    """Classify overall seismic risk from velocity anomalies.

    Returns (risk_level, max_anomaly, mean_anomaly).
    """
    if not anomalies:
        return "low", 0.0, 0.0

    anom_values = [a.anomaly_pct for a in anomalies]
    max_anom = max(anom_values)
    mean_anom = sum(anom_values) / len(anom_values)

    if max_anom > ANOMALY_THRESHOLDS["high"]:
        return "high", max_anom, mean_anom
    elif max_anom > ANOMALY_THRESHOLDS["moderate"]:
        return "elevated", max_anom, mean_anom
    elif max_anom > ANOMALY_THRESHOLDS["low"]:
        return "moderate", max_anom, mean_anom
    else:
        return "low", max_anom, mean_anom


# ---------------------------------------------------------------------------
# Surface feature queries
# ---------------------------------------------------------------------------

def _query_surface_features(lat: float, lon: float) -> List[SurfaceFeature]:
    """Query MOLA detect + HiRISE landforms for surface geological features."""
    features = []

    # 1. Try MOLA crater/volcanic/graben/ridge detection
    features.extend(_query_mola_features(lat, lon))

    # 2. Try HiRISE landform classification
    features.extend(_query_hirise_features(lat, lon))

    return features


def _query_mola_features(lat: float, lon: float) -> List[SurfaceFeature]:
    """Query MOLA-based feature detection near location."""
    features = []

    try:
        from analysis.mola_detect.common import build_shared_context
        from analysis.mola_detect.crater_detect import detect_craters_and_volcanics
        from analysis.mola_detect.lda_detect import detect_ldas
        from analysis.mola_detect.ridge_channel_detect import detect_ridges, detect_channels

        # Build shared DEM context (50 km radius scan)
        ctx = build_shared_context(lat, lon, radius_km=50.0)

        # Crater detection
        try:
            craters = detect_craters_and_volcanics(lat, lon, radius_km=50.0, ctx=ctx)
            if craters:
                n_volcanic = sum(1 for c in craters if c.feature_type == "volcanic")
                n_graben = sum(1 for c in craters if c.feature_type == "graben")
                n_crater = sum(1 for c in craters if "crater" in c.feature_type)

                if n_volcanic > 0:
                    features.append(SurfaceFeature(
                        feature_type="volcanic",
                        count=n_volcanic,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS["volcanic"],
                        details=f"{n_volcanic} volcanic construct(s) detected within 50km",
                    ))
                if n_graben > 0:
                    features.append(SurfaceFeature(
                        feature_type="graben",
                        count=n_graben,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS["graben"],
                        details=f"{n_graben} graben/extensional fault(s) detected within 50km",
                    ))
                if n_crater > 0:
                    features.append(SurfaceFeature(
                        feature_type="crater",
                        count=n_crater,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS["crater"],
                        details=f"{n_crater} impact crater(s) detected within 50km",
                    ))
        except Exception as exc:
            logger.debug("Crater detection failed: %s", exc)

        # Ridge/channel detection
        try:
            ridges = detect_ridges(lat, lon, radius_km=50.0, ctx=ctx)
            channels = detect_channels(lat, lon, radius_km=50.0, ctx=ctx)
            all_rc = (ridges or []) + (channels or [])
            if all_rc:
                n_ridge = sum(1 for f in all_rc if f.feature_type == "ridge")
                n_channel = sum(1 for f in all_rc if f.feature_type == "channel")

                if n_ridge > 0:
                    features.append(SurfaceFeature(
                        feature_type="ridge",
                        count=n_ridge,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS["ridge"],
                        details=f"{n_ridge} wrinkle ridge(s) — compressional tectonics",
                    ))
                if n_channel > 0:
                    features.append(SurfaceFeature(
                        feature_type="channel",
                        count=n_channel,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS["channel"],
                        details=f"{n_channel} channel/valley feature(s)",
                    ))
        except Exception as exc:
            logger.debug("Ridge/channel detection failed: %s", exc)

        # LDA detection
        try:
            ldas = detect_ldas(lat, lon, radius_km=50.0, ctx=ctx)
            if ldas:
                features.append(SurfaceFeature(
                    feature_type="LDA",
                    count=len(ldas),
                    seismic_relevance=FEATURE_SEISMIC_WEIGHTS["LDA"],
                    details=f"{len(ldas)} Lobate Debris Apron(s) — ice-rich mass wasting",
                ))
        except Exception as exc:
            logger.debug("LDA detection failed: %s", exc)

    except Exception as exc:
        logger.debug("MOLA feature detection unavailable: %s", exc)

    return features


def _query_hirise_features(lat: float, lon: float) -> List[SurfaceFeature]:
    """Query HiRISE landform classification results near location.

    This checks for pre-classified landform tiles in the vicinity.
    """
    try:
        from analysis.fusion.landform_cache import LandformCache
        cache = LandformCache()
        # Query nearby entries using a ~0.5° bounding box (~25km at equator)
        delta = 0.5
        entries = cache.get_entries_in_bounds(
            lat_min=lat - delta, lat_max=lat + delta,
            lon_min=lon - delta, lon_max=lon + delta,
        )

        if entries:
            features = []
            class_counts = {}
            for entry in entries:
                cls = entry.dominant_class or "OTHER"
                class_counts[cls] = class_counts.get(cls, 0) + 1

            for cls, count in class_counts.items():
                if cls in FEATURE_SEISMIC_WEIGHTS:
                    features.append(SurfaceFeature(
                        feature_type=cls,
                        count=count,
                        seismic_relevance=FEATURE_SEISMIC_WEIGHTS[cls],
                        details=f"HiRISE classified {count} observation(s) as {cls}",
                    ))
            return features

    except Exception as exc:
        logger.debug("HiRISE landform cache unavailable: %s", exc)

    return []


def _query_terrain(lat: float, lon: float) -> Dict:
    """Get terrain context (elevation, slope)."""
    try:
        from api.terrain_router import compute_slope_stats
        stats = compute_slope_stats(lat, lon, radius_m=2000)
        return {
            "elevation_m": stats.get("elevation_m", 0),
            "mean_slope_deg": stats.get("mean_slope", 0),
            "max_slope_deg": stats.get("max_slope", 0),
            "std_slope_deg": stats.get("std_slope", 0),
        }
    except Exception:
        return {
            "elevation_m": 0,
            "mean_slope_deg": 0,
            "max_slope_deg": 0,
            "std_slope_deg": 0,
        }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def _compute_correlation(
    seismic: SeismicProfile,
    features: List[SurfaceFeature],
    terrain: Dict,
) -> CorrelationResult:
    """Compute correlation between interior anomalies and surface features.

    High correlation = interior anomalies match surface tectonic features.
    Low correlation = mismatch (anomaly without surface expression, or vice versa).
    """
    supporting = []
    contradicting = []

    has_seismic_anomaly = seismic.max_anomaly_pct > ANOMALY_THRESHOLDS["low"]
    has_tectonic_features = any(
        f.seismic_relevance > 0.5 for f in features
    )

    # Check for supporting correlations
    if has_seismic_anomaly and has_tectonic_features:
        supporting.append(
            "Interior velocity anomaly corroborated by surface tectonic features"
        )
        for f in features:
            if f.seismic_relevance > 0.5:
                supporting.append(f"  - {f.feature_type}: {f.details}")

    # Volcanic features + low Vp = consistent with thermal anomaly
    volcanic = [f for f in features if f.feature_type == "volcanic"]
    shallow_anomalies = [
        a for a in seismic.anomalies
        if a.depth_km <= 50 and a.vp_predicted_km_s < a.vp_reference_km_s
    ]
    if volcanic and shallow_anomalies:
        supporting.append(
            "Volcanic surface features consistent with low-velocity shallow crust "
            "(potential thermal anomaly / magmatic conduit)"
        )

    # Graben + seismic anomaly = extensional tectonics
    grabens = [f for f in features if f.feature_type == "graben"]
    if grabens and has_seismic_anomaly:
        supporting.append(
            "Graben/extensional faults consistent with crustal stress from interior anomaly"
        )

    # LDA + ice-related features at mid-latitudes
    ldas = [f for f in features if f.feature_type in ("LDA", "LVF")]
    if ldas and terrain.get("mean_slope_deg", 0) > 5:
        supporting.append(
            "Ice-rich mass wasting features on sloped terrain — potential instability"
        )

    # Check for contradictions
    if has_seismic_anomaly and not has_tectonic_features and not features:
        contradicting.append(
            "Interior anomaly detected but no surface tectonic features found — "
            "anomaly may be deep-seated without surface expression"
        )

    if not has_seismic_anomaly and has_tectonic_features:
        contradicting.append(
            "Surface tectonic features present but no interior velocity anomaly — "
            "features may be ancient/inactive"
        )

    # Compute correlation score
    if not features and not has_seismic_anomaly:
        score = 0.5  # Neutral — no signal either way

    elif supporting and not contradicting:
        # Strong correlation
        relevance_sum = sum(f.seismic_relevance for f in features)
        score = min(0.95, 0.5 + 0.1 * len(supporting) + 0.05 * relevance_sum)

    elif contradicting and not supporting:
        score = max(0.1, 0.5 - 0.15 * len(contradicting))

    else:
        # Mixed signals
        score = 0.5 + 0.05 * (len(supporting) - len(contradicting))
        score = max(0.1, min(0.9, score))

    return CorrelationResult(
        correlation_score=round(score, 3),
        supporting_evidence=supporting,
        contradicting_evidence=contradicting,
    )


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def _compute_overall_risk(
    seismic: SeismicProfile,
    features: List[SurfaceFeature],
    correlation: CorrelationResult,
    terrain: Dict,
) -> Tuple[float, str]:
    """Compute overall seismic-geological risk score (0–1).

    Combines interior anomaly severity, surface feature relevance,
    correlation strength, and terrain instability.
    """
    # 1. Seismic component (40% weight)
    seismic_score = min(1.0, seismic.max_anomaly_pct / 25.0)

    # 2. Surface feature component (25% weight)
    if features:
        feature_score = min(1.0, sum(
            f.seismic_relevance * min(f.count, 5) / 5.0
            for f in features
        ) / max(len(features), 1))
    else:
        feature_score = 0.0

    # 3. Correlation boost/penalty (20% weight)
    corr_score = correlation.correlation_score

    # 4. Terrain instability (15% weight)
    slope = terrain.get("mean_slope_deg", 0)
    terrain_score = min(1.0, slope / 20.0)  # >20° = maximum risk

    # Weighted combination
    risk = (
        0.40 * seismic_score +
        0.25 * feature_score +
        0.20 * corr_score +
        0.15 * terrain_score
    )

    # Grade
    if risk >= 0.7:
        grade = "High"
    elif risk >= 0.4:
        grade = "Elevated"
    elif risk >= 0.2:
        grade = "Moderate"
    else:
        grade = "Low"

    return round(risk, 3), grade


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def assess_seismic_surface(
    lat: float,
    lon: float,
) -> SeismicSurfaceResult:
    """
    Full seismic risk + surface feature correlation assessment.

    Queries PINNs interior model for velocity anomalies, MOLA and
    HiRISE for surface geological features, and computes the correlation
    between interior structure and surface expression.

    Parameters
    ----------
    lat : float
        Latitude in degrees (-90 to 90).
    lon : float
        Longitude in degrees.

    Returns
    -------
    SeismicSurfaceResult with risk score, profile, features, and correlation.
    """
    # 1. Get seismic profile from PINNs
    anomalies, source = _query_pinns_profile()
    risk_level, max_anom, mean_anom = _classify_seismic_risk(anomalies)

    # Build interpretation
    if risk_level == "high":
        interp = (
            f"Significant velocity anomalies detected (max {max_anom:.1f}%). "
            "Interior structure deviates substantially from reference model — "
            "potential thermal anomaly, weak crustal zone, or compositional heterogeneity."
        )
    elif risk_level == "elevated":
        interp = (
            f"Notable velocity anomalies (max {max_anom:.1f}%). "
            "Interior structure shows moderate deviations — possible "
            "localized thermal or compositional variations."
        )
    elif risk_level == "moderate":
        interp = (
            f"Minor velocity variations (max {max_anom:.1f}%). "
            "Interior structure largely consistent with reference model."
        )
    else:
        interp = (
            "Velocity profile consistent with reference model. "
            "No significant interior anomalies detected."
        )

    seismic_profile = SeismicProfile(
        anomalies=anomalies,
        max_anomaly_pct=round(max_anom, 1),
        mean_anomaly_pct=round(mean_anom, 1),
        risk_level=risk_level,
        profile_source=source,
        interpretation=interp,
    )

    # 2. Get surface features
    features = _query_surface_features(lat, lon)

    # 3. Get terrain context
    terrain = _query_terrain(lat, lon)

    # 4. Compute correlation
    correlation = _compute_correlation(seismic_profile, features, terrain)

    # 5. Compute overall risk
    overall_risk, risk_grade = _compute_overall_risk(
        seismic_profile, features, correlation, terrain
    )

    # 6. Build summary
    summary_parts = []
    summary_parts.append(
        f"Seismic-geological assessment at ({lat:.1f}°, {lon:.1f}°):"
    )
    summary_parts.append(
        f"Interior: {risk_level} seismic risk "
        f"(max Vp anomaly {max_anom:.1f}%, source: {source})."
    )

    if features:
        feature_desc = ", ".join(
            f"{f.count} {f.feature_type}" for f in features
        )
        summary_parts.append(f"Surface features: {feature_desc}.")
    else:
        summary_parts.append("No significant surface geological features detected nearby.")

    summary_parts.append(
        f"Correlation score: {correlation.correlation_score:.2f} "
        f"({len(correlation.supporting_evidence)} supporting, "
        f"{len(correlation.contradicting_evidence)} contradicting)."
    )

    summary_parts.append(
        f"Overall risk: {risk_grade} ({overall_risk:.2f}/1.00)."
    )

    if terrain.get("mean_slope_deg", 0) > 10:
        summary_parts.append(
            f"⚠ Steep terrain (mean slope {terrain['mean_slope_deg']:.1f}°) "
            "increases instability risk."
        )

    return SeismicSurfaceResult(
        lat=lat,
        lon=lon,
        seismic_profile=seismic_profile,
        surface_features=features,
        terrain_context=terrain,
        correlation=correlation,
        overall_risk_score=overall_risk,
        risk_grade=risk_grade,
        summary=" ".join(summary_parts),
    )


def compare_seismic_risk(
    sites: List[Dict],
) -> List[SeismicSurfaceResult]:
    """
    Assess and rank multiple sites by seismic-geological risk.

    Parameters
    ----------
    sites : list of dict
        Each must have 'lat' and 'lon'.

    Returns
    -------
    List of SeismicSurfaceResult, sorted by overall_risk_score ascending (safest first).
    """
    results = []
    for site in sites:
        result = assess_seismic_surface(
            lat=site["lat"],
            lon=site["lon"],
        )
        results.append(result)

    results.sort(key=lambda r: r.overall_risk_score)
    return results
