"""Pydantic data models for the Crater Stratigraphy Analyzer."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class CraterInfo(BaseModel):
    """Crater metadata echoed back from the input + detected properties."""
    lat: float                                # planetographic, degrees
    lon: float                                # -180..180
    diameter_km: float
    depth_m: Optional[float] = None
    morphology: Optional[str] = None          # "simple", "complex", "terraced"
    n_terraces: int = 0
    terrace_depths: List[Dict] = Field(default_factory=list)
    dtm_source: str = "none"                  # "HiRISE" | "MOLA" | "none"


class RadialProfilePoint(BaseModel):
    """One point in the radial elevation profile."""
    radius_m: float                           # distance from crater center
    elevation_m: float                        # median elevation across azimuths
    elevation_min: Optional[float] = None     # min across azimuths (envelope)
    elevation_max: Optional[float] = None     # max across azimuths (envelope)


class SharadPickInfo(BaseModel):
    """Summary of SHARAD subsurface picks from one track."""
    product_id: str
    n_picks: int
    median_twt_us: float                      # median two-way travel time (µs)
    twt_unc_us: float                         # TWT uncertainty (µs)
    min_distance_km: float                    # closest pick to crater center


class EpsilonEstimateInfo(BaseModel):
    """One dielectric constant estimate from a depth metric."""
    depth_metric: str                         # "terrace_to_floor", "rim_to_terrace", etc.
    depth_m: float                            # true geometric depth
    depth_unc_m: float
    twt_us: float
    epsilon_r: float                          # central εr estimate
    epsilon_low: float                        # lower bound
    epsilon_high: float                       # upper bound
    interpretation: str                       # material classification
    quality: str                              # "good" | "marginal" | "unreliable"
    quality_note: str
    sharad_product: str = ""
    n_picks: int = 0


class StratigraphyOverlayPoint(BaseModel):
    """One SHARAD pick location for Cesium map rendering."""
    lat: float
    lon: float
    epsilon_r: Optional[float] = None
    color: List[int]                          # [r, g, b, a] each 0–255


class StratigraphySummary(BaseModel):
    """Aggregate statistics for the crater stratigraphy analysis."""
    n_sharad_tracks: int
    n_epsilon_estimates: int
    best_epsilon_r: Optional[float] = None
    best_interpretation: Optional[str] = None
    best_quality: Optional[str] = None
    dem_source: str = "none"
    total_sharad_picks: int = 0
    # Phase 2: quality-weighted aggregate fields
    weighted_epsilon_r: Optional[float] = None
    weighted_epsilon_std: Optional[float] = None
    confidence_interval_68: Optional[List[float]] = None
    confidence_interval_95: Optional[List[float]] = None
    n_good_estimates: int = 0
    n_marginal_estimates: int = 0


class StratigraphyParameters(BaseModel):
    """Input parameters echoed back for reproducibility."""
    crater_lat: float
    crater_lon: float
    diameter_km: float
    buffer_km: float
    n_azimuths: int
    max_radius_m: float
    min_snr: float


class StratigraphyResult(BaseModel):
    """Complete API response for /api/stratigraphy/analyze."""
    success: bool
    error: Optional[str] = None
    crater_info: Optional[CraterInfo] = None
    summary: Optional[StratigraphySummary] = None
    radial_profile: List[RadialProfilePoint] = Field(default_factory=list)
    epsilon_estimates: List[EpsilonEstimateInfo] = Field(default_factory=list)
    sharad_picks: List[SharadPickInfo] = Field(default_factory=list)
    overlay_points: List[StratigraphyOverlayPoint] = Field(default_factory=list)
    parameters: Optional[StratigraphyParameters] = None
