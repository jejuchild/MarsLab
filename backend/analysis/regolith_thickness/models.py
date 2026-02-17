"""Pydantic data models for the Regolith Thickness Estimator."""

from pydantic import BaseModel, Field
from typing import List, Optional


class RegolithSample(BaseModel):
    """One trace in the along-track regolith thickness profile."""
    trace_idx: int
    lat: float                            # planetographic, degrees
    lon: float                            # -180..180
    along_track_km: float
    surface_elev_m: float                 # DEM elevation
    interface_detected: bool
    delta_bins: Optional[int] = None
    twt_us: Optional[float] = None        # two-way travel time (µs)
    thickness_m: Optional[float] = None
    snr: Optional[float] = None
    confidence: Optional[float] = None    # 0.0–1.0


class OverlaySegment(BaseModel):
    """One colored polyline segment for Cesium map rendering."""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    thickness_m: Optional[float] = None
    color: List[float]                    # [r, g, b, a] each 0–255


class RegolithSummary(BaseModel):
    """Aggregate statistics for the entire track."""
    product_id: str
    epsilon_r: float
    total_traces: int
    valid_traces: int
    detection_rate: float                 # valid / total
    thickness_mean_m: Optional[float] = None
    thickness_median_m: Optional[float] = None
    thickness_std_m: Optional[float] = None
    thickness_min_m: Optional[float] = None
    thickness_max_m: Optional[float] = None
    mean_snr: Optional[float] = None
    mean_confidence: Optional[float] = None
    dem_source: str = "MOLA"              # "MOLA" or "HiRISE_DTM"
    total_distance_km: float = 0.0


class RegolithParameters(BaseModel):
    """Input parameters echoed back for reproducibility."""
    epsilon_r: float
    snr_threshold: float
    search_lo: int
    search_hi: int
    dem_source: str
    speed_of_light_mps: float = 299_792_458.0
    sample_interval_us: float = 0.0375


class RegolithResult(BaseModel):
    """Complete API response for /api/regolith/thickness_profile."""
    success: bool
    error: Optional[str] = None
    summary: Optional[RegolithSummary] = None
    profile: List[RegolithSample] = Field(default_factory=list)
    overlay_segments: List[OverlaySegment] = Field(default_factory=list)
    parameters: Optional[RegolithParameters] = None
