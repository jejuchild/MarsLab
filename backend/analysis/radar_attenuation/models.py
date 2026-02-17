"""Pydantic data models for the Radar Attenuation Mapper."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AttenuationSample(BaseModel):
    """One trace in the along-track attenuation profile."""
    trace_idx: int
    lat: float                                # planetographic, degrees
    lon: float                                # -180..180
    along_track_km: float
    surface_elev_m: float                     # DEM elevation
    interface_detected: bool
    surface_power_dB: Optional[float] = None
    subsurface_power_dB: Optional[float] = None
    depth_m: Optional[float] = None           # subsurface reflector depth
    alpha_dBm: Optional[float] = None         # two-way attenuation (dB/m)
    transparency: Optional[str] = None        # material transparency class
    snr: Optional[float] = None
    confidence: Optional[float] = None        # 0.0–1.0


class OverlaySegment(BaseModel):
    """One colored polyline segment for Cesium map rendering."""
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    alpha_dBm: Optional[float] = None
    color: List[float]                        # [r, g, b, a] each 0–255


class AttenuationSummary(BaseModel):
    """Aggregate statistics for the entire track."""
    product_id: str
    epsilon_r: float
    total_traces: int
    valid_traces: int
    detection_rate: float                     # valid / total
    alpha_mean_dBm: Optional[float] = None
    alpha_median_dBm: Optional[float] = None
    alpha_std_dBm: Optional[float] = None
    dominant_transparency: Optional[str] = None
    transparency_counts: Dict[str, int] = Field(default_factory=dict)
    dem_source: str = "MOLA"
    total_distance_km: float = 0.0


class AttenuationParameters(BaseModel):
    """Input parameters echoed back for reproducibility."""
    epsilon_r: float
    snr_threshold: float
    search_lo: int
    search_hi: int
    dem_source: str
    speed_of_light_mps: float = 299_792_458.0
    sample_interval_us: float = 0.0375


class AttenuationResult(BaseModel):
    """Complete API response for /api/attenuation/profile."""
    success: bool
    error: Optional[str] = None
    summary: Optional[AttenuationSummary] = None
    profile: List[AttenuationSample] = Field(default_factory=list)
    overlay_segments: List[OverlaySegment] = Field(default_factory=list)
    parameters: Optional[AttenuationParameters] = None
