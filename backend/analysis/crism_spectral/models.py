from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np


class BandParameterResult(BaseModel):
    """Band parameter values for diagnostic absorption features."""
    BD1500: Optional[float] = None  # 1.5um H2O ice absorption depth
    BD1900: Optional[float] = None  # 1.9um H2O absorption depth
    BD2100: Optional[float] = None  # 2.1um ice / monohydrated sulfate
    BD2200: Optional[float] = None  # 2.2um Al-OH absorption depth

    class Config:
        arbitrary_types_allowed = True


class SAMResult(BaseModel):
    """Spectral Angle Mapper result for a single pixel."""
    best_match: str  # endmember name
    best_angle_rad: float  # smallest SAM angle
    angles: Dict[str, float] = Field(default_factory=dict)  # all endmember angles
    confidence: float = 0.0  # 1 - (best_angle / max_discriminating_angle)


class PixelClassification(BaseModel):
    """Complete classification for a single pixel."""
    mineral_class: str
    sam_confidence: float
    band_params: BandParameterResult = Field(default_factory=BandParameterResult)
    band_strength: float = 0.0  # max band depth value


class CrismSpectralResult(BaseModel):
    """Full spectral analysis result for one CRISM observation."""
    obs_id: str
    rows: int = 0
    cols: int = 0
    n_valid_pixels: int = 0
    n_classified_pixels: int = 0

    # Per-class pixel counts
    class_counts: Dict[str, int] = Field(default_factory=dict)
    class_fractions: Dict[str, float] = Field(default_factory=dict)

    # Dominant mineral
    dominant_mineral: Optional[str] = None
    dominant_fraction: float = 0.0

    # Ice-specific metrics
    water_ice_fraction: float = 0.0
    water_ice_pixels: int = 0
    water_ice_mean_confidence: float = 0.0

    # Band parameter statistics (mean over classified water-ice pixels)
    mean_BD1500: Optional[float] = None
    mean_BD1900: Optional[float] = None
    mean_BD2100: Optional[float] = None
    mean_BD2200: Optional[float] = None

    # Overall metrics
    mean_sam_confidence: float = 0.0
    atmospheric_uncertainty_notes: str = ""

    # Physics score and interpretation (Phase 0.3)
    physics_score: float = 0.0  # 0-1, weighted combination of band depths + SAM
    interpretation_label: str = "Unanalyzed"  # "Likely H2O ice", "Likely hydrated minerals", "Ambiguous/atmospheric contamination risk"

    success: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    class Config:
        arbitrary_types_allowed = True
