"""Pydantic data models for the Stratigraphic Column Builder."""

from pydantic import BaseModel, Field
from typing import List, Optional


class ColumnLayer(BaseModel):
    """One layer in the stratigraphic column."""
    layer_idx: int
    depth_top_m: float                        # depth from rim (0 = rim)
    depth_bottom_m: float
    thickness_m: float
    source: str                               # "DTM_terrace" | "SHARAD_reflector" | "regolith"
    instrument: str                           # "HiRISE" | "MOLA" | "SHARAD" | "CRISM"
    mineral_name: Optional[str] = None
    geochem_group: Optional[str] = None
    epsilon_r: Optional[float] = None
    material_class: Optional[str] = None      # transparency classification from εr
    color: List[int] = Field(default_factory=lambda: [120, 120, 120, 180])  # RGBA
    confidence: Optional[float] = None


class ColumnSummary(BaseModel):
    """Aggregate statistics for the stratigraphic column."""
    crater_lat: float
    crater_lon: float
    diameter_km: float
    n_layers: int
    total_depth_m: float
    instruments_used: List[str] = Field(default_factory=list)
    dtm_source: str = "none"
    has_crism: bool = False
    has_sharad_subsurface: bool = False
    dominant_material: Optional[str] = None


class ColumnParameters(BaseModel):
    """Input parameters echoed back for reproducibility."""
    crater_lat: float
    crater_lon: float
    diameter_km: float
    buffer_km: float
    include_crism: bool
    include_sharad: bool


class StratColumnResult(BaseModel):
    """Complete API response for /api/strat-column/build."""
    success: bool
    error: Optional[str] = None
    summary: Optional[ColumnSummary] = None
    layers: List[ColumnLayer] = Field(default_factory=list)
    rim_elevation_m: Optional[float] = None
    parameters: Optional[ColumnParameters] = None
