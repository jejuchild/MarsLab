from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DerivationStep(BaseModel):
    """A single step in the numerical derivation chain."""
    step_number: int
    description: str
    formula: str = ""
    inputs: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

class PhysicalAssumption(BaseModel):
    """Logged physical assumption."""
    param: str
    value: Any
    source: str  # "measured", "DTM_inversion", "curvature_fit", "assumed", "literature"
    justification: str
    uncertainty: Optional[str] = None

class ClutterAssessment(BaseModel):
    """Cluttergram comparison result for a reflector."""
    cluttergram_available: bool = False
    clutter_likelihood_score: float = 0.0  # 0 = no clutter, 1 = definite clutter
    rejected: bool = False
    method: str = "cluttergram_comparison"
    notes: str = ""

class ReflectorPick(BaseModel):
    """A single SHARAD subsurface reflector pick with physics context."""
    product_id: str
    trace_idx: int
    lat: float
    lon: float
    surface_bin: int
    interface_bin: int
    delta_bins: int
    twt_us: float
    twt_unc_us: float
    snr: float
    clutter: ClutterAssessment = Field(default_factory=ClutterAssessment)

class TerraceDepthMeasurement(BaseModel):
    """Independent depth measurement from DTM terrace geometry."""
    crater_id: str
    dtm_product_id: str
    depth_m: float
    depth_unc_m: float
    depth_metric: str  # "upper_to_lower", "terrace_to_floor", "rim_to_terrace"
    crater_lat: float
    crater_lon: float
    n_azimuths: int = 36

class DielectricInversion(BaseModel):
    """Result of εr = (c·Δt / 2d)² inversion."""
    epsilon_r: float
    epsilon_r_low: float
    epsilon_r_high: float
    velocity_mps: float
    depth_m: float
    depth_unc_m: float
    twt_us: float
    twt_unc_us: float
    quality: str  # "good", "marginal", "unreliable"
    interpretation: str
    source: str = "DTM_terrace_inversion"
    derivation: List[DerivationStep] = Field(default_factory=list)

class HyperbolaValidation(BaseModel):
    """Cross-validation via hyperbola curvature fitting."""
    performed: bool = False
    epsilon_r_curvature: Optional[float] = None
    epsilon_r_curvature_ci95: Optional[List[float]] = None
    epsilon_r_dtm: Optional[float] = None
    agreement: Optional[str] = None  # "consistent", "marginal", "inconsistent"
    delta_epsilon: Optional[float] = None
    notes: str = ""

class InversionResult(BaseModel):
    """Complete result for one SHARAD-DTM pair."""
    crater_id: str
    dtm_product_id: str
    sharad_product_id: str
    terrace_depth: TerraceDepthMeasurement
    reflector_picks: List[ReflectorPick] = Field(default_factory=list)
    inversion: Optional[DielectricInversion] = None
    hyperbola_validation: HyperbolaValidation = Field(default_factory=HyperbolaValidation)
    assumptions: List[PhysicalAssumption] = Field(default_factory=list)
    derivation_steps: List[DerivationStep] = Field(default_factory=list)

class InversionPipelineResult(BaseModel):
    """Full pipeline output."""
    success: bool
    sharad_products_analyzed: int = 0
    dtm_intersections_found: int = 0
    inversions_completed: int = 0
    inversions: List[InversionResult] = Field(default_factory=list)
    best_epsilon_r: Optional[float] = None
    best_epsilon_r_ci: Optional[List[float]] = None
    best_interpretation: Optional[str] = None
    reflector_confidence: float = 0.0  # 0-1 overall
    assumptions: List[PhysicalAssumption] = Field(default_factory=list)
    derivation_log: List[DerivationStep] = Field(default_factory=list)
    error: Optional[str] = None
    notes: str = ""
