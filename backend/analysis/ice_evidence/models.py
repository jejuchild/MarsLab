"""
Pydantic data models for the Ice Evidence Synthesizer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Hyperbola Fit ────────────────────────────────────────

class HyperbolaFitOptions(BaseModel):
    bootstrap_iters: int = Field(200, ge=0, le=2000)
    epsr_presets: List[float] = Field(default_factory=lambda: [2.8, 3.15, 4.0])


class HyperbolaFitRequest(BaseModel):
    product_id: str
    apex_trace: int
    apex_bin: int
    roi_traces: int = Field(60, ge=5, le=500, description="Half-width in traces")
    roi_bins: int = Field(80, ge=5, le=400, description="Half-height in bins")
    dx_m_per_trace: float = Field(0.0, ge=0, description="Along-track m/trace; 0 = auto-compute")
    dt_s_per_bin: float = Field(3.75e-8, gt=0, description="Two-way travel time per bin (s)")
    surface_pick_bins: Optional[List[int]] = None
    options: HyperbolaFitOptions = Field(default_factory=HyperbolaFitOptions)


class OverlayPoint(BaseModel):
    trace: int
    bin: float


class OverlayCIPoint(BaseModel):
    trace: int
    bin_lo: float
    bin_hi: float


class HyperbolaFitQuality(BaseModel):
    snr: float = 0.0
    residual: float = 0.0
    support_traces: int = 0


class HyperbolaFitResult(BaseModel):
    v_mps: float
    epsr: float
    epsr_ci95: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    dt_surface_s: float = 0.0
    depth_m: float = 0.0
    depth_ci95: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    quality: HyperbolaFitQuality = Field(default_factory=HyperbolaFitQuality)
    flags: List[str] = Field(default_factory=list)
    overlay_polyline: List[OverlayPoint] = Field(default_factory=list)
    overlay_ci_band: Optional[List[OverlayCIPoint]] = None
    preset_curves: Optional[Dict[str, List[OverlayPoint]]] = None


# ── Ice Evidence ─────────────────────────────────────────

class CandidateLocation(BaseModel):
    lat: float
    lon: float
    id: str


class RegionSpec(BaseModel):
    name: str = ""
    bbox: List[float] = Field(default_factory=list, description="[minlon, minlat, maxlon, maxlat]")


class SharadSpec(BaseModel):
    tracks: Optional[List[str]] = None


class CrismSpec(BaseModel):
    products: Optional[List[str]] = None


class DtmSpec(BaseModel):
    hirise_dtm_products: Optional[List[str]] = None


class EvidenceWeights(BaseModel):
    E1: float = 0.30
    E2: float = 0.30
    E3: float = 0.15
    E4: float = 0.25


class EvidenceParams(BaseModel):
    coincidence_km_scales: List[float] = Field(default_factory=lambda: [25, 50, 100, 200])
    epsr_ice_range: List[float] = Field(default_factory=lambda: [2.7, 3.4])
    distance_penalty_km: float = 50.0
    weights: EvidenceWeights = Field(default_factory=EvidenceWeights)


class IceEvidenceRequest(BaseModel):
    region: RegionSpec = Field(default_factory=RegionSpec)
    candidates: List[CandidateLocation]
    sharad: SharadSpec = Field(default_factory=SharadSpec)
    crism: CrismSpec = Field(default_factory=CrismSpec)
    dtm: DtmSpec = Field(default_factory=DtmSpec)
    params: EvidenceParams = Field(default_factory=EvidenceParams)


# ── Evidence sub-scores ──────────────────────────────────

class E1Hyperbola(BaseModel):
    score: float = 0.0
    epsr: Optional[float] = None
    ci: Optional[List[float]] = None
    flags: List[str] = Field(default_factory=list)
    notes: str = ""


class E2Reflector(BaseModel):
    score: float = 0.0
    snr_stats: Optional[Dict[str, float]] = None
    continuity: float = 0.0
    notes: str = ""


class E3Terrain(BaseModel):
    score: float = 0.0
    slope_stats: Optional[Dict[str, float]] = None
    roughness_proxy: float = 0.0
    notes: str = ""


class E4Crism(BaseModel):
    score: float = 0.0
    ice_score: float = 0.0
    hyd_score: float = 0.0
    distance_km: float = 0.0
    notes: str = ""


class EvidenceBreakdown(BaseModel):
    E1_hyperbola: E1Hyperbola = Field(default_factory=E1Hyperbola)
    E2_reflector: E2Reflector = Field(default_factory=E2Reflector)
    E3_terrain: E3Terrain = Field(default_factory=E3Terrain)
    E4_crism: E4Crism = Field(default_factory=E4Crism)


class ConsistencyInfo(BaseModel):
    agreement_score: float = 0.0
    conflicts: List[str] = Field(default_factory=list)


class ArtifactPaths(BaseModel):
    json_path: str = ""
    optional_png_paths: List[str] = Field(default_factory=list)


class IceEvidenceResult(BaseModel):
    candidate_id: str
    lat: float
    lon: float
    ice_probability: float = 0.0
    confidence: float = 0.0
    evidence: EvidenceBreakdown = Field(default_factory=EvidenceBreakdown)
    consistency: ConsistencyInfo = Field(default_factory=ConsistencyInfo)
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
