"""Pydantic data models for the Aqueous Mineral Sequence Mapper."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TransectPoint(BaseModel):
    """One sample along the mineral transect."""
    position_idx: int
    row: int
    col: int
    mineral_id: Optional[int] = None          # CNN class ID (-1 or None = unclassified)
    mineral_name: Optional[str] = None
    geochem_group: Optional[str] = None
    confidence: Optional[float] = None         # CNN softmax confidence


class MineralTransition(BaseModel):
    """A geochemical group transition detected along the transect."""
    position_idx: int                          # index in transect where transition occurs
    from_group: str
    to_group: str
    from_mineral: str
    to_mineral: str


class SequenceMatch(BaseModel):
    """A matched paleo-environment from the observed mineral sequence."""
    environment: str
    matched_groups: List[str]
    confidence: float                          # fraction of transect classified × mean CNN conf


class MineralSequenceSummary(BaseModel):
    """Aggregate statistics for the mineral sequence analysis."""
    obs_id: str
    total_transect_points: int
    classified_points: int
    classification_rate: float                 # classified / total
    n_transitions: int
    dominant_group: Optional[str] = None
    n_groups_present: int = 0
    matched_environments: List[str] = Field(default_factory=list)
    mean_confidence: Optional[float] = None


class MineralSequenceParameters(BaseModel):
    """Input parameters echoed back for reproducibility."""
    obs_id: str
    transect_direction: str                    # "NS" or "EW"
    transect_offset: float                     # 0.0–1.0, position of transect through image


class MineralSequenceResult(BaseModel):
    """Complete API response for /api/mineral-sequence/analyze."""
    success: bool
    error: Optional[str] = None
    summary: Optional[MineralSequenceSummary] = None
    transect: List[TransectPoint] = Field(default_factory=list)
    transitions: List[MineralTransition] = Field(default_factory=list)
    sequence_matches: List[SequenceMatch] = Field(default_factory=list)
    group_histogram: Dict[str, int] = Field(default_factory=dict)
    parameters: Optional[MineralSequenceParameters] = None
