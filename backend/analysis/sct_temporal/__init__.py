"""
SCT Temporal Change Detection — Scalloped Terrain Scarp Retreat Measurement.

Two-stage pipeline:
  Stage 1: MarsLandformNet classifies SCT regions from HiRISE browse images
  Stage 2: Full-res HiRISE temporal pairs → phase correlation → sub-pixel displacement

Fills a literature gap: no direct HiRISE temporal measurement of SCT scarp retreat
exists (Dundas 2015 modeled only). This module provides the first systematic
measurement capability using COSI-Corr-style phase correlation at 25 cm/px.
"""

from .phase_correlation import (
    cosicorr_phase_correlation,
    sliding_window_correlation,
    CorrelationResult,
    DisplacementField,
)
from .pair_finder import find_temporal_pairs, HiRISEProduct, TemporalPair
from .hirise_download import download_hirise_rdr
from .coregistration import coregister_geotiffs, CoregisteredPair
from .scarp_analysis import measure_retreat, RetreatAnalysis, ScarpSegment
from .pipeline import SCTTemporalPipeline, PipelineConfig, PipelineResult
from .visualize import plot_displacement_field, plot_retreat_analysis, plot_temporal_comparison

__all__ = [
    # Phase correlation
    "cosicorr_phase_correlation",
    "sliding_window_correlation",
    "CorrelationResult",
    "DisplacementField",
    # Pair finding
    "find_temporal_pairs",
    "HiRISEProduct",
    "TemporalPair",
    # Download
    "download_hirise_rdr",
    # Co-registration
    "coregister_geotiffs",
    "CoregisteredPair",
    # Scarp analysis
    "measure_retreat",
    "RetreatAnalysis",
    "ScarpSegment",
    # Pipeline
    "SCTTemporalPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Visualization
    "plot_displacement_field",
    "plot_retreat_analysis",
    "plot_temporal_comparison",
]
