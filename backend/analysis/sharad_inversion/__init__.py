"""
SHARAD Physics-Based Dielectric Inversion Pipeline.

Mandatory 5-step pipeline:
  1. Terraced crater selection (DTM geometry)
  2. SHARAD two-way travel time measurement
  3. Dielectric constant inversion (εr = (c·Δt / 2d)²)
  4. Hyperbola curvature cross-validation
  5. Material interpretation

PRINCIPLE: Depth is measured independently from DTM terrace geometry.
εr is back-calculated from radar travel time. Never assume εr to compute depth.
"""
from .pipeline import run_inversion_pipeline, InversionPipelineResult
