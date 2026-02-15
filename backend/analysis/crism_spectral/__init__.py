"""
CRISM Physics-Based Spectral Analysis Module.

Replaces threshold-based ice scoring with rigorous spectral analysis:
  1. Continuum removal (convex hull method)
  2. Band parameter extraction (BD1500, BD1900, BD2100, BD2200)
  3. Spectral Angle Mapper (SAM) against USGS endmembers
  4. Per-pixel mineral classification with confidence

Endmembers: water ice, gypsum, polyhydrated sulfate, basalt
"""
from .pipeline import run_spectral_analysis
