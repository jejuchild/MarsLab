"""
AnalysisModule — abstract base class for MarsLab deterministic analysis modules.

All Tier-1 science algorithms (RTE, Crater Stratigraphy, RSL) extend this.
Each module must implement four methods that separate concerns:
  run()              — execute the analysis (compute-heavy)
  generate_profile() — return data suitable for a chart
  generate_overlay() — return data suitable for map rendering
  generate_summary() — return aggregate statistics
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AnalysisModule(ABC):
    """Reusable base class for MarsLab deterministic analysis modules."""

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the analysis. Returns module-specific result model.

        This is the compute-heavy method.  It should populate internal state
        so that generate_profile / generate_overlay / generate_summary can
        be called without re-running the algorithm.
        """
        ...

    @abstractmethod
    def generate_profile(self) -> List[Dict]:
        """Return along-track or spatial profile data for charting.

        Each element is a dict with at minimum:
          - a distance / position key (e.g. along_track_km)
          - one or more measurement keys
        """
        ...

    @abstractmethod
    def generate_overlay(self) -> List[Dict]:
        """Return map overlay segments.

        Each element describes one polyline segment with start/end coords
        and a color (RGBA list) determined by the measurement value.
        """
        ...

    @abstractmethod
    def generate_summary(self) -> Dict:
        """Return summary statistics dictionary.

        Must include at minimum: success (bool), error (str | None).
        """
        ...
