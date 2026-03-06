"""
MarsLab Integration Modules.

Cross-system analyses that combine Neural Climate, PINNs Interior,
SHARAD, CRISM, MOLA, and other MarsLab subsystems.

Modules:
    A. landing_site_scorer   — Landing site suitability scoring
    B. mineral_stability     — Climate-mineral stability mapping
    C. ice_evolution         — Subsurface ice evolution model
    D. seismic_surface       — Seismic risk + surface feature correlation
"""


from .landing_site_scorer import score_landing_site, compare_sites
from .mineral_stability import assess_mineral_stability, seasonal_stability_profile, brine_habitability_window
from .ice_evolution import compute_ice_stability, assess_ice_evolution, depth_stability_map
from .seismic_surface import assess_seismic_surface, compare_seismic_risk

__all__ = [
    "score_landing_site",
    "compare_sites",
    "assess_mineral_stability",
    "seasonal_stability_profile",
    "brine_habitability_window",
    "compute_ice_stability",
    "assess_ice_evolution",
    "depth_stability_map",
    "assess_seismic_surface",
    "compare_seismic_risk",
]