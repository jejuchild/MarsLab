# From Morgan & Putzig 2025, Table 1
SWIM_WEIGHTS = {
    "0-1m": {
        "neutron": 1.0,
        "thermal": 1.0,
        "radar_surface": 1.0,
        "geomorphic_shallow": 1.0,
    },
    "1-5m": {
        "radar_surface": 1.0,
        "radar_dielectric": 1.0,
        "geomorphic_shallow": 1.0,
    },
    ">5m": {
        "radar_dielectric": 1.0,
        "geomorphic_deep": 1.0,
    },
}

ALL_SWIM_METHODS = tuple(
    sorted({method for depth_weights in SWIM_WEIGHTS.values() for method in depth_weights.keys()})
)
