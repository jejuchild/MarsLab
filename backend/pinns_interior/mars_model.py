from __future__ import annotations

import numpy as np

MARS_RADIUS_KM = 3389.5


def _linear(start: float, end: float, t: np.ndarray) -> np.ndarray:
    return start + (end - start) * t


def get_reference_model(n_layers: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_layers = int(max(8, n_layers))
    depth_km = np.linspace(0.0, 1700.0, n_layers, dtype=np.float64)
    radius_km = MARS_RADIUS_KM - depth_km

    vp = np.zeros_like(depth_km)
    density = np.zeros_like(depth_km)

    crust = depth_km <= 50.0
    upper_mantle = (depth_km > 50.0) & (depth_km <= 400.0)
    lower_mantle = (depth_km > 400.0) & (depth_km <= 1500.0)
    core = depth_km > 1500.0

    if np.any(crust):
        t = (depth_km[crust] - 0.0) / 50.0
        vp[crust] = _linear(3.5, 6.0, t)
        density[crust] = _linear(2.8, 3.0, t)

    if np.any(upper_mantle):
        t = (depth_km[upper_mantle] - 50.0) / 350.0
        vp[upper_mantle] = _linear(7.5, 8.0, t)
        density[upper_mantle] = _linear(3.3, 3.5, t)

    if np.any(lower_mantle):
        t = (depth_km[lower_mantle] - 400.0) / 1100.0
        vp[lower_mantle] = _linear(8.0, 9.5, t)
        density[lower_mantle] = _linear(3.5, 4.2, t)

    if np.any(core):
        t = (depth_km[core] - 1500.0) / 200.0
        vp[core] = _linear(5.2, 4.8, t)
        density[core] = _linear(6.0, 6.5, t)

    vs = 0.55 * vp
    vs[core] = 0.0

    return radius_km.astype(np.float32), vp.astype(np.float32), vs.astype(np.float32), density.astype(np.float32)
