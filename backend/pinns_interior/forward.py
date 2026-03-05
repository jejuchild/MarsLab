from __future__ import annotations

import numpy as np

from .mars_model import MARS_RADIUS_KM


def _extract_profile(velocity_profile: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(velocity_profile, dict):
        radius = np.asarray(velocity_profile["radius_km"], dtype=np.float64)
        vp = np.asarray(velocity_profile["vp_km_s"], dtype=np.float64)
        return radius, vp
    if isinstance(velocity_profile, tuple) and len(velocity_profile) >= 2:
        radius = np.asarray(velocity_profile[0], dtype=np.float64)
        vp = np.asarray(velocity_profile[1], dtype=np.float64)
        return radius, vp
    raise TypeError("velocity_profile must be dict or tuple(radius, vp, ...)")


def compute_travel_time(velocity_profile: object, epicentral_distance_deg: float) -> float:
    radius_km, vp_km_s = _extract_profile(velocity_profile)
    order = np.argsort(radius_km)
    radius_km = radius_km[order]
    vp_km_s = vp_km_s[order]

    delta = np.deg2rad(float(epicentral_distance_deg))
    delta = float(np.clip(delta, np.deg2rad(1.0), np.deg2rad(179.0)))

    b = MARS_RADIUS_KM * np.cos(0.5 * delta)
    p = b / np.interp(b, radius_km, vp_km_s)
    r_min = max(float(radius_km.min()), b)
    r_grid = np.linspace(r_min, MARS_RADIUS_KM, 600)
    v_grid = np.interp(r_grid, radius_km, vp_km_s)

    denom = np.sqrt(np.maximum(1e-8, 1.0 - (p * v_grid / np.maximum(r_grid, 1e-6)) ** 2))
    integrand = 1.0 / np.maximum(v_grid, 1e-6) / denom
    one_leg = np.trapezoid(integrand, r_grid)
    return float(2.0 * one_leg)


def generate_synthetic_data(
    reference_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    n_events: int = 20,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    distances = np.sort(rng.uniform(20.0, 150.0, size=int(n_events))).astype(np.float32)
    clean = np.array([compute_travel_time(reference_model, d) for d in distances], dtype=np.float32)
    noisy = clean + rng.normal(0.0, 1.5, size=clean.shape).astype(np.float32)
    return distances, noisy
