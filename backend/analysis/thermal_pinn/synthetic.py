"""
Synthetic data generation for PINN validation (Phase 1).
V2: Extended simulation (1 Mars year) with dual time coordinates.

Generates ground-truth T(z,t) by solving 1D heat diffusion with a known
k(z) profile using Crank-Nicolson finite differences. The surface temperature
T(0,t) serves as synthetic "THEMIS observations" for PINN training.

The PINN must recover the known k(z) profile from surface observations alone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Mars physical constants
MARS_SOL = 88_775.0           # seconds per sol
MARS_YEAR_SOLS = 668.6        # sols per Mars year
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL  # seconds per Mars year
RHO = 1500.0                  # soil density [kg/m³]
C_P = 627.9                   # soil heat capacity [J/kg/K]
SIGMA = 5.670374419e-8        # Stefan-Boltzmann constant


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Domain
    z_max: float = 3.0          # maximum depth [m]
    n_z: int = 200              # spatial grid points
    n_sols: int = 668           # number of sols to simulate (1 Mars year)
    dt_per_sol: int = 144       # time steps per sol (10 min each)
    spinup_sols: int = 668      # spinup sols (1 Mars year)

    # Surface forcing
    T_mean: float = 210.0       # mean surface temperature [K]
    T_amp_diurnal: float = 40.0 # diurnal amplitude [K]
    T_amp_seasonal: float = 20.0# seasonal amplitude [K]

    # Latitude (controls diurnal cycle shape)
    latitude: float = 45.0      # degrees N (Arcadia Planitia)

    # Material properties (fixed ρ, c — only k varies with depth)
    rho: float = RHO
    c_p: float = C_P


def two_layer_k_profile(
    z: np.ndarray,
    k_upper: float = 0.02,        # dust-like [W/m/K] → TI ≈ 137 tiu
    k_lower: float = 2.0,         # ice-like  [W/m/K] → TI ≈ 1372 tiu
    boundary_depth: float = 0.5,  # layer boundary [m]
    transition_width: float = 0.05,  # smooth transition [m]
) -> np.ndarray:
    """
    Two-layer k(z) profile with smooth transition.
    Upper layer: low thermal conductivity (dust/sand).
    Lower layer: high thermal conductivity (ice-cemented soil).
    """
    k = k_upper + (k_lower - k_upper) / (
        1.0 + np.exp(-(z - boundary_depth) / transition_width)
    )
    return k


def three_layer_k_profile(
    z: np.ndarray,
    k1: float = 0.02,     # surface dust
    k2: float = 0.1,      # sand
    k3: float = 2.0,      # ice-cemented
    d1: float = 0.2,      # dust/sand boundary [m]
    d2: float = 0.8,      # sand/ice boundary [m]
    tw: float = 0.05,     # transition width [m]
) -> np.ndarray:
    """Three-layer k(z) profile."""
    s1 = 1.0 / (1.0 + np.exp(-(z - d1) / tw))
    s2 = 1.0 / (1.0 + np.exp(-(z - d2) / tw))
    k = k1 + (k2 - k1) * s1 + (k3 - k2) * s2
    return k


def surface_forcing(t: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    """
    Synthetic surface temperature forcing.
    Combines diurnal and seasonal cycles.

    T_surface(t) = T_mean + T_amp_diurnal * sin(2π t/P_sol)
                          + T_amp_seasonal * sin(2π t/P_year)
    """
    omega_d = 2.0 * np.pi / MARS_SOL
    omega_s = 2.0 * np.pi / MARS_YEAR_SEC

    T = (config.T_mean
         + config.T_amp_diurnal * np.sin(omega_d * t)
         + config.T_amp_seasonal * np.sin(omega_s * t))
    return T


def solve_heat_equation(
    k_z: np.ndarray,
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 1D heat diffusion with Crank-Nicolson scheme.

        ρc ∂T/∂t = ∂/∂z(k(z) ∂T/∂z)

    Uses precomputed Thomas factorization (constant tridiagonal matrix)
    and vectorized RHS computation for performance.

    Returns:
        T: temperature array (n_t, n_z) [K]
        z: depth array (n_z,) [m]
        t: time array (n_t,) [s]
    """
    n_z = config.n_z
    dz = config.z_max / (n_z - 1)
    z = np.linspace(0, config.z_max, n_z)

    n_spinup = config.spinup_sols
    total_sols = config.n_sols + n_spinup
    n_t_total = total_sols * config.dt_per_sol
    dt = MARS_SOL / config.dt_per_sol
    t_total = np.arange(n_t_total) * dt

    rho_c = config.rho * config.c_p

    # Interface conductivities (harmonic mean) — vectorized
    k_half = 2.0 * k_z[:-1] * k_z[1:] / (k_z[:-1] + k_z[1:])

    # Courant numbers
    r = k_half * dt / (rho_c * dz * dz)

    # Build CONSTANT tridiagonal matrix for Crank-Nicolson: (I + 0.5*A)
    a_diag = np.zeros(n_z)  # lower diagonal
    b_diag = np.zeros(n_z)  # main diagonal
    c_diag = np.zeros(n_z)  # upper diagonal

    b_diag[0] = 1.0  # Dirichlet top BC
    # Interior: vectorized
    a_diag[1:-1] = -0.5 * r[:-1]
    b_diag[1:-1] = 1.0 + 0.5 * (r[:-1] + r[1:])
    c_diag[1:-1] = -0.5 * r[1:]
    # Bottom: zero-flux BC (uses last interface r[-1])
    a_diag[-1] = -0.5 * r[-1]
    b_diag[-1] = 1.0 + 0.5 * r[-1]

    # Precompute Thomas factorization (constant — computed once)
    c_prime = np.zeros(n_z)
    w = np.zeros(n_z)
    w[0] = b_diag[0]
    c_prime[0] = c_diag[0] / w[0]
    for i in range(1, n_z):
        w[i] = b_diag[i] - a_diag[i] * c_prime[i - 1]
        if abs(w[i]) < 1e-30:
            w[i] = 1e-30
        c_prime[i] = c_diag[i] / w[i]

    # Surface forcing
    T_surface = surface_forcing(t_total, config)

    # Initialize
    T_current = np.full(n_z, config.T_mean)

    # Storage (only after spinup)
    n_t_out = config.n_sols * config.dt_per_sol
    T_out = np.zeros((n_t_out, n_z))

    # Preallocate work arrays
    rhs = np.zeros(n_z)
    d_prime = np.zeros(n_z)
    x = np.zeros(n_z)

    for n in range(n_t_total):
        # Build RHS: (I - 0.5*A) T^n + BC — vectorized
        rhs[0] = T_surface[n]
        rhs[1:-1] = (T_current[1:-1]
                     + 0.5 * r[:-1] * (T_current[:-2] - T_current[1:-1])
                     + 0.5 * r[1:] * (T_current[2:] - T_current[1:-1]))
        rhs[-1] = T_current[-1] + 0.5 * r[-1] * (T_current[-2] - T_current[-1])

        # Thomas solve — forward sweep (uses precomputed w, c_prime)
        d_prime[0] = rhs[0] / w[0]
        for i in range(1, n_z):
            d_prime[i] = (rhs[i] - a_diag[i] * d_prime[i - 1]) / w[i]

        # Back substitution
        x[-1] = d_prime[-1]
        for i in range(n_z - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        T_current[:] = x

        # Store output (after spinup)
        out_idx = n - n_spinup * config.dt_per_sol
        if out_idx >= 0:
            T_out[out_idx] = T_current

    t_out = np.arange(n_t_out) * dt
    return T_out, z, t_out


def generate_synthetic_dataset(
    k_profile_fn=None,
    config: SyntheticConfig | None = None,
    n_surface_obs: int = 3000,
    n_colloc: int = 8000,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic dataset for PINN validation.

    Returns dict with dual time coordinates:
        - z, t, T_full, k_true: ground truth
        - t_obs, T_obs: surface observations (physical time + temperature)
        - t_obs_diurnal, t_obs_seasonal: dual phase coordinates for obs
        - z_colloc, t_colloc_diurnal, t_colloc_seasonal: collocation points
    """
    if config is None:
        config = SyntheticConfig()

    z = np.linspace(0, config.z_max, config.n_z)

    if k_profile_fn is None:
        k_profile_fn = two_layer_k_profile
    k_true = k_profile_fn(z)

    logger.info("Solving heat equation (n_z=%d, n_sols=%d, spinup=%d, dt=%d/sol)...",
                config.n_z, config.n_sols, config.spinup_sols, config.dt_per_sol)
    T_full, z, t = solve_heat_equation(k_true, config)
    logger.info("Solution shape: %s, T range: %.1f - %.1f K",
                T_full.shape, T_full.min(), T_full.max())

    # Sample surface observations (uniform over full time range)
    rng = np.random.default_rng(seed)
    n_t = len(t)
    obs_idx = rng.choice(n_t, size=min(n_surface_obs, n_t), replace=False)
    obs_idx.sort()
    t_obs = t[obs_idx]
    T_obs = T_full[obs_idx, 0]  # surface temperature

    # Compute dual time coordinates for observations
    t_obs_diurnal = (t_obs % MARS_SOL) / MARS_SOL       # diurnal phase [0, 1)
    t_obs_seasonal = t_obs / MARS_YEAR_SEC                # seasonal phase [0, ~1]

    # Generate collocation points for physics loss
    # Non-uniform z: beta(2,5) biased toward surface where gradients are strongest
    z_colloc_raw = rng.beta(2, 5, n_colloc)  # ∈ [0,1], concentrated near 0
    z_colloc = (z_colloc_raw * config.z_max).astype(np.float32)

    # Independent diurnal and seasonal phases for collocation
    t_d_colloc = rng.uniform(0, 1, n_colloc).astype(np.float32)
    n_years = config.n_sols / MARS_YEAR_SOLS
    t_s_colloc = rng.uniform(0, n_years, n_colloc).astype(np.float32)

    TI_upper = np.sqrt(k_true[0] * config.rho * config.c_p)
    TI_lower = np.sqrt(k_true[-1] * config.rho * config.c_p)
    logger.info("Ground truth: TI_upper=%.0f, TI_lower=%.0f tiu",
                TI_upper, TI_lower)
    logger.info("Surface obs: %d points, T range: %.1f - %.1f K",
                len(T_obs), T_obs.min(), T_obs.max())
    logger.info("Seasonal coverage: %.2f - %.2f Mars years",
                t_obs_seasonal.min(), t_obs_seasonal.max())

    return {
        "z": z.astype(np.float32),
        "t": t.astype(np.float32),
        "T_full": T_full.astype(np.float32),
        "k_true": k_true.astype(np.float32),
        "t_obs": t_obs.astype(np.float32),
        "T_obs": T_obs.astype(np.float32),
        "t_obs_diurnal": t_obs_diurnal.astype(np.float32),
        "t_obs_seasonal": t_obs_seasonal.astype(np.float32),
        "z_colloc": z_colloc,
        "t_colloc_diurnal": t_d_colloc,
        "t_colloc_seasonal": t_s_colloc,
        "config": config,
    }
