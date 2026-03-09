"""
Synthetic data generation for PINN validation (Phase 1).

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
MARS_SOL = 88_775.0     # seconds per sol
MARS_YEAR_SOLS = 668.6  # sols per Mars year
RHO = 1500.0            # soil density [kg/m³]
C_P = 627.9             # soil heat capacity [J/kg/K]
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Domain
    z_max: float = 3.0          # maximum depth [m]
    n_z: int = 200              # spatial grid points
    n_sols: int = 20            # number of sols to simulate
    dt_per_sol: int = 144       # time steps per sol (10 min each)

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
    k_upper: float = 0.02,     # dust-like [W/m/K] → TI ≈ 137 tiu
    k_lower: float = 2.0,      # ice-like  [W/m/K] → TI ≈ 1372 tiu
    boundary_depth: float = 0.5,  # layer boundary [m]
    transition_width: float = 0.05,  # smooth transition [m]
) -> np.ndarray:
    """
    Two-layer k(z) profile with smooth transition.
    Upper layer: low thermal conductivity (dust/sand).
    Lower layer: high thermal conductivity (ice-cemented soil).
    """
    # Smooth step function (sigmoid)
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
    omega_s = 2.0 * np.pi / (MARS_YEAR_SOLS * MARS_SOL)

    T = (config.T_mean
         + config.T_amp_diurnal * np.sin(omega_d * t)
         + config.T_amp_seasonal * np.sin(omega_s * t))
    return T


def solve_heat_equation(
    k_z: np.ndarray,
    config: SyntheticConfig,
    n_spinup_sols: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 1D heat diffusion with Crank-Nicolson scheme.

        ρc ∂T/∂t = ∂/∂z(k(z) ∂T/∂z)

    Args:
        k_z: thermal conductivity at each depth node [W/m/K]
        config: simulation configuration
        n_spinup_sols: initial sols discarded for spinup

    Returns:
        T: temperature array (n_t, n_z) [K]
        z: depth array (n_z,) [m]
        t: time array (n_t,) [s]
    """
    n_z = config.n_z
    dz = config.z_max / (n_z - 1)
    z = np.linspace(0, config.z_max, n_z)

    total_sols = config.n_sols + n_spinup_sols
    n_t_total = total_sols * config.dt_per_sol
    dt = MARS_SOL / config.dt_per_sol
    t_total = np.arange(n_t_total) * dt

    rho_c = config.rho * config.c_p

    # Diffusivity at each node
    alpha = k_z / rho_c  # [m²/s]

    # Interface conductivities (harmonic mean)
    k_half = np.zeros(n_z - 1)
    for i in range(n_z - 1):
        k_half[i] = 2.0 * k_z[i] * k_z[i + 1] / (k_z[i] + k_z[i + 1])

    # Courant numbers for Crank-Nicolson
    r = np.zeros(n_z - 1)
    for i in range(n_z - 1):
        r[i] = k_half[i] * dt / (rho_c * dz * dz)

    # Build tridiagonal matrices for Crank-Nicolson
    # (I + 0.5*A) T^{n+1} = (I - 0.5*A) T^n
    # where A is the diffusion operator

    # Initialize
    T_current = np.full(n_z, config.T_mean)
    T_surface = surface_forcing(t_total, config)

    # Storage for output (only after spinup)
    n_t_out = config.n_sols * config.dt_per_sol
    T_out = np.zeros((n_t_out, n_z))

    for n in range(n_t_total):
        # Build RHS: (I - 0.5*A) T^n + boundary terms
        rhs = np.zeros(n_z)
        for i in range(1, n_z - 1):
            rhs[i] = (T_current[i]
                       + 0.5 * r[i - 1] * (T_current[i - 1] - T_current[i])
                       + 0.5 * r[i] * (T_current[i + 1] - T_current[i]))

        # Boundary conditions
        # Top: Dirichlet (prescribed surface temperature)
        rhs[0] = T_surface[n]
        # Bottom: Neumann (zero flux)
        rhs[-1] = T_current[-1] + 0.5 * r[-2] * (T_current[-2] - T_current[-1])

        # Build and solve tridiagonal system
        # (I + 0.5*A) T^{n+1} = rhs
        a_diag = np.zeros(n_z)  # lower diagonal
        b_diag = np.zeros(n_z)  # main diagonal
        c_diag = np.zeros(n_z)  # upper diagonal

        b_diag[0] = 1.0  # Dirichlet BC
        for i in range(1, n_z - 1):
            a_diag[i] = -0.5 * r[i - 1]
            b_diag[i] = 1.0 + 0.5 * (r[i - 1] + r[i])
            c_diag[i] = -0.5 * r[i]
        # Bottom: zero flux
        a_diag[-1] = -0.5 * r[-2]
        b_diag[-1] = 1.0 + 0.5 * r[-2]

        # Thomas algorithm (tridiagonal solve)
        T_new = _thomas_solve(a_diag, b_diag, c_diag, rhs)
        T_current = T_new

        # Store output (after spinup)
        out_idx = n - n_spinup_sols * config.dt_per_sol
        if out_idx >= 0:
            T_out[out_idx] = T_current

    t_out = np.arange(n_t_out) * dt
    return T_out, z, t_out


def _thomas_solve(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Tridiagonal matrix solver (Thomas algorithm)."""
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


def generate_synthetic_dataset(
    k_profile_fn=None,
    config: SyntheticConfig | None = None,
    n_surface_obs: int = 500,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic dataset for PINN validation.

    Returns dict with:
        - z: depth grid (n_z,)
        - t: time grid (n_t,)
        - T_full: full temperature field (n_t, n_z)
        - k_true: ground truth k(z) (n_z,)
        - t_obs: observation times (n_obs,)
        - T_obs: observed surface temperatures (n_obs,)  [= T(z=0, t_obs)]
        - z_colloc: collocation points for physics loss (n_colloc, 2)
    """
    if config is None:
        config = SyntheticConfig()

    z = np.linspace(0, config.z_max, config.n_z)

    if k_profile_fn is None:
        k_profile_fn = two_layer_k_profile
    k_true = k_profile_fn(z)

    logger.info("Solving heat equation (n_z=%d, n_sols=%d, dt=%d/sol)...",
                config.n_z, config.n_sols, config.dt_per_sol)
    T_full, z, t = solve_heat_equation(k_true, config)
    logger.info("Solution shape: %s, T range: %.1f - %.1f K",
                T_full.shape, T_full.min(), T_full.max())

    # Sample surface observations
    rng = np.random.default_rng(seed)
    n_t = len(t)
    obs_idx = rng.choice(n_t, size=min(n_surface_obs, n_t), replace=False)
    obs_idx.sort()
    t_obs = t[obs_idx]
    T_obs = T_full[obs_idx, 0]  # surface temperature

    # Generate collocation points for physics loss
    n_colloc = 5000
    z_colloc = rng.uniform(0, config.z_max, n_colloc).astype(np.float32)
    t_colloc = rng.uniform(0, t[-1], n_colloc).astype(np.float32)

    TI_upper = np.sqrt(k_true[0] * config.rho * config.c_p)
    TI_lower = np.sqrt(k_true[-1] * config.rho * config.c_p)
    logger.info("Ground truth: TI_upper=%.0f, TI_lower=%.0f tiu",
                TI_upper, TI_lower)

    return {
        "z": z.astype(np.float32),
        "t": t.astype(np.float32),
        "T_full": T_full.astype(np.float32),
        "k_true": k_true.astype(np.float32),
        "t_obs": t_obs.astype(np.float32),
        "T_obs": T_obs.astype(np.float32),
        "z_colloc": z_colloc,
        "t_colloc": t_colloc,
        "config": config,
    }
