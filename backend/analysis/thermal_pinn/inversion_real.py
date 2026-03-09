"""
Differentiable FDM thermal inversion with energy-balance surface BC.

For real THEMIS observations: only surface brightness temperature is available.
The energy-balance BC makes T_surface a MODEL PREDICTION that depends on k(z),
enabling gradient-based inversion from surface-only data.

Surface BC physics:
    ρc·(dz/2)·dT₀/dt = Q_solar(t) − εσT₀⁴ + k_h·(T₁ − T₀)/dz

This replaces the Dirichlet BC (T₀ = prescribed) used in synthetic validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from backend.analysis.thermal_pinn.mars_solar import (
    surface_solar_flux_timeseries,
    STEFAN_BOLTZMANN,
    MARS_SOL,
    MARS_YEAR_SEC,
    DEFAULT_ALBEDO,
    DEFAULT_EMISSIVITY,
    DEFAULT_DUST_TAU,
)
from backend.analysis.thermal_pinn.pinn_model import (
    ParametricConductivity,
    RHO_CP,
)

logger = logging.getLogger(__name__)

RHO = 1500.0
C_P = 627.9


# ── Configuration ───────────────────────────────────────────────

@dataclass
class EBBCConfig:
    """Config for energy-balance boundary condition FDM inversion."""
    nz: int = 40
    z_max: float = 3.0
    dt_per_sol: int = 24          # 24 steps/sol = 1-hour resolution
    spinup_sols: int = 200        # thermal equilibration
    sim_sols: int = 200           # output simulation period

    # Location & season
    latitude: float = 45.0        # degrees N (Arcadia Planitia)
    Ls_start: float = 0.0         # starting solar longitude

    # Surface properties
    albedo: float = DEFAULT_ALBEDO
    emissivity: float = DEFAULT_EMISSIVITY
    dust_tau: float = DEFAULT_DUST_TAU
    T_init: float = 210.0         # initial temperature (K)

    # Optimizer
    lr: float = 0.02
    n_steps: int = 200
    scheduler_step: int = 80
    scheduler_gamma: float = 0.5

    # k(z) bounds
    k_min_clamp: float = 0.001
    k_max_clamp: float = 10.0


# ── Numpy forward model (truth generation) ──────────────────────

def forward_ebbc_numpy(k_z, cfg: EBBCConfig):
    """
    Forward FDM with energy-balance surface BC. Pure numpy, for truth generation.

    Args:
        k_z: thermal conductivity profile (nz,) array
        cfg: EBBCConfig

    Returns:
        T_surf: (n_sim_steps,) surface temperature timeseries (K)
        T_full: (n_sim_steps, nz) full temperature field
        z: (nz,) depth grid
        Ls_out: (n_sim_steps,) Ls values for output phase
        lt_out: (n_sim_steps,) local time values for output phase
    """
    nz = cfg.nz
    dz = cfg.z_max / (nz - 1)
    dt = MARS_SOL / cfg.dt_per_sol
    n_spinup = cfg.spinup_sols * cfg.dt_per_sol
    n_sim = cfg.sim_sols * cfg.dt_per_sol
    n_total = n_spinup + n_sim

    # Solar forcing for all steps
    Q_all, Ls_all, lt_all = surface_solar_flux_timeseries(
        cfg.latitude, cfg.Ls_start, n_total, dt,
        albedo=cfg.albedo, tau=cfg.dust_tau,
    )

    # Harmonic mean conductivities at cell faces
    kh = 2 * k_z[:-1] * k_z[1:] / (k_z[:-1] + k_z[1:])
    r = kh * dt / (RHO_CP * dz**2)

    # Tridiagonal coefficients (interior + bottom, surface updated per step)
    a_diag = np.zeros(nz)    # lower
    b_diag = np.zeros(nz)    # main
    c_diag = np.zeros(nz)    # upper

    for i in range(1, nz - 1):
        a_diag[i] = -r[i - 1]
        b_diag[i] = 1.0 + r[i - 1] + r[i]
        c_diag[i] = -r[i]

    # Bottom BC (zero flux, half-cell)
    a_diag[-1] = -r[-1]
    b_diag[-1] = 1.0 + r[-1]

    # Surface energy-balance constants
    eps_sig = cfg.emissivity * STEFAN_BOLTZMANN
    norm = dt / (RHO_CP * dz / 2)           # normalization for half-cell
    r_surf = kh[0] * dt / (RHO_CP * (dz / 2) * dz)

    # Initialize
    T = np.full(nz, float(cfg.T_init))
    T_surf_out = np.zeros(n_sim)
    T_full_out = np.zeros((n_sim, nz))

    for n in range(n_total):
        Q = Q_all[n]
        T0_prev = T[0]

        # Update surface row
        f_rad = 4 * eps_sig * T0_prev**3 * norm
        b_diag[0] = 1.0 + f_rad + r_surf
        c_diag[0] = -r_surf

        f_solar = Q * norm
        f_rad_const = 3 * eps_sig * T0_prev**4 * norm

        # RHS
        rhs = T.copy()
        rhs[0] += f_solar + f_rad_const

        # Thomas algorithm (tridiagonal solve)
        cp = np.zeros(nz)
        dp = np.zeros(nz)
        cp[0] = c_diag[0] / b_diag[0]
        dp[0] = rhs[0] / b_diag[0]
        for i in range(1, nz):
            denom = b_diag[i] - a_diag[i] * cp[i - 1]
            if abs(denom) < 1e-30:
                denom = 1e-30
            cp[i] = c_diag[i] / denom
            dp[i] = (rhs[i] - a_diag[i] * dp[i - 1]) / denom

        T[-1] = dp[-1]
        for i in range(nz - 2, -1, -1):
            T[i] = dp[i] - cp[i] * T[i + 1]

        # Store output
        oi = n - n_spinup
        if oi >= 0:
            T_surf_out[oi] = T[0]
            T_full_out[oi] = T.copy()

    z = np.linspace(0, cfg.z_max, nz)
    return (T_surf_out, T_full_out, z,
            Ls_all[n_spinup:], lt_all[n_spinup:])


# ── Observation mapping ─────────────────────────────────────────

def map_observations_to_steps(observations, cfg: EBBCConfig):
    """
    Map THEMIS observations to FDM output-phase timestep indices.

    Each observation has: local_time (hr), solar_lon (Ls deg), bt_kelvin (K).
    Returns list of (output_step_index, bt_kelvin) tuples, sorted by step.

    Mapping approach:
        1. From Ls: determine which sol (day number) this observation falls on
        2. From local_time: determine the step within that sol
        3. Combine: output_step = sol_number * dt_per_sol + step_in_sol

    This works for both synthetic (where Ls encodes full time) and real THEMIS
    (where Ls and local_time are independent metadata fields).
    """
    n_sim = cfg.sim_sols * cfg.dt_per_sol
    Ls_output_start = cfg.Ls_start + cfg.spinup_sols / 668.6 * 360.0
    Ls_output_end = Ls_output_start + cfg.sim_sols / 668.6 * 360.0
    ls_per_sol = 360.0 / 668.6  # ~0.5385° per sol
    mapped = []

    for obs in observations:
        dLs = (obs["solar_lon"] - Ls_output_start) % 360.0
        # Skip if outside simulation window
        Ls_span = cfg.sim_sols * ls_per_sol
        if dLs > Ls_span + ls_per_sol:
            continue

        # Sol number from Ls (which sol does this observation belong to?)
        sol_number = int(dLs / ls_per_sol)

        # Step within sol from local_time
        step_in_sol = int(round(obs["local_time"] / 24.0 * cfg.dt_per_sol))
        step_in_sol = step_in_sol % cfg.dt_per_sol  # wrap 24:00 → 0:00

        output_step = sol_number * cfg.dt_per_sol + step_in_sol

        if 0 <= output_step < n_sim:
            mapped.append((output_step, obs["bt_kelvin"]))

    mapped.sort(key=lambda x: x[0])
    logger.info("Mapped %d / %d observations to output window (Ls %.0f\u2013%.0f)",
                len(mapped), len(observations), Ls_output_start, Ls_output_end)
    return mapped


# ── PyTorch differentiable inversion ────────────────────────────

def run_inversion_ebbc(obs_mapped, z_np, cfg: EBBCConfig, k_model=None):
    """
    Differentiable FDM inversion with energy-balance surface BC.

    Args:
        obs_mapped: list of (output_step_index, bt_kelvin) from map_observations_to_steps
        z_np: (nz,) depth grid
        cfg: EBBCConfig
        k_model: optional pre-initialized ParametricConductivity

    Returns:
        (k_model, history_dict)
    """
    nz = cfg.nz
    dz = cfg.z_max / (nz - 1)
    dt = MARS_SOL / cfg.dt_per_sol
    n_spinup = cfg.spinup_sols * cfg.dt_per_sol
    n_sim = cfg.sim_sols * cfg.dt_per_sol
    n_total = n_spinup + n_sim

    # Pre-compute solar forcing
    Q_all_np, _, _ = surface_solar_flux_timeseries(
        cfg.latitude, cfg.Ls_start, n_total, dt,
        albedo=cfg.albedo, tau=cfg.dust_tau,
    )
    Q_all = torch.tensor(Q_all_np, dtype=torch.float64)

    # Observation targets
    obs_steps = [s for s, _ in obs_mapped]
    T_obs = torch.tensor([bt for _, bt in obs_mapped], dtype=torch.float64)

    z_t = torch.tensor(z_np, dtype=torch.float64)

    if k_model is None:
        k_model = ParametricConductivity((cfg.k_min_clamp, cfg.k_max_clamp))

    optimizer = torch.optim.Adam(k_model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg.scheduler_step, cfg.scheduler_gamma)

    eps_sig = cfg.emissivity * STEFAN_BOLTZMANN
    norm_val = dt / (RHO_CP * dz / 2)

    hist = {"loss": [], "k_upper": [], "k_lower": [], "boundary": [], "width": []}

    for step in range(cfg.n_steps):
        optimizer.zero_grad()

        k_nodes = k_model(z_t)
        kh = 2 * k_nodes[:-1] * k_nodes[1:] / (k_nodes[:-1] + k_nodes[1:])
        r = kh * dt / (RHO_CP * dz**2)
        r_surf = kh[0] * dt / (RHO_CP * (dz / 2) * dz)

        # Build interior + bottom matrix (surface row filled per timestep)
        A_base = torch.zeros(nz, nz, dtype=torch.float64)
        for i in range(1, nz - 1):
            A_base[i, i - 1] = -r[i - 1]
            A_base[i, i] = 1.0 + r[i - 1] + r[i]
            A_base[i, i + 1] = -r[i]
        A_base[-1, -2] = -r[-1]
        A_base[-1, -1] = 1.0 + r[-1]

        T = torch.full((nz,), cfg.T_init, dtype=torch.float64)

        # ── Spinup (no gradient) ──
        with torch.no_grad():
            A_spin = A_base.detach().clone()
            for n in range(n_spinup):
                T0p = T[0].item()
                Q = Q_all_np[n]

                f_rad = 4 * eps_sig * T0p**3 * norm_val
                f_solar = Q * norm_val
                f_rad_const = 3 * eps_sig * T0p**4 * norm_val

                A_spin[0, 0] = 1.0 + f_rad + r_surf.item()
                A_spin[0, 1] = -r_surf.item()

                rhs = T.clone()
                rhs[0] = rhs[0] + f_solar + f_rad_const

                T = torch.linalg.solve(A_spin, rhs)

        # ── Output phase (gradient tracked) ──
        T_predictions = []
        obs_idx = 0

        for n_out in range(n_sim):
            n_abs = n_spinup + n_out
            T0p = T[0].detach()  # linearization uses previous-step T (detached for coefficient)
            Q = Q_all[n_abs]

            f_rad = 4 * eps_sig * T0p**3 * norm_val
            f_solar = Q * norm_val
            f_rad_const = 3 * eps_sig * T0p**4 * norm_val

            A = A_base.clone()
            A[0, 0] = 1.0 + f_rad + r_surf
            A[0, 1] = -r_surf

            rhs = T.clone()
            rhs[0] = rhs[0] + f_solar + f_rad_const

            T = torch.linalg.solve(A, rhs)

            # Collect if this is an observation step
            if obs_idx < len(obs_steps) and n_out == obs_steps[obs_idx]:
                T_predictions.append(T[0])
                obs_idx += 1

        if not T_predictions:
            logger.warning("Step %d: no predictions collected", step + 1)
            continue

        T_pred = torch.stack(T_predictions)
        loss = ((T_pred - T_obs[:len(T_predictions)]) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        kp = k_model.get_params(cfg.z_max)
        hist["loss"].append(loss.item())
        hist["k_upper"].append(kp["k_upper"])
        hist["k_lower"].append(kp["k_lower"])
        hist["boundary"].append(kp["boundary"])
        hist["width"].append(kp["width"])

        if (step + 1) % 20 == 0 or step == 0:
            logger.info(
                "Step %3d: loss=%.4f  k_u=%.4f k_l=%.4f bd=%.3fm w=%.4fm  "
                "(matched %d obs)",
                step + 1, loss.item(), kp["k_upper"], kp["k_lower"],
                kp["boundary"], kp["width"], len(T_predictions))

    return k_model, hist
