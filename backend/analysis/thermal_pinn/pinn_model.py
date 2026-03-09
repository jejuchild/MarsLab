"""
Differentiable FDM thermal inversion — V5.

Replaces PINN (V1-V4 all failed) with a differentiable forward model:
- Parametric k(z) = k_upper + (k_lower-k_upper)*sigmoid((z-d)/w), 4 learnable scalars
- FDM time-stepping in PyTorch (implicit Euler), torch.linalg.solve for AD
- Gradient dL/dk is O(1) — 6 orders of magnitude stronger than PINN's O(1e-6)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import torch

logger = logging.getLogger(__name__)

RHO = 1500.0; C_P = 627.9; RHO_CP = RHO * C_P
MARS_SOL = 88_775.0; MARS_YEAR_SOLS = 668.6
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL


@dataclass
class InversionConfig:
    nz: int = 40
    z_max: float = 3.0
    n_sols: int = 50
    dt_per_sol: int = 48
    spinup_sols: int = 50
    T_mean: float = 210.0
    T_amp_diurnal: float = 40.0
    T_amp_seasonal: float = 20.0
    k_min_clamp: float = 0.001
    k_max_clamp: float = 10.0
    lr: float = 0.03
    n_steps: int = 300
    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    obs_subsample: int = 6


class ParametricConductivity(torch.nn.Module):
    """k(z) = k_upper + (k_lower - k_upper) * sigmoid((z - d) / w), 4 learnable scalars."""
    def __init__(self, k_clamp=(0.001, 10.0)):
        super().__init__()
        self.k_min, self.k_max = k_clamp
        self.log_k_upper = torch.nn.Parameter(torch.tensor(np.log(0.1), dtype=torch.float64))
        self.log_k_lower = torch.nn.Parameter(torch.tensor(np.log(0.3), dtype=torch.float64))
        self.raw_d = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_w = torch.nn.Parameter(torch.tensor(np.log(0.03), dtype=torch.float64))

    def forward(self, z):
        ku = torch.exp(self.log_k_upper).clamp(self.k_min, self.k_max)
        kl = torch.exp(self.log_k_lower).clamp(self.k_min, self.k_max)
        d = torch.sigmoid(self.raw_d) * z.max()
        w = torch.exp(self.log_w).clamp(0.005, 1.0) * z.max()
        return ku + (kl - ku) * torch.sigmoid((z - d) / w)

    def get_params(self, z_max):
        with torch.no_grad():
            ku = torch.exp(self.log_k_upper).clamp(self.k_min, self.k_max).item()
            kl = torch.exp(self.log_k_lower).clamp(self.k_min, self.k_max).item()
            d = torch.sigmoid(self.raw_d).item() * z_max
            w = torch.exp(self.log_w).clamp(0.005, 1.0).item() * z_max
        return {"k_upper": ku, "k_lower": kl, "boundary": d, "width": w}


def _build_implicit_matrix(k_nodes, dz, dt):
    nz = len(k_nodes)
    kh = 2 * k_nodes[:-1] * k_nodes[1:] / (k_nodes[:-1] + k_nodes[1:])
    r = kh * dt / (RHO_CP * dz**2)
    A = torch.zeros(nz, nz, dtype=torch.float64)
    A[0, 0] = 1.0
    for i in range(1, nz - 1):
        A[i, i-1] = -r[i-1]
        A[i, i] = 1.0 + r[i-1] + r[i]
        A[i, i+1] = -r[i]
    A[-1, -2] = -r[-1]
    A[-1, -1] = 1.0 + r[-1]
    return A


def _surface_forcing(t, T_mean, T_amp_d, T_amp_s):
    wd = 2.0 * np.pi / MARS_SOL
    ws = 2.0 * np.pi / MARS_YEAR_SEC
    return T_mean + T_amp_d * np.sin(wd * t) + T_amp_s * np.sin(ws * t)


def run_inversion(T_obs, obs_time_indices, obs_depth_indices, z_np, cfg, k_model=None):
    """
    Differentiable FDM inversion: recover k(z) from subsurface temperature observations.

    Args:
        T_obs: (n_obs, n_depths) tensor — observed temperatures
        obs_time_indices: output timestep indices where observations exist
        obs_depth_indices: depth node indices where observations exist
        z_np: depth grid (numpy array)
        cfg: InversionConfig
        k_model: optional pre-initialized ParametricConductivity
    Returns:
        (k_model, history_dict)
    """
    nz = cfg.nz
    dz = cfg.z_max / (nz - 1)
    dt = MARS_SOL / cfg.dt_per_sol
    n_total = (cfg.n_sols + cfg.spinup_sols) * cfg.dt_per_sol
    spinup_steps = cfg.spinup_sols * cfg.dt_per_sol

    t_all = np.arange(n_total) * dt
    T_surf_np = _surface_forcing(t_all, cfg.T_mean, cfg.T_amp_diurnal, cfg.T_amp_seasonal)
    T_surf = torch.tensor(T_surf_np, dtype=torch.float64)
    z_t = torch.tensor(z_np, dtype=torch.float64)

    if k_model is None:
        k_model = ParametricConductivity((cfg.k_min_clamp, cfg.k_max_clamp))

    optimizer = torch.optim.Adam(k_model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler_step, cfg.scheduler_gamma)

    hist = {"loss": [], "k_upper": [], "k_lower": [], "boundary": [], "width": []}

    for step in range(cfg.n_steps):
        optimizer.zero_grad()

        k_nodes = k_model(z_t)
        A = _build_implicit_matrix(k_nodes, dz, dt)

        # Spinup without gradient tracking
        T = torch.full((nz,), cfg.T_mean, dtype=torch.float64)
        with torch.no_grad():
            A_det = A.detach()
            for n in range(spinup_steps):
                rhs = T.clone()
                rhs[0] = T_surf[n].item()
                T = torch.linalg.solve(A_det, rhs)

        # Output phase: tracked through A(k)
        T_predictions = []
        obs_counter = 0
        for n in range(spinup_steps, n_total):
            rhs = T.clone()
            rhs[0] = T_surf[n]
            T = torch.linalg.solve(A, rhs)
            output_step = n - spinup_steps
            if obs_counter < len(obs_time_indices) and output_step == obs_time_indices[obs_counter]:
                T_predictions.append(T[obs_depth_indices])
                obs_counter += 1

        loss = ((torch.stack(T_predictions) - T_obs) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        kp = k_model.get_params(cfg.z_max)
        hist["loss"].append(loss.item())
        hist["k_upper"].append(kp["k_upper"])
        hist["k_lower"].append(kp["k_lower"])
        hist["boundary"].append(kp["boundary"])
        hist["width"].append(kp["width"])

        if (step + 1) % 50 == 0 or step == 0:
            logger.info(
                "Step %3d: loss=%.6f  k_u=%.4f k_l=%.4f bd=%.3fm w=%.4fm",
                step + 1, loss.item(), kp["k_upper"], kp["k_lower"],
                kp["boundary"], kp["width"])

    return k_model, hist


def evaluate_inversion(k_model, z_np, k_true_np, z_max):
    z_t = torch.tensor(z_np, dtype=torch.float64)
    with torch.no_grad():
        k_pred = k_model(z_t).numpy()
    k_rmse = float(np.sqrt(np.mean((k_pred - k_true_np) ** 2)))
    TI_pred = np.sqrt(np.clip(k_pred, 0, None) * RHO * C_P)
    TI_true = np.sqrt(k_true_np * RHO * C_P)
    kp = k_model.get_params(z_max)

    bi = np.argmax(np.abs(np.gradient(k_pred)))
    bd = z_np[bi]

    logger.info("=" * 50)
    logger.info("k_RMSE=%.4f  boundary=%.3fm", k_rmse, bd)
    logger.info("TI: pred %.0f-%.0f  true %.0f-%.0f",
                TI_pred.min(), TI_pred.max(), TI_true.min(), TI_true.max())
    logger.info("Recovered: k_upper=%.4f(0.02) k_lower=%.4f(2.0) "
                "boundary=%.3fm(0.5) width=%.4fm(0.05)",
                kp["k_upper"], kp["k_lower"], kp["boundary"], kp["width"])
    logger.info("=" * 50)

    return {
        "z": z_np, "k_pred": k_pred, "k_true": k_true_np,
        "TI_pred": TI_pred, "TI_true": TI_true,
        "k_rmse": k_rmse, "boundary_depth": bd,
        "k_params": kp,
    }
