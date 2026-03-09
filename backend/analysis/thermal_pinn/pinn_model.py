"""
Physics-Informed Neural Network (PINN) for Mars thermal inversion.

Architecture:
    NN₁(z, t) → T(z, t)      Temperature field
    NN₂(z)    → k(z)          Thermal conductivity profile

Physics loss via finite-difference approximation (CPU-optimized):
    ρc · ∂T/∂t ≈ ∂/∂z(k(z) · ∂T/∂z)

Data loss from surface observations:
    L_data = |NN₁(z=0, t_obs) - T_observed|²

Key insight: PINN only needs PDE ① (heat diffusion) because THEMIS
observations already encode effects of ②③④⑤ (surface energy balance,
atmospheric RT, orbital mechanics, CO₂ frost).

Performance note: Uses numerical finite differences instead of torch.autograd
for PDE residual computation — 10x faster on CPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Mars constants
RHO = 1500.0      # soil density [kg/m³]
C_P = 627.9       # heat capacity [J/kg/K]
RHO_CP = RHO * C_P  # volumetric heat capacity
MARS_SOL = 88_775.0  # seconds


@dataclass
class PINNConfig:
    """PINN training configuration."""
    # Network architecture
    n_hidden_T: int = 4         # hidden layers for temperature net
    n_neurons_T: int = 64       # neurons per layer
    n_hidden_k: int = 3         # hidden layers for conductivity net
    n_neurons_k: int = 32       # neurons per layer
    n_fourier: int = 16         # Fourier features for T net

    # Physical bounds
    k_min: float = 0.005        # min conductivity [W/m/K] (fine dust)
    k_max: float = 4.0          # max conductivity [W/m/K] (ice)
    T_min: float = 140.0        # min temperature [K] (CO₂ frost)
    T_max: float = 300.0        # max temperature [K]

    # Normalization ranges (set from data)
    z_max: float = 3.0          # depth range [m]
    t_max: float = 1.0          # time range [s] (set from data)

    # Finite difference stencil sizes (normalized units)
    eps_z: float = 0.005        # Δz for FD stencil
    eps_t: float = 0.005        # Δt for FD stencil

    # Loss weights
    w_physics: float = 1.0      # PDE residual weight
    w_data: float = 10.0        # surface observation weight
    w_k_smooth: float = 0.01    # k(z) smoothness regularization

    # Training
    lr: float = 1e-3
    n_epochs: int = 5000
    batch_colloc: int = 2048    # collocation points per batch
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TemperatureNet(nn.Module):
    """
    NN₁: (z, t) → T(z, t)
    Maps normalized (z, t) coordinates to temperature.
    Uses Fourier feature encoding for periodic signal capture.
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 64,
                 n_fourier: int = 16):
        super().__init__()
        self.n_fourier = n_fourier
        input_dim = 2 + 2 * n_fourier

        layers = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)

        self.register_buffer("freqs", torch.randn(2, n_fourier) * 2.0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, 1)
        t = t.view(-1, 1)
        zt = torch.cat([z, t], dim=1)
        proj = zt @ self.freqs
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        x = torch.cat([zt, fourier], dim=1)
        return self.net(x)


class ConductivityNet(nn.Module):
    """
    NN₂: z → k(z)
    Maps normalized depth to thermal conductivity.
    Output bounded to [k_min, k_max] via sigmoid.
    """

    def __init__(self, n_hidden: int = 3, n_neurons: int = 32,
                 k_min: float = 0.005, k_max: float = 4.0):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max

        layers = [nn.Linear(1, n_neurons), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(n_neurons, n_neurons), nn.Tanh()])
        layers.append(nn.Linear(n_neurons, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(-1, 1)
        raw = self.net(z)
        k = self.k_min + (self.k_max - self.k_min) * torch.sigmoid(raw)
        return k


class ThermalPINN(nn.Module):
    """
    Combined PINN for thermal inversion.

    Uses finite-difference stencils for PDE residual (fast on CPU):
        ∂T/∂t ≈ [T(z,t+ε) - T(z,t-ε)] / (2ε)
        ∂/∂z(k·∂T/∂z) ≈ [k(z+ε)·(T(z+2ε,t)-T(z,t)) - k(z-ε)·(T(z,t)-T(z-2ε,t))] / (2ε²)
    """

    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.T_net = TemperatureNet(
            n_hidden=config.n_hidden_T,
            n_neurons=config.n_neurons_T,
            n_fourier=config.n_fourier,
        )
        self.k_net = ConductivityNet(
            n_hidden=config.n_hidden_k,
            n_neurons=config.n_neurons_k,
            k_min=config.k_min,
            k_max=config.k_max,
        )
        self.to(self.device)

    def predict_T(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict temperature T(z, t) in physical units [K]."""
        T_norm = self.T_net(z, t)
        T = self.config.T_min + (self.config.T_max - self.config.T_min) * (
            0.5 * (T_norm + 1.0)
        )
        return T

    def predict_k(self, z: torch.Tensor) -> torch.Tensor:
        """Predict thermal conductivity k(z) in [W/m/K]."""
        return self.k_net(z)

    def physics_loss_fd(
        self, z_c: torch.Tensor, t_c: torch.Tensor
    ) -> torch.Tensor:
        """
        PDE residual via finite differences (no autograd needed).

        ρc · ∂T/∂t = ∂/∂z(k(z) · ∂T/∂z)

        All inputs are in normalized [0,1] coordinates.
        """
        eps_z = self.config.eps_z
        eps_t = self.config.eps_t

        # Clamp to avoid boundary stencil issues
        z_c = z_c.clamp(2 * eps_z, 1.0 - 2 * eps_z)
        t_c = t_c.clamp(eps_t, 1.0 - eps_t)

        # ∂T/∂t via central difference
        T_tp = self.predict_T(z_c, t_c + eps_t)  # T(z, t+ε)
        T_tm = self.predict_T(z_c, t_c - eps_t)  # T(z, t-ε)
        dT_dt = (T_tp - T_tm) / (2.0 * eps_t)    # normalized

        # ∂/∂z(k · ∂T/∂z) via conservative stencil
        # flux at z+ε/2: k(z+ε/2) · [T(z+ε) - T(z)] / ε
        # flux at z-ε/2: k(z-ε/2) · [T(z) - T(z-ε)] / ε
        # divergence: [flux(z+ε/2) - flux(z-ε/2)] / ε
        T_c = self.predict_T(z_c, t_c)
        T_zp = self.predict_T(z_c + eps_z, t_c)
        T_zm = self.predict_T(z_c - eps_z, t_c)

        k_zph = self.predict_k(z_c + 0.5 * eps_z)  # k at z+ε/2
        k_zmh = self.predict_k(z_c - 0.5 * eps_z)  # k at z-ε/2

        flux_p = k_zph * (T_zp - T_c) / eps_z
        flux_m = k_zmh * (T_c - T_zm) / eps_z
        div_flux = (flux_p - flux_m) / eps_z  # ∂/∂z(k·∂T/∂z) in normalized z

        # Convert to physical units:
        # ∂T/∂t_phys = (1/t_max) · dT_dt_norm
        # ∂²/∂z²_phys = (1/z_max²) · div_flux_norm
        scale_t = 1.0 / self.config.t_max
        scale_z = 1.0 / (self.config.z_max ** 2)

        residual = RHO_CP * scale_t * dT_dt - scale_z * div_flux

        # Normalize residual to make it dimensionless
        # Characteristic residual scale: RHO_CP * T_amp * omega_diurnal
        T_amp = 0.5 * (self.config.T_max - self.config.T_min)
        omega = 2.0 * np.pi / MARS_SOL
        char_scale = RHO_CP * T_amp * omega
        residual_norm = residual / max(char_scale, 1e-10)

        return (residual_norm ** 2).mean()

    def data_loss(
        self, t_obs: torch.Tensor, T_obs: torch.Tensor
    ) -> torch.Tensor:
        """Surface observation loss: |T_pred(z=0, t_obs) - T_observed|²"""
        z_surface = torch.zeros_like(t_obs)
        T_pred = self.predict_T(z_surface, t_obs)
        # Normalize by temperature scale
        T_scale = self.config.T_max - self.config.T_min
        return ((T_pred - T_obs.view(-1, 1)) ** 2).mean() / (T_scale ** 2)

    def k_smoothness_loss_fd(self, z_points: torch.Tensor) -> torch.Tensor:
        """k(z) smoothness via FD second derivative."""
        eps = self.config.eps_z
        z = z_points.clamp(eps, 1.0 - eps)
        k_c = self.predict_k(z)
        k_p = self.predict_k(z + eps)
        k_m = self.predict_k(z - eps)
        d2k = (k_p - 2.0 * k_c + k_m) / (eps ** 2)
        k_range = self.config.k_max - self.config.k_min
        return ((d2k / k_range) ** 2).mean()


def train_pinn(
    model: ThermalPINN,
    data: dict,
    config: PINNConfig,
) -> dict:
    """
    Train the PINN model with curriculum learning.

    Phase 1 (0-40%):  High data weight, low physics — learn surface T
    Phase 2 (40-100%): Gradually increase physics weight — learn k(z)
    """
    device = model.device

    z_max = float(data["z"].max())
    t_max = float(data["t"].max()) if "t" in data else float(data["t_obs"].max())
    config.z_max = z_max
    config.t_max = t_max

    # Surface observations
    t_obs_norm = torch.tensor(
        data["t_obs"] / t_max, dtype=torch.float32
    ).to(device)
    T_obs = torch.tensor(data["T_obs"], dtype=torch.float32).to(device)

    # Collocation points
    z_colloc_all = torch.tensor(
        data["z_colloc"] / z_max, dtype=torch.float32
    ).to(device)
    t_colloc_all = torch.tensor(
        data["t_colloc"] / t_max, dtype=torch.float32
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )

    history = {
        "loss_total": [], "loss_physics": [],
        "loss_data": [], "loss_smooth": [],
    }

    rng = np.random.default_rng(0)
    n_epochs = config.n_epochs
    curriculum_switch = int(0.4 * n_epochs)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Curriculum: ramp up physics weight
        if epoch < curriculum_switch:
            # Phase 1: focus on data
            w_phys = config.w_physics * 0.01
            w_data = config.w_data * 5.0
        else:
            # Phase 2: ramp physics from 0.01 to 1.0
            progress = (epoch - curriculum_switch) / max(n_epochs - curriculum_switch, 1)
            w_phys = config.w_physics * (0.01 + 0.99 * progress)
            w_data = config.w_data

        # Sample collocation batch
        idx = rng.choice(
            len(z_colloc_all),
            size=min(config.batch_colloc, len(z_colloc_all)),
            replace=False,
        )
        z_c = z_colloc_all[idx]
        t_c = t_colloc_all[idx]

        # Compute losses
        loss_phys = model.physics_loss_fd(z_c, t_c)
        loss_data = model.data_loss(t_obs_norm, T_obs)

        z_smooth = torch.linspace(0.01, 0.99, 50, device=device)
        loss_smooth = model.k_smoothness_loss_fd(z_smooth)

        loss = w_phys * loss_phys + w_data * loss_data + config.w_k_smooth * loss_smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["loss_total"].append(loss.item())
        history["loss_physics"].append(loss_phys.item())
        history["loss_data"].append(loss_data.item())
        history["loss_smooth"].append(loss_smooth.item())

        if (epoch + 1) % 500 == 0 or epoch == 0:
            logger.info(
                "Epoch %5d/%d | L=%.3e | Phys=%.3e | Data=%.3e | "
                "w_p=%.3f | LR=%.1e",
                epoch + 1, n_epochs,
                loss.item(), loss_phys.item(), loss_data.item(),
                w_phys, optimizer.param_groups[0]["lr"],
            )

    return history


def evaluate_pinn(model: ThermalPINN, data: dict) -> dict:
    """Evaluate trained PINN against ground truth."""
    device = model.device
    config = model.config

    z = data["z"]
    z_norm = torch.tensor(z / config.z_max, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        k_pred = model.predict_k(z_norm).cpu().numpy().flatten()

    k_true = data["k_true"]

    t_obs = data["t_obs"]
    t_norm = torch.tensor(t_obs / config.t_max, dtype=torch.float32).to(device)
    z_zero = torch.zeros_like(t_norm)

    with torch.no_grad():
        T_pred = model.predict_T(z_zero, t_norm).cpu().numpy().flatten()

    T_true = data["T_obs"]

    k_rmse = float(np.sqrt(np.mean((k_pred - k_true) ** 2)))
    T_rmse = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))

    TI_pred = np.sqrt(np.clip(k_pred, 0, None) * RHO * C_P)
    TI_true = np.sqrt(k_true * RHO * C_P)

    logger.info("Evaluation:")
    logger.info("  k RMSE:  %.4f W/m/K", k_rmse)
    logger.info("  T RMSE:  %.2f K", T_rmse)
    logger.info("  TI_pred: %.0f - %.0f tiu", TI_pred.min(), TI_pred.max())
    logger.info("  TI_true: %.0f - %.0f tiu", TI_true.min(), TI_true.max())

    return {
        "z": z,
        "k_pred": k_pred,
        "k_true": k_true,
        "TI_pred": TI_pred,
        "TI_true": TI_true,
        "T_pred_surface": T_pred,
        "T_true_surface": T_true,
        "k_rmse": k_rmse,
        "T_rmse": T_rmse,
    }
