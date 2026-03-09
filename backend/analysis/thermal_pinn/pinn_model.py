"""
Physics-Informed Neural Network (PINN) for Mars thermal inversion.
V2: Dual-coordinate time encoding (diurnal + seasonal phases).

Architecture:
    NN₁(z, t_d, t_s) → T(z, t)      Temperature field
    NN₂(z)           → k(z)          Thermal conductivity profile

Physics loss via finite-difference approximation (CPU-optimized):
    ρc · ∂T/∂t ≈ ∂/∂z(k(z) · ∂T/∂z)

Key improvements over v1:
    - Dual time coordinates: diurnal phase t_d ∈ [0,1] + seasonal phase t_s
      → FD stencil sizes are period-relative, NOT simulation-length-relative
    - Physically motivated Fourier features at known diurnal/seasonal frequencies
    - Reduced smoothness regularization for step-like k(z) recovery
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Mars constants
RHO = 1500.0                  # soil density [kg/m³]
C_P = 627.9                   # heat capacity [J/kg/K]
RHO_CP = RHO * C_P            # volumetric heat capacity
MARS_SOL = 88_775.0           # seconds per sol
MARS_YEAR_SOLS = 668.6        # sols per Mars year
MARS_YEAR_SEC = MARS_YEAR_SOLS * MARS_SOL  # seconds per Mars year


@dataclass
class PINNConfig:
    """PINN training configuration — V2 with dual time coords."""
    # Network architecture
    n_hidden_T: int = 4         # hidden layers for temperature net
    n_neurons_T: int = 64       # neurons per layer
    n_hidden_k: int = 3         # hidden layers for conductivity net
    n_neurons_k: int = 32       # neurons per layer

    # Fourier feature counts
    n_harmonics_diurnal: int = 4    # sin/cos(n·2π·t_d), n=1..4
    n_harmonics_seasonal: int = 3   # sin/cos(n·2π·t_s), n=1..3
    n_depth_modes: int = 4          # sin/cos(n·π·z), n=1..4

    # Physical bounds
    k_min: float = 0.005        # min conductivity [W/m/K] (fine dust)
    k_max: float = 4.0          # max conductivity [W/m/K] (ice)
    T_min: float = 140.0        # min temperature [K]
    T_max: float = 300.0        # max temperature [K]

    # Domain (set from data)
    z_max: float = 3.0          # depth range [m]

    # FD stencil sizes (in phase/normalized coordinates)
    # These are INDEPENDENT of simulation length — key V2 improvement
    eps_z: float = 0.002        # normalized depth: 0.002 × 3m = 0.006m
    eps_d: float = 0.01         # diurnal phase: 0.01 sol ≈ 15 min
    eps_s: float = 0.005        # seasonal phase: 0.005 year ≈ 3.3 sols

    # Loss weights
    w_physics: float = 1.0      # PDE residual weight
    w_data: float = 10.0        # surface observation weight
    w_k_smooth: float = 1e-4    # k(z) smoothness (very low — allow steps)

    # Training
    lr: float = 1e-3
    n_epochs: int = 6000
    batch_colloc: int = 4096    # collocation points per batch
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TemperatureNet(nn.Module):
    """
    NN₁: (z_norm, t_diurnal, t_seasonal) → T

    Physically motivated Fourier features:
      - Diurnal harmonics: sin/cos(n·2π·t_d), n = 1,...,N_d
      - Seasonal harmonics: sin/cos(n·2π·t_s), n = 1,...,N_s
      - Depth modes: sin/cos(n·π·z_norm), n = 1,...,N_z

    Input dim = 1 (z) + 2·N_d + 2·N_s + 2·N_z
    """

    def __init__(self, n_hidden: int = 4, n_neurons: int = 64,
                 n_harmonics_d: int = 4, n_harmonics_s: int = 3,
                 n_depth_modes: int = 4):
        super().__init__()
        self.n_hd = n_harmonics_d
        self.n_hs = n_harmonics_s
        self.n_dz = n_depth_modes

        input_dim = 1 + 2 * n_harmonics_d + 2 * n_harmonics_s + 2 * n_depth_modes

        layers = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
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

    def forward(self, z: torch.Tensor, t_d: torch.Tensor,
                t_s: torch.Tensor) -> torch.Tensor:
        """
        z: normalized depth [0, 1]
        t_d: diurnal phase [0, 1), wraps every sol
        t_s: seasonal phase, ~[0, 1] for 1 Mars year
        """
        z = z.view(-1, 1)
        t_d = t_d.view(-1, 1)
        t_s = t_s.view(-1, 1)

        TWO_PI = 2.0 * np.pi
        PI = np.pi

        features = [z]

        # Diurnal harmonics — captures daily temperature cycle
        for n in range(1, self.n_hd + 1):
            phase = TWO_PI * n * t_d
            features.append(torch.sin(phase))
            features.append(torch.cos(phase))

        # Seasonal harmonics — captures yearly temperature cycle
        for n in range(1, self.n_hs + 1):
            phase = TWO_PI * n * t_s
            features.append(torch.sin(phase))
            features.append(torch.cos(phase))

        # Depth modes — captures exponential decay structure
        for n in range(1, self.n_dz + 1):
            phase = PI * n * z
            features.append(torch.sin(phase))
            features.append(torch.cos(phase))

        x = torch.cat(features, dim=1)
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
    Combined PINN for thermal inversion — V2 dual time coordinates.

    Physics loss uses dual-coordinate FD stencils:
        ∂T/∂t = (∂T/∂t_d)(1/P_sol) + (∂T/∂t_s)(1/P_year)
        ∂/∂z(k·∂T/∂z) via conservative stencil in normalized z
    """

    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        self.T_net = TemperatureNet(
            n_hidden=config.n_hidden_T,
            n_neurons=config.n_neurons_T,
            n_harmonics_d=config.n_harmonics_diurnal,
            n_harmonics_s=config.n_harmonics_seasonal,
            n_depth_modes=config.n_depth_modes,
        )
        self.k_net = ConductivityNet(
            n_hidden=config.n_hidden_k,
            n_neurons=config.n_neurons_k,
            k_min=config.k_min,
            k_max=config.k_max,
        )
        self.to(self.device)

    def predict_T(self, z: torch.Tensor, t_d: torch.Tensor,
                  t_s: torch.Tensor) -> torch.Tensor:
        """Predict temperature T(z, t) in physical units [K]."""
        T_norm = self.T_net(z, t_d, t_s)
        T = self.config.T_min + (self.config.T_max - self.config.T_min) * (
            0.5 * (T_norm + 1.0)
        )
        return T

    def predict_k(self, z: torch.Tensor) -> torch.Tensor:
        """Predict thermal conductivity k(z) in [W/m/K]."""
        return self.k_net(z)

    def physics_loss_fd(
        self, z_c: torch.Tensor, t_d_c: torch.Tensor, t_s_c: torch.Tensor
    ) -> torch.Tensor:
        """
        PDE residual via finite differences with dual time coordinates.

        ρc · ∂T/∂t = ∂/∂z(k(z) · ∂T/∂z)

        ∂T/∂t = (∂T/∂t_d)(1/P_sol) + (∂T/∂t_s)(1/P_year)

        Key V2 advantage: eps_d and eps_s are in phase coordinates,
        so they always correspond to fixed physical time intervals
        regardless of simulation length.
        """
        eps_z = self.config.eps_z
        eps_d = self.config.eps_d
        eps_s = self.config.eps_s

        # Clamp z to avoid boundary stencil issues
        z_c = z_c.clamp(2 * eps_z, 1.0 - 2 * eps_z)

        # --- Temporal derivatives (dual coordinates) ---
        # ∂T/∂t_d via central difference (diurnal)
        T_dp = self.predict_T(z_c, t_d_c + eps_d, t_s_c)
        T_dm = self.predict_T(z_c, t_d_c - eps_d, t_s_c)
        dT_dtd = (T_dp - T_dm) / (2.0 * eps_d)  # K per diurnal-phase-unit

        # ∂T/∂t_s via central difference (seasonal)
        T_sp = self.predict_T(z_c, t_d_c, t_s_c + eps_s)
        T_sm = self.predict_T(z_c, t_d_c, t_s_c - eps_s)
        dT_dts = (T_sp - T_sm) / (2.0 * eps_s)  # K per seasonal-phase-unit

        # Physical ∂T/∂t via chain rule [K/s]:
        #   t_d = t / P_sol      → dt_d/dt = 1/P_sol
        #   t_s = t / P_year     → dt_s/dt = 1/P_year
        #   ∂T/∂t = (∂T/∂t_d)(1/P_sol) + (∂T/∂t_s)(1/P_year)
        dT_dt = dT_dtd / MARS_SOL + dT_dts / MARS_YEAR_SEC

        # --- Spatial derivative ---
        # ∂/∂z(k · ∂T/∂z) via conservative FD stencil (normalized z)
        T_c = self.predict_T(z_c, t_d_c, t_s_c)
        T_zp = self.predict_T(z_c + eps_z, t_d_c, t_s_c)
        T_zm = self.predict_T(z_c - eps_z, t_d_c, t_s_c)

        k_zph = self.predict_k(z_c + 0.5 * eps_z)  # k at z+ε/2
        k_zmh = self.predict_k(z_c - 0.5 * eps_z)  # k at z-ε/2

        flux_p = k_zph * (T_zp - T_c) / eps_z
        flux_m = k_zmh * (T_c - T_zm) / eps_z
        div_flux_norm = (flux_p - flux_m) / eps_z

        # Convert to physical: ∂/∂z_phys = (1/z_max)·∂/∂z_norm
        # → ∂/∂z(k·∂T/∂z) = div_flux_norm / z_max²
        div_flux = div_flux_norm / (self.config.z_max ** 2)

        # PDE residual: ρc · ∂T/∂t - ∂/∂z(k·∂T/∂z) = 0
        residual = RHO_CP * dT_dt - div_flux

        # Normalize by characteristic scale: ρc · T_amp · ω_diurnal
        T_amp = 0.5 * (self.config.T_max - self.config.T_min)
        omega_d = 2.0 * np.pi / MARS_SOL
        char_scale = RHO_CP * T_amp * omega_d

        return ((residual / max(char_scale, 1e-10)) ** 2).mean()

    def data_loss(
        self, t_d_obs: torch.Tensor, t_s_obs: torch.Tensor,
        T_obs: torch.Tensor
    ) -> torch.Tensor:
        """Surface observation loss: |T_pred(z=0, t) - T_observed|²"""
        z_surface = torch.zeros_like(t_d_obs)
        T_pred = self.predict_T(z_surface, t_d_obs, t_s_obs)
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
    Train the PINN model with 3-phase curriculum learning.

    Phase 1 (0-25%):   Data focus — learn surface T accurately
    Phase 2 (25-60%):  Physics ramp — begin constraining interior, learn k(z)
    Phase 3 (60-100%): Physics dominance — fine-tune k(z) recovery
    """
    device = model.device

    z_max = float(data["z"].max())
    config.z_max = z_max

    # Surface observations (dual coordinates)
    t_d_obs = torch.tensor(
        data["t_obs_diurnal"], dtype=torch.float32
    ).to(device)
    t_s_obs = torch.tensor(
        data["t_obs_seasonal"], dtype=torch.float32
    ).to(device)
    T_obs = torch.tensor(data["T_obs"], dtype=torch.float32).to(device)

    # Collocation points (normalized z, phase coordinates)
    z_colloc_all = torch.tensor(
        data["z_colloc"] / z_max, dtype=torch.float32
    ).to(device)
    t_d_colloc_all = torch.tensor(
        data["t_colloc_diurnal"], dtype=torch.float32
    ).to(device)
    t_s_colloc_all = torch.tensor(
        data["t_colloc_seasonal"], dtype=torch.float32
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
    phase1_end = int(0.25 * n_epochs)
    phase2_end = int(0.60 * n_epochs)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # 3-phase curriculum
        if epoch < phase1_end:
            # Phase 1: Data focus — nail the surface T fit
            w_phys = config.w_physics * 0.001
            w_data = config.w_data * 5.0
        elif epoch < phase2_end:
            # Phase 2: Ramp physics — force PDE consistency, begin k(z) learning
            progress = (epoch - phase1_end) / max(phase2_end - phase1_end, 1)
            w_phys = config.w_physics * (0.001 + 4.999 * progress)
            w_data = config.w_data * (5.0 - 3.0 * progress)
        else:
            # Phase 3: Physics dominance — fine-tune k(z)
            w_phys = config.w_physics * 5.0
            w_data = config.w_data * 2.0

        # Sample collocation batch
        idx = rng.choice(
            len(z_colloc_all),
            size=min(config.batch_colloc, len(z_colloc_all)),
            replace=False,
        )
        z_c = z_colloc_all[idx]
        t_d_c = t_d_colloc_all[idx]
        t_s_c = t_s_colloc_all[idx]

        # Compute losses
        loss_phys = model.physics_loss_fd(z_c, t_d_c, t_s_c)
        loss_data = model.data_loss(t_d_obs, t_s_obs, T_obs)

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
            if epoch < phase1_end:
                phase_label = "P1-data"
            elif epoch < phase2_end:
                phase_label = "P2-ramp"
            else:
                phase_label = "P3-phys"

            logger.info(
                "Epoch %5d/%d [%s] | L=%.3e | Phys=%.3e | Data=%.3e | "
                "w_p=%.3f | w_d=%.1f | LR=%.1e",
                epoch + 1, n_epochs, phase_label,
                loss.item(), loss_phys.item(), loss_data.item(),
                w_phys, w_data, optimizer.param_groups[0]["lr"],
            )

            # Quick k(z) diagnostic at key depths
            model.eval()
            with torch.no_grad():
                z_test = torch.tensor(
                    [0.0, 0.05, 0.1, 0.167, 0.333, 0.5, 0.667, 1.0],
                    device=device
                )
                k_test = model.predict_k(z_test).cpu().numpy().flatten()
                # z_test * z_max = physical depth
                depths = z_test.cpu().numpy() * config.z_max
                parts = [f"z={d:.1f}m:{k:.4f}" for d, k in zip(depths, k_test)]
                logger.info("  k(z): %s", "  ".join(parts[:4]))
                logger.info("        %s", "  ".join(parts[4:]))

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

    # Surface T comparison
    t_d_obs = torch.tensor(
        data["t_obs_diurnal"], dtype=torch.float32
    ).to(device)
    t_s_obs = torch.tensor(
        data["t_obs_seasonal"], dtype=torch.float32
    ).to(device)
    z_zero = torch.zeros_like(t_d_obs)

    with torch.no_grad():
        T_pred = model.predict_T(z_zero, t_d_obs, t_s_obs).cpu().numpy().flatten()

    T_true = data["T_obs"]

    k_rmse = float(np.sqrt(np.mean((k_pred - k_true) ** 2)))
    T_rmse = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))

    TI_pred = np.sqrt(np.clip(k_pred, 0, None) * RHO * C_P)
    TI_true = np.sqrt(k_true * RHO * C_P)

    # Layer detection: check if PINN finds the transition
    boundary_idx = np.argmax(np.abs(np.gradient(k_pred)))
    boundary_depth = z[boundary_idx]
    k_above = np.mean(k_pred[:max(boundary_idx, 1)])
    k_below = np.mean(k_pred[min(boundary_idx + 10, len(k_pred)):])

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("  k RMSE:         %.4f W/m/K", k_rmse)
    logger.info("  T surface RMSE: %.2f K", T_rmse)
    logger.info("  TI predicted:   %.0f - %.0f tiu", TI_pred.min(), TI_pred.max())
    logger.info("  TI true:        %.0f - %.0f tiu", TI_true.min(), TI_true.max())
    logger.info("  Layer boundary: %.2f m (detected)", boundary_depth)
    logger.info("  k above/below:  %.4f / %.4f W/m/K", k_above, k_below)
    logger.info("  k true  range:  %.4f - %.4f W/m/K", k_true.min(), k_true.max())
    logger.info("=" * 50)

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
        "boundary_depth": boundary_depth,
    }
