from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mars_model import MARS_RADIUS_KM


class MarsInteriorPINN(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, r_norm: torch.Tensor) -> torch.Tensor:
        if r_norm.ndim == 1:
            r_norm = r_norm.unsqueeze(-1)
        raw = self.net(r_norm)
        return F.softplus(raw) + 0.25

    def predict_travel_times(self, distances_deg: torch.Tensor, n_steps: int = 160) -> torch.Tensor:
        distances_deg = distances_deg.view(-1)
        delta = torch.deg2rad(distances_deg).clamp(math.radians(1.0), math.radians(179.0))

        chord_km = 2.0 * MARS_RADIUS_KM * torch.sin(0.5 * delta)
        b_km = MARS_RADIUS_KM * torch.cos(0.5 * delta)

        s = torch.linspace(-0.5, 0.5, steps=n_steps, device=distances_deg.device)
        ds = 1.0 / max(n_steps - 1, 1)
        x = chord_km[:, None] * s[None, :]
        r_km = torch.sqrt(torch.clamp(b_km[:, None] ** 2 + x**2, min=1e-6))
        r_norm = (r_km / MARS_RADIUS_KM).clamp(0.0, 1.0)

        vp = self.forward(r_norm.reshape(-1, 1)).reshape_as(r_km)
        dt = (chord_km[:, None] * ds) / torch.clamp(vp, min=1e-4)
        return dt.sum(dim=1)


def compute_physics_loss(model: MarsInteriorPINN, n_points: int = 256) -> torch.Tensor:
    device = next(model.parameters()).device
    r = torch.linspace(0.0, 1.0, n_points, device=device, requires_grad=True).unsqueeze(-1)
    v = model(r)
    dv = torch.autograd.grad(v.sum(), r, create_graph=True)[0]
    d2v = torch.autograd.grad(dv.sum(), r, create_graph=True)[0]

    smooth = torch.mean(d2v.pow(2))
    positive = torch.mean(F.relu(0.1 - v))

    r_cmb = (MARS_RADIUS_KM - 1500.0) / MARS_RADIUS_KM
    v_core = model(torch.tensor([[max(0.0, r_cmb - 0.02)]], device=device))
    v_mantle = model(torch.tensor([[min(1.0, r_cmb + 0.02)]], device=device))
    cmb_drop = F.relu(v_core - v_mantle)

    return smooth + 0.5 * positive + 0.2 * cmb_drop.mean()


def compute_boundary_loss(model: MarsInteriorPINN) -> torch.Tensor:
    device = next(model.parameters()).device
    v_surface = model(torch.tensor([[1.0]], device=device))
    surface = torch.mean((v_surface - 3.5) ** 2)

    r = torch.linspace(0.0, 1.0, 64, device=device).unsqueeze(-1)
    v = model(r)
    positive = torch.mean(F.relu(0.0 - v))
    return surface + positive
