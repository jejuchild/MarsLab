"""
Mars GCM Neural Emulator — PyTorch Model Definition.

4-layer MLP with batch normalization and skip connections that maps
(lat, lon, Ls, elevation) to 7 climate variables.

Input encoding:
    lat         → normalized to [-1, 1]  (lat / 90)
    lon         → sin(lon_rad), cos(lon_rad)  (avoids 0/360 discontinuity)
    Ls          → sin(ls_rad), cos(ls_rad)   (avoids 0/360 discontinuity)
    elevation   → normalized by scale height  (elev / 10800)
    Total: 6 input features

Output (7 values):
    T_mean (K), T_max (K), T_min (K), pressure (Pa),
    dust_tau, wind_mean (m/s), frost_prob (0-1)
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature encoding constants
# ---------------------------------------------------------------------------
INPUT_DIM = 6   # lat_norm, sin_lon, cos_lon, sin_ls, cos_ls, elev_norm
OUTPUT_DIM = 7  # T_mean, T_max, T_min, P, tau, wind, frost

OUTPUT_NAMES = [
    "temperature_mean_k",
    "temperature_max_k",
    "temperature_min_k",
    "pressure_pa",
    "dust_tau_mean",
    "wind_mean_ms",
    "frost_probability",
]

# Physical bounds for output clamping (safety net)
OUTPUT_BOUNDS = {
    "temperature_mean_k": (148.0, 280.0),
    "temperature_max_k":  (148.0, 310.0),
    "temperature_min_k":  (148.0, 250.0),
    "pressure_pa":        (30.0,  1200.0),
    "dust_tau_mean":      (0.05,  5.0),
    "wind_mean_ms":       (0.5,   25.0),
    "frost_probability":  (0.0,   1.0),
}


# ---------------------------------------------------------------------------
# Input encoding / decoding
# ---------------------------------------------------------------------------

def encode_inputs(
    lat: np.ndarray,
    lon: np.ndarray,
    ls: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    """
    Encode raw (lat, lon, Ls, elevation) into 6-dim feature vector.

    Args:
        lat: latitude in degrees [-90, 90]
        lon: longitude in degrees [-180, 360]
        ls:  solar longitude in degrees [0, 360]
        elevation: surface elevation in meters

    Returns:
        (N, 6) float32 array
    """
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    ls = np.asarray(ls, dtype=np.float32)
    elevation = np.asarray(elevation, dtype=np.float32)

    lat_norm = lat / 90.0                          # [-1, 1]
    lon_rad = np.radians(lon)
    ls_rad = np.radians(ls)
    elev_norm = elevation / 10800.0                 # scale height normalization

    features = np.stack([
        lat_norm,
        np.sin(lon_rad),
        np.cos(lon_rad),
        np.sin(ls_rad),
        np.cos(ls_rad),
        elev_norm,
    ], axis=-1)

    return features


def decode_outputs(predictions: np.ndarray) -> list[dict]:
    """
    Convert raw model output (N, 7) to list of named dicts.

    Each dict contains OUTPUT_NAMES keys with physically-clamped values.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)

    results = []
    for row in predictions:
        entry = {}
        for i, name in enumerate(OUTPUT_NAMES):
            val = float(row[i])
            lo, hi = OUTPUT_BOUNDS[name]
            val = max(lo, min(hi, val))
            entry[name] = round(val, 4)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """MLP block with skip connection and batch normalization."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MarsClimateEmulator(nn.Module):
    """
    Neural emulator for Mars parametric climate model.

    Architecture:
        Linear(6 → hidden) → [ResidualBlock × n_blocks] → Linear(hidden → 7)

    The frost_probability output uses a sigmoid to constrain it to [0, 1].
    Other outputs are unconstrained (clamped post-hoc at inference).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])

        self.output_head = nn.Linear(hidden_dim, OUTPUT_DIM)
        self.frost_sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, 6) encoded input features

        Returns:
            (batch, 7) climate predictions
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        out = self.output_head(h)

        # Constrain frost probability to [0, 1] via sigmoid
        out = torch.cat([
            out[:, :6],                          # T_mean, T_max, T_min, P, tau, wind
            self.frost_sigmoid(out[:, 6:7]),      # frost_prob
        ], dim=1)

        return out

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    hidden_dim: int = 256,
    n_blocks: int = 4,
    dropout: float = 0.05,
) -> MarsClimateEmulator:
    """Factory function to create a new emulator model."""
    return MarsClimateEmulator(
        hidden_dim=hidden_dim,
        n_blocks=n_blocks,
        dropout=dropout,
    )
