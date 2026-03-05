"""
Mars GCM Neural Emulator — Training Dataset Generator.

Generates (input, target) pairs by evaluating the parametric Mars climate
model on a dense grid of (lat, lon, Ls) coordinates.  Elevation is looked
up from MOLA DEM for each (lat, lon) pair.

Grid design:
    lat:  -90 to 90    (step configurable, default 5°  → 37 values)
    lon:  0 to 355     (step configurable, default 10° → 36 values)
    Ls:   0 to 350     (step configurable, default 10° → 36 values)
    Total default: 37 × 36 × 36 = 47,952 samples

Each sample produces:
    Input:  (lat, lon, Ls, elevation_m)
    Target: (T_mean, T_max, T_min, pressure, dust_tau, wind_mean, frost_prob)
"""

import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Add backend to path so we can import the parametric model
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from api.mars_climate import (
    surface_temperature_k,
    surface_pressure_pa,
    dust_opacity,
    wind_speed,
    co2_frost_probability,
    get_elevation_m,
)
from neural_climate.model import encode_inputs, OUTPUT_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid(
    lat_step: float = 5.0,
    lon_step: float = 10.0,
    ls_step: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate regular (lat, lon, Ls) grid.

    Returns:
        lats: 1-D array of latitudes
        lons: 1-D array of longitudes
        ls_vals: 1-D array of solar longitudes
    """
    lats = np.arange(-90, 90 + lat_step / 2, lat_step)
    lons = np.arange(0, 360, lon_step)
    ls_vals = np.arange(0, 360, ls_step)
    return lats, lons, ls_vals


def sample_parametric_model(
    lat: float,
    lon: float,
    ls: float,
) -> Optional[dict]:
    """
    Query the parametric climate model for a single (lat, lon, Ls) point.

    Returns dict with all 7 target values, or None on failure.
    """
    try:
        elevation = get_elevation_m(lat, lon)

        temp = surface_temperature_k(lat, ls, elevation)
        pressure = surface_pressure_pa(elevation)
        dust = dust_opacity(lat, ls)
        wind = wind_speed(lat, ls)
        frost = co2_frost_probability(lat, ls, elevation)

        return {
            "lat": lat,
            "lon": lon,
            "ls": ls,
            "elevation_m": elevation,
            "temperature_mean_k": temp["mean_k"],
            "temperature_max_k": temp["max_k"],
            "temperature_min_k": temp["min_k"],
            "pressure_pa": pressure,
            "dust_tau_mean": dust["tau_mean"],
            "wind_mean_ms": wind["mean_ms"],
            "frost_probability": frost["frost_probability"],
        }
    except Exception as e:
        logger.warning(f"Parametric model failed at ({lat}, {lon}, Ls={ls}): {e}")
        return None


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

def generate_training_data(
    lat_step: float = 5.0,
    lon_step: float = 10.0,
    ls_step: float = 10.0,
    add_noise: bool = False,
    noise_fraction: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate full training dataset from parametric model.

    Args:
        lat_step: latitude grid spacing (degrees)
        lon_step: longitude grid spacing (degrees)
        ls_step: solar longitude grid spacing (degrees)
        add_noise: if True, add small Gaussian noise to targets
                   (regularization for training)
        noise_fraction: relative noise magnitude (fraction of value)
        seed: random seed for reproducibility

    Returns:
        inputs:     (N, 4) raw inputs [lat, lon, Ls, elevation]
        targets:    (N, 7) target climate variables
        encoded:    (N, 6) encoded input features (ready for model)
        meta:       (N, 3) metadata [lat, lon, Ls] for debugging
    """
    rng = np.random.default_rng(seed)
    lats, lons, ls_vals = generate_grid(lat_step, lon_step, ls_step)

    n_total = len(lats) * len(lons) * len(ls_vals)
    logger.info(
        f"Generating {n_total} samples: "
        f"{len(lats)} lats × {len(lons)} lons × {len(ls_vals)} Ls values"
    )

    raw_inputs = []    # (lat, lon, ls, elevation)
    raw_targets = []   # 7 target values
    meta_data = []     # (lat, lon, ls) for debugging

    t0 = time.time()
    n_done = 0
    n_failed = 0

    for lat in lats:
        for lon in lons:
            elevation = get_elevation_m(lat, lon)
            for ls in ls_vals:
                sample = sample_parametric_model(lat, lon, ls)
                if sample is None:
                    n_failed += 1
                    continue

                raw_inputs.append([lat, lon, ls, elevation])
                raw_targets.append([
                    sample["temperature_mean_k"],
                    sample["temperature_max_k"],
                    sample["temperature_min_k"],
                    sample["pressure_pa"],
                    sample["dust_tau_mean"],
                    sample["wind_mean_ms"],
                    sample["frost_probability"],
                ])
                meta_data.append([lat, lon, ls])
                n_done += 1

        # Progress logging every 10 latitude bands
        if int(lat) % 50 == 0 and lat != -90:
            elapsed = time.time() - t0
            logger.info(
                f"  lat={lat:+6.1f}° | {n_done}/{n_total} samples "
                f"({100*n_done/n_total:.0f}%) | {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Generated {n_done} samples in {elapsed:.1f}s "
        f"({n_failed} failures)"
    )

    inputs = np.array(raw_inputs, dtype=np.float32)
    targets = np.array(raw_targets, dtype=np.float32)
    meta = np.array(meta_data, dtype=np.float32)

    # Optional noise injection for regularization
    if add_noise and noise_fraction > 0:
        noise = rng.normal(0, noise_fraction, size=targets.shape).astype(np.float32)
        # Scale noise relative to each column's magnitude
        col_scales = np.abs(targets).mean(axis=0) + 1e-8
        targets = targets + noise * col_scales

    # Encode inputs for the model
    encoded = encode_inputs(
        inputs[:, 0],  # lat
        inputs[:, 1],  # lon
        inputs[:, 2],  # ls
        inputs[:, 3],  # elevation
    )

    return inputs, targets, encoded, meta


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MarsClimateDataset(Dataset):
    """PyTorch Dataset wrapper for Mars climate training data."""

    def __init__(
        self,
        encoded_inputs: np.ndarray,
        targets: np.ndarray,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(encoded_inputs).float()
        self.Y_raw = torch.from_numpy(targets).float()

        # Target normalization (z-score) for better training dynamics
        if target_mean is not None and target_std is not None:
            self.target_mean = torch.from_numpy(target_mean).float()
            self.target_std = torch.from_numpy(target_std).float()
        else:
            self.target_mean = self.Y_raw.mean(dim=0)
            self.target_std = self.Y_raw.std(dim=0) + 1e-8

        self.Y = (self.Y_raw - self.target_mean) / self.target_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

    def denormalize(self, y_normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized targets back to physical units."""
        return y_normalized * self.target_std + self.target_mean


def create_dataloaders(
    encoded_inputs: np.ndarray,
    targets: np.ndarray,
    train_fraction: float = 0.85,
    batch_size: int = 512,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train/val DataLoaders with target normalization.

    Returns:
        train_loader: training DataLoader
        val_loader: validation DataLoader
        norm_stats: dict with 'target_mean' and 'target_std' numpy arrays
    """
    # Compute normalization stats on full dataset
    target_mean = targets.mean(axis=0)
    target_std = targets.std(axis=0) + 1e-8

    dataset = MarsClimateDataset(
        encoded_inputs, targets,
        target_mean=target_mean,
        target_std=target_std,
    )

    n_total = len(dataset)
    n_train = int(n_total * train_fraction)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    norm_stats = {
        "target_mean": target_mean,
        "target_std": target_std,
    }

    logger.info(
        f"DataLoaders: {n_train} train / {n_val} val "
        f"(batch_size={batch_size})"
    )

    return train_loader, val_loader, norm_stats


# ---------------------------------------------------------------------------
# Save / load generated data (cache for re-training)
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def save_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    encoded: np.ndarray,
    meta: np.ndarray,
    data_dir: Optional[str] = None,
):
    """Save generated dataset to .npz file."""
    data_dir = data_dir or _DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "mars_climate_dataset.npz")
    np.savez_compressed(
        path,
        inputs=inputs,
        targets=targets,
        encoded=encoded,
        meta=meta,
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"Saved dataset to {path} ({size_mb:.1f} MB)")
    return path


def load_dataset(
    data_dir: Optional[str] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Load previously generated dataset. Returns None if not found."""
    data_dir = data_dir or _DATA_DIR
    path = os.path.join(data_dir, "mars_climate_dataset.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    logger.info(f"Loaded dataset from {path} ({len(data['inputs'])} samples)")
    return data["inputs"], data["targets"], data["encoded"], data["meta"]
