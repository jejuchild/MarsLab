"""
Mars GCM Neural Emulator — Inference Predictor.

Loads a trained model checkpoint and provides a clean prediction API.
Handles model loading, input encoding, output decoding, and comparison
with the parametric model for validation.

Usage:
    from neural_climate.predictor import MarsClimatePredictor

    predictor = MarsClimatePredictor.load()
    result = predictor.predict(lat=0, lon=0, ls=180)
    batch = predictor.predict_batch(lats, lons, ls_vals)
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import torch

from neural_climate.model import (
    MarsClimateEmulator,
    create_model,
    encode_inputs,
    decode_outputs,
    OUTPUT_NAMES,
    OUTPUT_BOUNDS,
)

logger = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_MODULE_DIR, "checkpoints")
_BEST_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "best_model.pt")
_NORM_STATS_PATH = os.path.join(_CHECKPOINT_DIR, "norm_stats.npz")


class MarsClimatePredictor:
    """
    Inference wrapper for the trained Mars climate emulator.

    Loads model weights + normalization stats, provides single-point
    and batch prediction APIs with automatic input encoding and
    output denormalization.
    """

    def __init__(
        self,
        model: MarsClimateEmulator,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_mean = torch.from_numpy(target_mean).float().to(device)
        self.target_std = torch.from_numpy(target_std).float().to(device)

    @classmethod
    def load(
        cls,
        checkpoint_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "MarsClimatePredictor":
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: path to model .pt file (default: best_model.pt)
            norm_stats_path: path to norm_stats.npz (default: alongside model)
            device: inference device ('cpu' or 'cuda')

        Returns:
            MarsClimatePredictor instance ready for inference

        Raises:
            FileNotFoundError: if checkpoint files don't exist
        """
        checkpoint_path = checkpoint_path or _BEST_MODEL_PATH
        norm_stats_path = norm_stats_path or _NORM_STATS_PATH

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Run training first: python -m neural_climate.trainer"
            )
        if not os.path.exists(norm_stats_path):
            raise FileNotFoundError(
                f"Normalization stats not found: {norm_stats_path}. "
                f"Run training first: python -m neural_climate.trainer"
            )

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
        config = checkpoint.get("model_config", {})
        hidden_dim = config.get("hidden_dim", 256)
        n_blocks = config.get("n_blocks", 4)

        model = create_model(hidden_dim=hidden_dim, n_blocks=n_blocks)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load normalization stats
        norm_data = np.load(norm_stats_path)
        target_mean = norm_data["target_mean"]
        target_std = norm_data["target_std"]

        logger.info(
            f"Loaded emulator: {model.param_count:,} params, "
            f"hidden={hidden_dim}, blocks={n_blocks}, device={device}"
        )

        return cls(model, target_mean, target_std, device=device)

    @torch.no_grad()
    def predict(
        self,
        lat: float,
        lon: float,
        ls: float,
        elevation: Optional[float] = None,
    ) -> dict:
        """
        Predict climate for a single (lat, lon, Ls) point.

        Args:
            lat: latitude (-90 to 90)
            lon: longitude (-180 to 360)
            ls: solar longitude (0 to 360)
            elevation: surface elevation in meters.
                       If None, looks up from MOLA DEM.

        Returns:
            dict with OUTPUT_NAMES keys + metadata
        """
        if elevation is None:
            from api.mars_climate import get_elevation_m
            elevation = get_elevation_m(lat, lon)

        encoded = encode_inputs(
            np.array([lat]),
            np.array([lon]),
            np.array([ls]),
            np.array([elevation]),
        )

        X = torch.from_numpy(encoded).float().to(self.device)
        Y_norm = self.model(X)
        Y_phys = Y_norm * self.target_std + self.target_mean
        result = decode_outputs(Y_phys.cpu().numpy())[0]

        result["lat"] = lat
        result["lon"] = lon
        result["ls"] = ls
        result["elevation_m"] = elevation
        result["source"] = "neural_emulator"

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        ls_vals: np.ndarray,
        elevations: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Batch prediction for multiple points.

        Args:
            lats: (N,) latitudes
            lons: (N,) longitudes
            ls_vals: (N,) solar longitudes
            elevations: (N,) elevations. If None, looks up from MOLA DEM.

        Returns:
            list of N dicts with predictions
        """
        lats = np.asarray(lats, dtype=np.float32)
        lons = np.asarray(lons, dtype=np.float32)
        ls_vals = np.asarray(ls_vals, dtype=np.float32)

        if elevations is None:
            from api.mars_climate import get_elevation_m
            elevations = np.array(
                [get_elevation_m(lat, lon) for lat, lon in zip(lats, lons)],
                dtype=np.float32,
            )

        encoded = encode_inputs(lats, lons, ls_vals, elevations)
        X = torch.from_numpy(encoded).float().to(self.device)

        # Process in chunks if very large
        chunk_size = 4096
        all_preds = []
        for i in range(0, len(X), chunk_size):
            chunk = X[i:i + chunk_size]
            Y_norm = self.model(chunk)
            Y_phys = Y_norm * self.target_std + self.target_mean
            all_preds.append(Y_phys.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        results = decode_outputs(preds)

        for j, r in enumerate(results):
            r["lat"] = float(lats[j])
            r["lon"] = float(lons[j])
            r["ls"] = float(ls_vals[j])
            r["elevation_m"] = float(elevations[j])
            r["source"] = "neural_emulator"

        return results

    @torch.no_grad()
    def predict_global_map(
        self,
        ls: float,
        lat_step: float = 2.0,
        lon_step: float = 2.0,
        output_var: str = "temperature_mean_k",
    ) -> dict:
        """
        Generate a global climate map for a given Ls.

        Args:
            ls: solar longitude
            lat_step: latitude resolution (degrees)
            lon_step: longitude resolution (degrees)
            output_var: which output variable to map

        Returns:
            dict with 'lats', 'lons', 'values' (2D array), 'ls', 'variable'
        """
        lats_1d = np.arange(-90, 90 + lat_step / 2, lat_step)
        lons_1d = np.arange(0, 360, lon_step)

        lat_grid, lon_grid = np.meshgrid(lats_1d, lons_1d, indexing="ij")
        flat_lats = lat_grid.flatten()
        flat_lons = lon_grid.flatten()
        flat_ls = np.full_like(flat_lats, ls)

        results = self.predict_batch(flat_lats, flat_lons, flat_ls)

        if output_var not in OUTPUT_NAMES:
            raise ValueError(
                f"Unknown output variable: {output_var}. "
                f"Choose from: {OUTPUT_NAMES}"
            )

        values = np.array([r[output_var] for r in results]).reshape(lat_grid.shape)

        return {
            "lats": lats_1d.tolist(),
            "lons": lons_1d.tolist(),
            "values": values.tolist(),
            "ls": ls,
            "variable": output_var,
            "shape": list(values.shape),
        }

    def compare_with_parametric(
        self,
        lat: float,
        lon: float,
        ls: float,
    ) -> dict:
        """
        Compare neural emulator output with parametric model for validation.

        Returns dict with 'neural', 'parametric', and 'errors' sub-dicts.
        """
        from api.mars_climate import (
            surface_temperature_k,
            surface_pressure_pa,
            dust_opacity,
            wind_speed,
            co2_frost_probability,
            get_elevation_m,
        )

        elevation = get_elevation_m(lat, lon)
        neural = self.predict(lat, lon, ls, elevation=elevation)

        temp = surface_temperature_k(lat, ls, elevation)
        pressure = surface_pressure_pa(elevation)
        dust = dust_opacity(lat, ls)
        wind = wind_speed(lat, ls)
        frost = co2_frost_probability(lat, ls, elevation)

        parametric = {
            "temperature_mean_k": temp["mean_k"],
            "temperature_max_k": temp["max_k"],
            "temperature_min_k": temp["min_k"],
            "pressure_pa": round(pressure, 4),
            "dust_tau_mean": dust["tau_mean"],
            "wind_mean_ms": wind["mean_ms"],
            "frost_probability": frost["frost_probability"],
        }

        errors = {}
        for name in OUTPUT_NAMES:
            n_val = neural[name]
            p_val = parametric[name]
            abs_err = abs(n_val - p_val)
            rel_err = abs_err / max(abs(p_val), 1e-8)
            errors[name] = {
                "absolute": round(abs_err, 4),
                "relative_pct": round(rel_err * 100, 2),
            }

        return {
            "lat": lat,
            "lon": lon,
            "ls": ls,
            "elevation_m": elevation,
            "neural": neural,
            "parametric": parametric,
            "errors": errors,
        }

    @property
    def model_info(self) -> dict:
        """Return model metadata."""
        return {
            "param_count": self.model.param_count,
            "trainable_params": self.model.trainable_param_count,
            "hidden_dim": self.model.input_proj[0].out_features if hasattr(self.model.input_proj[0], 'out_features') else None,
            "n_blocks": len(self.model.blocks),
            "device": str(self.device),
            "output_variables": OUTPUT_NAMES,
            "checkpoint_path": _BEST_MODEL_PATH,
            "checkpoint_exists": os.path.exists(_BEST_MODEL_PATH),
        }


# ---------------------------------------------------------------------------
# Singleton loader (lazy)
# ---------------------------------------------------------------------------

_predictor_instance: Optional[MarsClimatePredictor] = None


def get_predictor(device: str = "cpu") -> MarsClimatePredictor:
    """
    Get or create singleton predictor instance.

    Raises FileNotFoundError if model hasn't been trained yet.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MarsClimatePredictor.load(device=device)
    return _predictor_instance


def is_model_trained() -> bool:
    """Check if a trained model checkpoint exists."""
    return (
        os.path.exists(_BEST_MODEL_PATH)
        and os.path.exists(_NORM_STATS_PATH)
    )
