from __future__ import annotations

import os

import numpy as np
import torch

from .mars_model import MARS_RADIUS_KM, get_reference_model
from .pinn_model import MarsInteriorPINN

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_MODULE_DIR, "checkpoints")
_BEST_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "best_pinn_model.pt")


class MarsInteriorPredictor:
    def __init__(self, model: MarsInteriorPINN, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | None = None, device: str = "cpu") -> "MarsInteriorPredictor":
        path = checkpoint_path or _BEST_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=True)
        hidden_dim = int(ckpt.get("hidden_dim", 64))
        model = MarsInteriorPINN(hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model=model, device=device)

    @torch.no_grad()
    def predict_velocity(self, depth_km: float | np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_km, dtype=np.float32)
        depth = np.clip(depth, 0.0, 1700.0)
        radius = MARS_RADIUS_KM - depth
        r_norm = (radius / MARS_RADIUS_KM).astype(np.float32)

        tensor = torch.from_numpy(r_norm.reshape(-1, 1)).to(self.device)
        vp = self.model(tensor).cpu().numpy().reshape(-1)
        return vp

    @torch.no_grad()
    def profile(self, n_points: int = 200) -> dict[str, object]:
        depth = np.linspace(0.0, 1700.0, n_points, dtype=np.float32)
        vp = self.predict_velocity(depth)
        return {
            "depth_km": depth.tolist(),
            "vp_km_s": vp.tolist(),
        }

    @torch.no_grad()
    def compare_with_reference(self, n_points: int = 120) -> dict[str, object]:
        depth = np.linspace(0.0, 1700.0, n_points, dtype=np.float32)
        learned = self.predict_velocity(depth)
        radius_ref, vp_ref, _, _ = get_reference_model(n_layers=n_points)
        depth_ref = (MARS_RADIUS_KM - radius_ref).astype(np.float32)
        mae = float(np.mean(np.abs(learned - vp_ref)))
        return {
            "depth_km": depth.tolist(),
            "learned_vp_km_s": learned.tolist(),
            "reference_depth_km": depth_ref.tolist(),
            "reference_vp_km_s": vp_ref.tolist(),
            "mae_km_s": mae,
        }


_predictor_instance: MarsInteriorPredictor | None = None


def get_predictor(device: str = "cpu") -> MarsInteriorPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = MarsInteriorPredictor.load(device=device)
    return _predictor_instance


def is_model_trained() -> bool:
    return os.path.exists(_BEST_MODEL_PATH)
