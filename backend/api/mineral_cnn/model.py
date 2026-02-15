"""
CNN Mineral Classification — Model Architecture & Singleton

MultiBranchAttnCNN with 7 spectral branches and attention fusion.
Model weights loaded once (lazy) and held in module-level state.
"""

import json
import threading

import torch
import torch.nn as nn

from .constants import (
    MODEL_WEIGHTS, CLASS_MAP_PATH, INV_CLASS_MAP_PATH,
    GROUPS_UM, CLASS_NAME,
)


# ============================================================
# Model Architecture (must match training exactly)
# ============================================================
class SpectralBranch(nn.Module):
    def __init__(self, out_ch=64, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool),
        )

    def forward(self, x):
        x = self.net(x.unsqueeze(1))
        return x.flatten(1)


class MultiBranchAttnCNN(nn.Module):
    def __init__(self, n_branches, n_classes, feat_ch=64, pool=2, att_temperature=0.5):
        super().__init__()
        feat_dim = feat_ch * pool

        self.branches = nn.ModuleList(
            [SpectralBranch(out_ch=feat_ch, pool=pool) for _ in range(n_branches)]
        )

        self.att_fc = nn.Sequential(
            nn.Linear(feat_dim * n_branches, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, n_branches),
        )

        self.cls = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

        self.att_temperature = float(att_temperature)

    def forward(self, *xg, return_attn=False):
        feats = [b(x) for b, x in zip(self.branches, xg)]
        cat = torch.cat(feats, dim=1)

        att_logits = self.att_fc(cat)
        w = torch.softmax(att_logits / self.att_temperature, dim=1)

        f = sum(w[:, i:i + 1] * feats[i] for i in range(len(feats)))
        logits = self.cls(f)
        return (logits, w) if return_attn else logits


# ============================================================
# Singleton Model Loading
# ============================================================
_model: MultiBranchAttnCNN | None = None
_class_map: dict | None = None
_inv_class_map: dict | None = None
_device: str = "cpu"
_lock = threading.Lock()


def get_model() -> tuple:
    """
    Return (model, class_map, inv_class_map, device).
    Loads the model on first call (thread-safe lazy singleton).
    """
    global _model, _class_map, _inv_class_map, _device
    if _model is not None:
        return _model, _class_map, _inv_class_map, _device

    with _lock:
        if _model is not None:  # double-check after acquiring lock
            return _model, _class_map, _inv_class_map, _device

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(CLASS_MAP_PATH) as f:
            _class_map = {int(k): int(v) for k, v in json.load(f).items()}
        with open(INV_CLASS_MAP_PATH) as f:
            _inv_class_map = {int(k): int(v) for k, v in json.load(f).items()}

        n_classes = len(_class_map)
        _model = MultiBranchAttnCNN(len(GROUPS_UM), n_classes).to(_device)
        _model.load_state_dict(
            torch.load(MODEL_WEIGHTS, map_location=_device, weights_only=True)
        )
        _model.eval()
        print(f"[CNN] Model loaded: {n_classes} classes, device={_device}")

        return _model, _class_map, _inv_class_map, _device


def model_status() -> dict:
    """Return model status without triggering load."""
    return {
        "loaded": _model is not None,
        "device": _device if _model else None,
        "n_classes": len(_class_map) if _class_map else 0,
        "model_path": MODEL_WEIGHTS,
        "class_names": {str(k): v for k, v in CLASS_NAME.items()},
    }
