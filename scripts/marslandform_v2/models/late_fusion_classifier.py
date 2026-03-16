"""
V6 Late Fusion Classifier — HiRISE visual = main, MOLA = sub.

Two independent classification heads produce logits from their respective
modalities, then combine at the logit level with learnable weights.

Architecture:
  visual_head: DINOv2 CLS (768) → MLP → logits (4)
  mola_head:   MOLA (25)       → MLP → logits (4)
  final:       w_vis * visual_logits + w_mola * mola_logits

The visual head sees raw DINOv2 features directly (no FiLM conditioning),
so it learns to classify based on image content alone. MOLA provides
a supplementary signal.

MOLA inputs are z-score normalized using training distribution statistics
(stored as registered buffers so they travel with the checkpoint). This
prevents OOD MOLA inputs from producing extreme logits.

Classes: LDA (0), LVF (1), CCF (2), OTHER (3)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionClassifier(nn.Module):
    """
    Late fusion: two independent heads, combined at logit level.

    forward(embeddings, mola) interface is identical to FiLMClassifier
    for drop-in replacement.

    MOLA features are z-score normalized using training stats stored
    as registered buffers (mola_mean, mola_std). This ensures OOD MOLA
    inputs are bounded and don't produce extreme logits.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        mola_dim: int = 25,
        visual_hidden: int = 256,
        mola_hidden: int = 64,
        num_classes: int = 4,
        dropout: float = 0.3,
        init_vis_weight: float = 0.7,
    ):
        super().__init__()
        self.num_classes = num_classes

        # MOLA normalization stats — registered buffers (not parameters)
        # Will be overwritten with actual training stats during training or loading
        self.register_buffer("mola_mean", torch.zeros(mola_dim))
        self.register_buffer("mola_std", torch.ones(mola_dim))

        # Visual head — main classifier
        self.visual_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_hidden, visual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(visual_hidden, num_classes),
        )

        # MOLA head — supplementary (no BatchNorm, we normalize explicitly)
        self.mola_head = nn.Sequential(
            nn.Linear(mola_dim, mola_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mola_hidden, mola_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mola_hidden, num_classes),
        )

        # Learnable fusion weight (initialized to favor visual)
        # Parameterized in log-space for stability
        self._log_vis_weight = nn.Parameter(
            torch.tensor(init_vis_weight).log()
        )
        self._log_mola_weight = nn.Parameter(
            torch.tensor(1.0 - init_vis_weight).log()
        )

    def set_mola_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set MOLA normalization statistics from training data."""
        self.mola_mean.copy_(mean)
        # Clamp std to avoid division by zero for constant features
        self.mola_std.copy_(std.clamp(min=1e-6))

    def _normalize_mola(self, mola: torch.Tensor) -> torch.Tensor:
        """Z-score normalize MOLA features using training statistics."""
        return (mola - self.mola_mean) / self.mola_std

    @property
    def vis_weight(self) -> torch.Tensor:
        """Normalized visual weight via softmax."""
        weights = torch.softmax(
            torch.stack([self._log_vis_weight, self._log_mola_weight]), dim=0
        )
        return weights[0]

    @property
    def mola_weight(self) -> torch.Tensor:
        """Normalized MOLA weight via softmax."""
        weights = torch.softmax(
            torch.stack([self._log_vis_weight, self._log_mola_weight]), dim=0
        )
        return weights[1]

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, 768)
        mola: torch.Tensor,        # (B, 25)
    ) -> torch.Tensor:
        """Returns fused logits (B, num_classes)."""
        vis_logits = self.visual_head(embeddings)    # (B, 4)

        # Normalize MOLA features using training stats
        mola_normed = self._normalize_mola(mola)
        mola_logits = self.mola_head(mola_normed)    # (B, 4)

        w_vis = self.vis_weight
        w_mola = self.mola_weight

        return w_vis * vis_logits + w_mola * mola_logits

    def get_visual_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Visual-only logits (for diagnostics)."""
        return self.visual_head(embeddings)

    def get_mola_logits(self, mola: torch.Tensor) -> torch.Tensor:
        """MOLA-only logits (for diagnostics). Applies normalization."""
        mola_normed = self._normalize_mola(mola)
        return self.mola_head(mola_normed)
