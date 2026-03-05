"""
V4b FiLM Classifier — Feature-wise Linear Modulation.

MOLA terrain features condition how DINOv2 visual features are interpreted
via multiplicative modulation (gamma * CLS + beta), rather than concatenation.

This head-only module takes pre-extracted embeddings (768-dim CLS token)
and MOLA features (25-dim), and produces 4-class logits.

Classes: LDA (0), LVF (1), CCF (2), OTHER (3)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    MOLA features generate per-feature scale (gamma) and shift (beta)
    that modulate the visual CLS token:
        output = gamma * visual_features + beta

    This lets terrain context CONDITION how visual features are interpreted,
    rather than being concatenated as a flat vector.
    """

    def __init__(self, mola_dim: int, visual_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mola_encoder = nn.Sequential(
            nn.BatchNorm1d(mola_dim),
            nn.Linear(mola_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Generate gamma (scale) and beta (shift) for each visual feature
        self.gamma_proj = nn.Linear(hidden_dim, visual_dim)
        self.beta_proj = nn.Linear(hidden_dim, visual_dim)

        # Initialize gamma close to 1, beta close to 0 (identity init)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)

    def forward(self, visual_features: torch.Tensor, mola_features: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.
        visual_features: (B, 768) — DINOv2 CLS token
        mola_features: (B, 25) — terrain features
        Returns: (B, 768) — modulated visual features
        """
        h = self.mola_encoder(mola_features)  # (B, hidden_dim)
        gamma = self.gamma_proj(h)  # (B, 768)
        beta = self.beta_proj(h)    # (B, 768)
        return gamma * visual_features + beta


class FiLMClassifier(nn.Module):
    """
    V4b FiLM-based tile classifier.

    Takes pre-extracted DINOv2 CLS embeddings (768) + MOLA features (25),
    applies FiLM conditioning, then classifies via MLP head.

    Same forward(embeddings, mola) interface as TileLandformClassifier
    for drop-in replacement.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        mola_dim: int = 25,
        film_hidden: int = 64,
        head_hidden: int = 128,
        num_classes: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.film = FiLMLayer(mola_dim, visual_dim, film_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, 768)
        mola: torch.Tensor,        # (B, 25)
    ) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        modulated = self.film(embeddings, mola)  # (B, 768)
        return self.classifier(modulated)
