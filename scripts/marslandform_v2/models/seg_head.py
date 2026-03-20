"""
V8.1 Segmentation Head — DINOv2 patch tokens → per-patch class prediction.

Architecture:
  Visual: (B,256,1024) → BN → Conv1×1→hidden → [Conv3×3 + DilatedConv3×3] (residual)
  MOLA (optional): (B,25) → MLP→mola_hidden → broadcast to (B,mola_hidden,16,16)
  Fusion: cat([visual, mola]) → Conv1×1 → logits

~143K trainable params. Backbone is FROZEN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSegmentationHead(nn.Module):
    """
    Segmentation head for DINOv2 patch tokens with spatial context
    and optional MOLA elevation feature fusion.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 4,
        patches_per_side: int = 16,
        hidden_dim: int = 64,
        mola_dim: int = 0,
        mola_hidden: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patches_per_side = patches_per_side
        self.use_mola = mola_dim > 0

        self.bn = nn.BatchNorm2d(embed_dim)
        self.project = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )

        fusion_dim = hidden_dim
        if self.use_mola:
            self.mola_head = nn.Sequential(
                nn.Linear(mola_dim, mola_hidden),
                nn.GELU(),
                nn.Linear(mola_hidden, mola_hidden),
            )
            fusion_dim += mola_hidden

        self.classifier = nn.Conv2d(fusion_dim, num_classes, kernel_size=1)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        mola_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 256, 1024) from frozen DINOv2 ViT-L
            mola_features: (B, mola_dim) per-tile MOLA features, or None

        Returns:
            logits: (B, num_classes, 16, 16)
        """
        B, N, D = patch_tokens.shape
        H = W = self.patches_per_side

        x = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        x = self.bn(x)
        x = F.gelu(self.project(x))
        x = x + self.spatial(x)

        if self.use_mola and mola_features is not None:
            m = self.mola_head(mola_features)
            m = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            x = torch.cat([x, m], dim=1)

        return self.classifier(x)

    def predict(
        self,
        patch_tokens: torch.Tensor,
        mola_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (B, num_classes, H, W) softmax probabilities
            preds: (B, H, W) class predictions
        """
        logits = self.forward(patch_tokens, mola_features)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return probs, preds
