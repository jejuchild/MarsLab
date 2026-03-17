"""
V8 Segmentation Head — DINOv2 patch tokens → per-patch class prediction.

Architecture:
  DINOv2 ViT-L/14 patch tokens: (B, 256, 1024)  ← 16×16 grid, 1024-dim
  Reshape → (B, 1024, 16, 16)
  BatchNorm2d(1024) → Conv2d(1024, num_classes, 1×1)
  Output: (B, num_classes, 16, 16)

Trainable parameters: ~6K (BN: 2×1024=2048, Conv: 1024×4+4=4100)
Backbone is FROZEN — only the head trains.

Classes: LDA (0), LVF (1), CCF (2), OTHER (3)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSegmentationHead(nn.Module):
    """
    Lightweight segmentation head for DINOv2 patch tokens.

    Takes patch tokens from a frozen ViT backbone and produces per-patch
    class logits via BN + Conv1×1.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_classes: int = 4,
        patches_per_side: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patches_per_side = patches_per_side

        self.bn = nn.BatchNorm2d(embed_dim)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N_patches, embed_dim) from DINOv2 backbone
                          N_patches = patches_per_side² = 256 for 224×224 input

        Returns:
            logits: (B, num_classes, patches_per_side, patches_per_side)
        """
        B, N, D = patch_tokens.shape
        H = W = self.patches_per_side

        # Reshape: (B, N, D) → (B, D, H, W)
        x = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        # BN + Conv1×1
        x = self.bn(x)
        logits = self.classifier(x)  # (B, num_classes, H, W)

        return logits

    def predict(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: returns (probs, preds).

        Args:
            patch_tokens: (B, N_patches, embed_dim)

        Returns:
            probs: (B, num_classes, H, W) softmax probabilities
            preds: (B, H, W) int64 class predictions
        """
        logits = self.forward(patch_tokens)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return probs, preds
