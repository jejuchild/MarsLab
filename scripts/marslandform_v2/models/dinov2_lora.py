from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import Dinov2Model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import DINOv2Config


class DinoV2LoRA(nn.Module):
    def __init__(
        self,
        config: DINOv2Config,
        use_lora: bool = True,
        lora_weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_lora = use_lora

        try:
            self.backbone = Dinov2Model.from_pretrained(config.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load DINOv2 backbone '{config.model_name}': {exc}") from exc

        if self.use_lora:
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type=TaskType.FEATURE_EXTRACTION,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            if lora_weights_path is not None:
                self.load_pretrained(lora_weights_path)

        self._freeze_all_parameters()
        self._unfreeze_last_blocks(config.unfreeze_last_n_blocks)
        self._print_trainable_params()

    def _freeze_all_parameters(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def _unfreeze_last_blocks(self, n_blocks: int) -> None:
        if n_blocks <= 0:
            return

        blocks = self._encoder_layers()
        n_blocks = min(n_blocks, len(blocks))
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def _encoder_layers(self) -> list[nn.Module]:
        if self.use_lora:
            base_model = getattr(self.backbone, "base_model", None)
            model = getattr(base_model, "model", None)
        else:
            model = self.backbone

        encoder = getattr(model, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError("Could not resolve DINOv2 encoder layers.")
        return list(layers)

    def num_layers(self) -> int:
        return len(self._encoder_layers())

    def _print_trainable_params(self) -> None:
        total = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        pct = (100.0 * trainable / total) if total > 0 else 0.0
        print(f"[DinoV2LoRA] trainable params: {trainable:,}/{total:,} ({pct:.2f}%)")

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0]
        if cls.shape[-1] != self.config.embed_dim:
            raise RuntimeError(
                f"Unexpected CLS embedding dim {cls.shape[-1]} (expected {self.config.embed_dim})"
            )
        return cls

    @torch.no_grad()
    def get_intermediate_features(self, pixel_values: Tensor) -> list[Tensor]:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 2:
            raise RuntimeError("No intermediate hidden states returned by DINOv2 model.")

        selected = [int(x) for x in torch.linspace(1, len(hidden_states) - 1, steps=4, dtype=torch.long).tolist()]
        return [hidden_states[idx][:, 0] for idx in selected]

    def save_pretrained(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.use_lora:
            self.backbone.save_pretrained(str(output_path))
        else:
            self.backbone.save_pretrained(str(output_path))

    def load_pretrained(self, path: str | Path) -> None:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"LoRA checkpoint path does not exist: {model_path}")

        if not self.use_lora:
            raise RuntimeError("load_pretrained is only supported when use_lora=True")

        self.backbone = PeftModel.from_pretrained(self.backbone, str(model_path), is_trainable=True)
