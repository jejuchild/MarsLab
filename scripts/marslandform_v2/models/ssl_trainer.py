from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import cast
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import DINOv2Config, get_config
from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA


class RandomDiscreteRotation:
    def __init__(self, angles: Sequence[int]) -> None:
        if not angles:
            raise ValueError("angles must be non-empty")
        self.angles: list[int] = list(angles)

    def __call__(self, img: Image.Image) -> Image.Image:
        idx = int(torch.randint(0, len(self.angles), (1,)).item())
        return img.rotate(int(self.angles[idx]))


class GaussianNoise:
    def __init__(self, std: float = 0.02) -> None:
        self.std: float = std

    def __call__(self, x: Tensor) -> Tensor:
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std


class MarsDINOCrops:
    def __init__(self, cfg: DINOv2Config) -> None:
        if cfg.aug_use_color_jitter:
            raise ValueError("Color jitter must remain disabled for Mars SSL training.")

        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        common = [
            RandomDiscreteRotation(cfg.aug_rotation_degrees),
            transforms.RandomHorizontalFlip(p=0.5 if cfg.aug_hflip else 0.0),
            transforms.RandomVerticalFlip(p=0.5 if cfg.aug_vflip else 0.0),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            GaussianNoise(std=cfg.aug_gaussian_noise_std),
            normalize,
        ]

        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=cfg.image_size,
                    scale=tuple(cfg.ssl_crop_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=cfg.image_size,
                    scale=(0.05, 0.4),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                *common,
            ]
        )
        self.num_local_crops = 6

    def __call__(self, img: Image.Image) -> list[Tensor]:
        crops = [
            cast(Tensor, self.global_transform(img)),
            cast(Tensor, self.global_transform(img)),
        ]
        for _ in range(self.num_local_crops):
            crops.append(cast(Tensor, self.local_transform(img)))
        return crops


class MarsImageDataset(Dataset[list[Tensor]]):
    def __init__(self, data_dir: Path, transform: MarsDINOCrops) -> None:
        self.data_dir = data_dir
        self.transform = transform
        patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        self.image_paths: list[Path] = []
        for pattern in patterns:
            self.image_paths.extend(sorted(data_dir.rglob(pattern)))

        if not self.image_paths:
            raise FileNotFoundError(f"No JPEG images found under: {data_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> list[Tensor]:
        img_path = self.image_paths[index]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                return self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Failed loading image {img_path}: {exc}") from exc


def dino_collate_fn(batch: list[list[Tensor]]) -> list[Tensor]:
    n_crops = len(batch[0])
    collated: list[Tensor] = []
    for idx in range(n_crops):
        collated.append(torch.stack([sample[idx] for sample in batch], dim=0))
    return collated


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 4096) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        return F.normalize(x, dim=-1)


class DINOTrainer:
    def __init__(self, cfg: DINOv2Config, data_dir: Path, output_dir: Path, epochs: int | None = None) -> None:
        self.cfg = cfg
        self.epochs = epochs if epochs is not None else cfg.ssl_epochs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = torch.cuda.is_available()

        transform = MarsDINOCrops(cfg)
        dataset = MarsImageDataset(data_dir=data_dir, transform=transform)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.ssl_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=dino_collate_fn,
        )
        if len(self.loader) == 0:
            raise RuntimeError("No training steps available. Reduce batch size or add more data.")

        self.student_backbone = DinoV2LoRA(cfg, use_lora=True).to(self.device)
        self.teacher_backbone = copy.deepcopy(self.student_backbone).to(self.device)

        self.student_head = DINOHead(cfg.embed_dim).to(self.device)
        self.teacher_head = copy.deepcopy(self.student_head).to(self.device)

        for module in (self.teacher_backbone, self.teacher_head):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

        self.optimizer = AdamW(self._build_param_groups(), lr=cfg.ssl_lr, weight_decay=0.04)

        total_steps = self.epochs * len(self.loader)
        warmup_steps = max(1, self.cfg.ssl_warmup_epochs * len(self.loader))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        self.student_temp = 0.1
        self.teacher_temp = cfg.ssl_temperature
        self.center_momentum = 0.9
        self.registered_center = torch.zeros(1, 4096, device=self.device)

        self.best_loss = float("inf")

    def _build_param_groups(self) -> list[dict[str, object]]:
        groups: list[dict[str, object]] = []
        layer_count = self.student_backbone.num_layers()

        def layer_scale(layer_idx: int) -> float:
            depth = (layer_count - 1) - layer_idx
            return self.cfg.ssl_layerwise_decay ** depth

        for name, param in self.student_backbone.named_parameters():
            if not param.requires_grad:
                continue

            lr = self.cfg.ssl_lr
            if "encoder.layer." in name:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                lr = self.cfg.ssl_lr * layer_scale(layer_idx)

            groups.append(
                {
                    "params": [param],
                    "lr": lr,
                    "weight_decay": 0.0 if name.endswith("bias") else 0.04,
                }
            )

        groups.append(
            {
                "params": list(self.student_head.parameters()),
                "lr": self.cfg.ssl_lr,
                "weight_decay": 0.04,
            }
        )
        return groups

    @torch.no_grad()
    def _update_teacher(self, momentum: float) -> None:
        for student_p, teacher_p in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_p.data.mul_(momentum).add_(student_p.data * (1.0 - momentum))
        for student_p, teacher_p in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher_p.data.mul_(momentum).add_(student_p.data * (1.0 - momentum))

    @torch.no_grad()
    def _update_center(self, teacher_logits: Tensor) -> None:
        batch_center = torch.mean(teacher_logits, dim=0, keepdim=True)
        self.registered_center = self.registered_center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def _dino_loss(self, student_out: list[Tensor], teacher_out: list[Tensor]) -> Tensor:
        teacher_probs = [F.softmax((x - self.registered_center) / self.teacher_temp, dim=-1).detach() for x in teacher_out]
        student_log_probs = [F.log_softmax(x / self.student_temp, dim=-1) for x in student_out]

        total_loss = torch.tensor(0.0, device=self.device)
        n_terms = 0
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                if s_idx == t_idx:
                    continue
                total_loss += torch.mean(torch.sum(-t_prob * s_log_prob, dim=-1))
                n_terms += 1

        if n_terms == 0:
            raise RuntimeError("No DINO loss terms were computed.")
        return total_loss / n_terms

    def _momentum_at_step(self, step: int, total_steps: int) -> float:
        base = 0.996
        cosine = 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
        return 1.0 - (1.0 - base) * cosine

    def train(self) -> None:
        global_step = 0
        total_steps = self.epochs * len(self.loader)
        momentum = 0.996

        for epoch in range(1, self.epochs + 1):
            self.student_backbone.train()
            self.student_head.train()
            running_loss = 0.0

            for crops in self.loader:
                crops = [c.to(self.device, non_blocking=True) for c in crops]
                teacher_views = crops[:2]

                momentum = self._momentum_at_step(global_step, total_steps)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.mixed_precision):
                    student_embeddings = [self.student_backbone(view) for view in crops]
                    student_logits = [self.student_head(x) for x in student_embeddings]

                    with torch.no_grad():
                        teacher_embeddings = [self.teacher_backbone(view) for view in teacher_views]
                        teacher_logits = [self.teacher_head(x) for x in teacher_embeddings]

                    loss = self._dino_loss(student_logits, teacher_logits)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                with torch.no_grad():
                    cat_teacher = torch.cat(teacher_logits, dim=0)
                    self._update_center(cat_teacher)
                    self._update_teacher(momentum)

                running_loss += loss.item()
                global_step += 1

            epoch_loss = running_loss / len(self.loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{self.epochs} | loss={epoch_loss:.5f} | lr={current_lr:.6e} | ema_m={momentum:.6f}"
            )

            if epoch % 5 == 0:
                self._save_checkpoint(self.output_dir / f"checkpoint_epoch_{epoch}.pt", epoch, epoch_loss)

            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self._save_checkpoint(self.output_dir / "best_model.pt", epoch, epoch_loss)

    def _save_checkpoint(self, path: Path, epoch: int, loss_value: float) -> None:
        state = {
            "epoch": epoch,
            "loss": loss_value,
            "student_backbone": self.student_backbone.state_dict(),
            "student_head": self.student_head.state_dict(),
            "teacher_backbone": self.teacher_backbone.state_dict(),
            "teacher_head": self.teacher_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "center": self.registered_center,
            "config": self.cfg,
        }
        torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised DINOv2+LoRA trainer for Mars tiles")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing JPEG tiles")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {args.data_dir}")

    pipeline_cfg = get_config()
    trainer = DINOTrainer(
        cfg=pipeline_cfg.dinov2,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
