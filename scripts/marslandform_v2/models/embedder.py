from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.marslandform_v2.config import get_config
from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA


class MarsTileDataset(Dataset[tuple[Tensor, str]]):
    def __init__(self, image_dir: Path, image_size: int) -> None:
        self.image_dir = image_dir
        patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        self.image_paths: list[Path] = []
        for pattern in patterns:
            self.image_paths.extend(sorted(image_dir.rglob(pattern)))

        if not self.image_paths:
            raise FileNotFoundError(f"No JPEG images found under: {image_dir}")

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.image_paths[index]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor_img = cast(Tensor, self.transform(img))
                return tensor_img, str(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to process image {path}: {exc}") from exc


def load_model(model_path: Path | None) -> DinoV2LoRA:
    cfg = get_config().dinov2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model = DinoV2LoRA(cfg, use_lora=False)
        model.eval()
        return model.to(device)

    if not model_path.exists():
        raise FileNotFoundError(f"model_path does not exist: {model_path}")

    if model_path.is_dir():
        model = DinoV2LoRA(cfg, use_lora=True, lora_weights_path=model_path)
    else:
        if model_path.suffix not in {".pt", ".pth"}:
            raise ValueError("model_path must be a PEFT directory or a .pt/.pth checkpoint")
        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint = cast(dict[str, Any], checkpoint)
        if "student_backbone" not in checkpoint:
            raise KeyError("Checkpoint missing 'student_backbone' state_dict")
        model = DinoV2LoRA(cfg, use_lora=True)
        model.load_state_dict(checkpoint["student_backbone"], strict=False)

    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_embeddings(
    model: DinoV2LoRA,
    image_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config().dinov2
    dataset = MarsTileDataset(image_dir=image_dir, image_size=cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []

    for images, paths in tqdm(loader, desc="Extracting embeddings", total=len(loader)):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            emb = model(images)
        all_embeddings.append(emb.cpu().numpy())
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0)
    if embeddings.shape[1] != cfg.embed_dim:
        raise RuntimeError(f"Expected embedding dim {cfg.embed_dim}, got {embeddings.shape[1]}")

    npy_path = output_dir / "embeddings.npy"
    csv_path = output_dir / "metadata.csv"

    np.save(npy_path, embeddings)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_path"])
        for idx, p in enumerate(all_paths):
            writer.writerow([idx, p])

    return npy_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch embedding extraction for Mars DINOv2")
    parser.add_argument("--model_path", type=Path, default=None, help="LoRA dir or checkpoint .pt/.pth")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing JPEG tiles")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for embeddings outputs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {args.image_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    else:
        print("GPU not available, running on CPU.")

    model = load_model(args.model_path)
    npy_path, csv_path = extract_embeddings(
        model=model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(f"Saved embeddings: {npy_path}")
    print(f"Saved metadata: {csv_path}")


if __name__ == "__main__":
    main()
