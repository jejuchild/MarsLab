#!/usr/bin/env python3
"""MarsRefSR — Kaggle GPU training script.

This script runs on Kaggle's GPU (T4/P100, 30h free/week).
Data is uploaded as a Kaggle dataset, results are saved to /kaggle/working/.

Usage (from local server):
    kaggle kernels push -p remote_gpu/kaggle/
    kaggle kernels status INSERT_USERNAME/marsrefsr-experiment
    kaggle kernels output INSERT_USERNAME/marsrefsr-experiment -p results/
"""

import os
import sys
import time
import json
import subprocess

# ── Install dependencies (skip if already available, e.g. on Kaggle) ──
def _try_install(packages):
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q"] + missing,
                timeout=120,
            )
        except Exception as e:
            print(f"  pip install skipped ({e}), using pre-installed packages")

_try_install(["torch", "torchvision", "timm", "einops", "kornia"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Config ────────────────────────────────────────────────────────────
# Override via environment variables or modify here
CONFIG = {
    "experiment": os.environ.get("EXPERIMENT", "E1_sisr_baseline"),
    "model": os.environ.get("MODEL", "swinir"),
    "data_dir": "/kaggle/working/marsortho-benchmark",
    "output_dir": "/kaggle/working",
    "patch_size": 256,
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 50,
    "scale": 4,
}


# ── Dataset ───────────────────────────────────────────────────────────
class MarsOrthoPatchDataset(Dataset):
    """Paired LR-HR patch dataset for MarsRefSR.

    Expected directory structure:
        data_dir/
            train/
                lr/  (HiRISE 64x64 patches)
                hr/  (Mastcam-Z 256x256 patches)
                dtm/ (DTM condition maps, optional)
                ref/ (Reference Mastcam-Z patches, optional)
            test/
                lr/ hr/ dtm/ ref/
    """
    def __init__(self, data_dir, split="train", scale=4, use_ref=False, use_dtm=False):
        self.split_dir = Path(data_dir) / split
        self.scale = scale
        self.use_ref = use_ref
        self.use_dtm = use_dtm

        lr_dir = self.split_dir / "lr"
        if lr_dir.exists():
            self.lr_files = sorted(lr_dir.glob("*.png"))
        else:
            self.lr_files = []
            print(f"WARNING: {lr_dir} not found")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = self.lr_files[idx]
        stem = lr_path.stem

        lr = Image.open(lr_path).convert("RGB")
        hr_path = self.split_dir / "hr" / f"{stem}.png"
        hr = Image.open(hr_path).convert("RGB") if hr_path.exists() else lr.resize(
            (lr.size[0] * self.scale, lr.size[1] * self.scale), Image.BICUBIC)

        sample = {
            "lr": self.transform(lr),
            "hr": self.transform(hr),
            "name": stem,
        }

        if self.use_ref:
            ref_path = self.split_dir / "ref" / f"{stem}.png"
            if ref_path.exists():
                sample["ref"] = self.transform(Image.open(ref_path).convert("RGB"))

        if self.use_dtm:
            dtm_path = self.split_dir / "dtm" / f"{stem}.npy"
            if dtm_path.exists():
                dtm = np.load(str(dtm_path))  # (H, W, 3): elevation, slope, aspect
                sample["dtm"] = torch.from_numpy(dtm).permute(2, 0, 1).float()

        return sample


# ── Models ────────────────────────────────────────────────────────────
class SimpleSRBaseline(nn.Module):
    """Simple EDSR-like baseline for testing the pipeline."""
    def __init__(self, scale=4, n_feats=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[
            ResBlock(n_feats) for _ in range(n_blocks)
        ])
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        h = self.upsample(h)
        return self.tail(h)


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x):
        return x + self.conv(x) * 0.1


# ── Metrics ───────────────────────────────────────────────────────────
def compute_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(sr, hr, window_size=11):
    """Simplified SSIM computation."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_sr = F.avg_pool2d(sr, window_size, stride=1, padding=window_size//2)
    mu_hr = F.avg_pool2d(hr, window_size, stride=1, padding=window_size//2)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = F.avg_pool2d(sr ** 2, window_size, stride=1, padding=window_size//2) - mu_sr_sq
    sigma_hr_sq = F.avg_pool2d(hr ** 2, window_size, stride=1, padding=window_size//2) - mu_hr_sq
    sigma_sr_hr = F.avg_pool2d(sr * hr, window_size, stride=1, padding=window_size//2) - mu_sr_hr

    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return ssim_map.mean().item()


# ── Training ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(loader):
        lr = batch["lr"].to(DEVICE)
        hr = batch["hr"].to(DEVICE)

        sr = model(lr)

        # Ensure sr matches hr size
        if sr.shape != hr.shape:
            sr = F.interpolate(sr, size=hr.shape[2:], mode="bilinear", align_corners=False)

        loss = F.l1_loss(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    print(f"  Epoch {epoch}: loss={avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_list = []
    ssim_list = []

    for batch in loader:
        lr = batch["lr"].to(DEVICE)
        hr = batch["hr"].to(DEVICE)

        sr = model(lr)
        if sr.shape != hr.shape:
            sr = F.interpolate(sr, size=hr.shape[2:], mode="bilinear", align_corners=False)

        sr = sr.clamp(0, 1)

        for i in range(sr.size(0)):
            psnr_list.append(compute_psnr(sr[i:i+1], hr[i:i+1]))
            ssim_list.append(compute_ssim(sr[i:i+1], hr[i:i+1]))

    return {
        "psnr": np.mean(psnr_list) if psnr_list else 0,
        "ssim": np.mean(ssim_list) if ssim_list else 0,
        "n_samples": len(psnr_list),
    }


# ── Main ──────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    print(f"\n{'='*60}")
    print(f"MarsRefSR Experiment: {cfg['experiment']}")
    print(f"Model: {cfg['model']}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg["data_dir"])

    # Check if data exists
    if not data_dir.exists():
        print(f"Data dir {data_dir} not found.")
        print("Creating synthetic test data for pipeline validation...")

        # Generate synthetic patches for testing
        for split in ["train", "test"]:
            for subdir in ["lr", "hr"]:
                d = data_dir / split / subdir
                d.mkdir(parents=True, exist_ok=True)

                for i in range(20 if split == "train" else 5):
                    if subdir == "lr":
                        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    else:
                        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    Image.fromarray(img).save(d / f"patch_{i:04d}.png")

        print(f"  Created synthetic data in {data_dir}")

    # Load data
    use_ref = "refsr" in cfg["experiment"].lower()
    use_dtm = "dtm" in cfg["experiment"].lower()

    train_ds = MarsOrthoPatchDataset(data_dir, "train", cfg["scale"], use_ref, use_dtm)
    test_ds = MarsOrthoPatchDataset(data_dir, "test", cfg["scale"], use_ref, use_dtm)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    print(f"Train: {len(train_ds)} patches, Test: {len(test_ds)} patches")

    # Create model
    if cfg["model"] == "simple_baseline":
        model = SimpleSRBaseline(scale=cfg["scale"]).to(DEVICE)
    else:
        # For other models (swinir, hat, etc.), load pretrained
        # This is the simple baseline for pipeline testing
        model = SimpleSRBaseline(scale=cfg["scale"]).to(DEVICE)
        print(f"  Using SimpleSRBaseline (replace with {cfg['model']} for full experiment)")

    params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    # Train
    best_psnr = 0
    history = []

    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            metrics = evaluate(model, test_loader)
            print(f"  Eval: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

            history.append({
                "epoch": epoch,
                "loss": loss,
                **metrics,
            })

            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save(model.state_dict(), output_dir / "best_model.pth")

    elapsed = time.time() - t0

    # Save results
    results = {
        "experiment": cfg["experiment"],
        "model": cfg["model"],
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": params,
        "best_psnr": best_psnr,
        "training_time_min": round(elapsed / 60, 1),
        "history": history,
        "config": cfg,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Best PSNR: {best_psnr:.2f}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
