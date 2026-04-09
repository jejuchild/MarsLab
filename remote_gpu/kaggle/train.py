#!/usr/bin/env python3
"""HiRISE → Mastcam-Z Image-to-Image Translation on Kaggle.

Pix2Pix-style U-Net trained on paired (HiRISE, Mastcam) patches.
Predicts Mastcam-Z ortho appearance from HiRISE orbital patches.
"""

import os
import sys
import time
import json
import subprocess

# Force CPU on old GPUs (Kaggle P100 is sm_60, incompatible with current PyTorch)
def _detect_old_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_name = result.stdout.strip()
        old_gpus = ["P100", "P40", "K80", "M60", "K40"]
        return any(g in gpu_name for g in old_gpus)
    except Exception:
        return False

if _detect_old_gpu():
    print("⚠ Detected old GPU — forcing CPU mode")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "experiment": "i2i_unet",
    "data_dir": "/kaggle/input/marsortho-i2i",
    "output_dir": "/kaggle/working",
    "patch_size": 256,
    "batch_size": 8,
    "lr": 2e-4,
    "epochs": 30,
}


# ── Dataset ──────────────────────────────────────────────────────────
class MarsI2IDataset(Dataset):
    """Paired (HiRISE input, Mastcam target) for image-to-image translation."""
    def __init__(self, data_dir, split="train"):
        self.split_dir = Path(data_dir) / split
        self.hirise_files = sorted((self.split_dir / "hirise").glob("*.png"))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.hirise_files)

    def __getitem__(self, idx):
        hir_path = self.hirise_files[idx]
        mas_path = self.split_dir / "mastcam" / hir_path.name

        # HiRISE: grayscale (1 channel)
        hir = Image.open(hir_path).convert("L")
        hir_tensor = self.transform(hir)  # (1, 256, 256)

        # Mastcam: RGB (3 channels)
        mas = Image.open(mas_path).convert("RGB")
        mas_tensor = self.transform(mas)  # (3, 256, 256)

        # Normalize to [-1, 1] for tanh output
        hir_tensor = hir_tensor * 2 - 1
        mas_tensor = mas_tensor * 2 - 1

        return {
            "input": hir_tensor,
            "target": mas_tensor,
            "name": hir_path.stem,
        }


# ── U-Net Generator (Pix2Pix-style) ──────────────────────────────────
class UNetGenerator(nn.Module):
    """Pix2Pix U-Net for HiRISE → Mastcam translation.

    Encoder: 256 → 128 → 64 → 32 → 16 → 8
    Decoder: symmetric with skip connections
    """
    def __init__(self, in_ch=1, out_ch=3, n_feats=64):
        super().__init__()

        # Encoder
        self.e1 = self._down(in_ch, n_feats, batch_norm=False)       # 256 → 128
        self.e2 = self._down(n_feats, n_feats * 2)                    # 128 → 64
        self.e3 = self._down(n_feats * 2, n_feats * 4)                # 64 → 32
        self.e4 = self._down(n_feats * 4, n_feats * 8)                # 32 → 16
        self.e5 = self._down(n_feats * 8, n_feats * 8)                # 16 → 8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(n_feats * 8, n_feats * 8, 4, 2, 1),             # 8 → 4
            nn.ReLU(inplace=True),
        )

        # Decoder (with skip connections)
        self.d1 = self._up(n_feats * 8, n_feats * 8, dropout=True)   # 4 → 8
        self.d2 = self._up(n_feats * 16, n_feats * 8, dropout=True)  # 8 → 16
        self.d3 = self._up(n_feats * 16, n_feats * 4)                 # 16 → 32
        self.d4 = self._up(n_feats * 8, n_feats * 2)                  # 32 → 64
        self.d5 = self._up(n_feats * 4, n_feats)                      # 64 → 128

        # Final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(n_feats * 2, out_ch, 4, 2, 1),         # 128 → 256
            nn.Tanh(),
        )

    def _down(self, in_c, out_c, batch_norm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up(self, in_c, out_c, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.e1(x)    # 128
        e2 = self.e2(e1)   # 64
        e3 = self.e3(e2)   # 32
        e4 = self.e4(e3)   # 16
        e5 = self.e5(e4)   # 8

        b = self.bottleneck(e5)  # 4

        d1 = self.d1(b)                              # 8
        d2 = self.d2(torch.cat([d1, e5], dim=1))     # 16
        d3 = self.d3(torch.cat([d2, e4], dim=1))     # 32
        d4 = self.d4(torch.cat([d3, e3], dim=1))     # 64
        d5 = self.d5(torch.cat([d4, e2], dim=1))     # 128

        return self.final(torch.cat([d5, e1], dim=1))  # 256


# ── Metrics ──────────────────────────────────────────────────────────
def compute_psnr(pred, target):
    """PSNR for tensors in [-1, 1] range."""
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = F.mse_loss(pred_01, target_01)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(1.0 / mse).item()


def compute_ssim(pred, target, window_size=11):
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(pred_01, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target_01, window_size, 1, window_size // 2)
    mu_x_sq, mu_y_sq = mu_x ** 2, mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred_01 ** 2, window_size, 1, window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target_01 ** 2, window_size, 1, window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred_01 * target_01, window_size, 1, window_size // 2) - mu_xy

    ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim.mean().item()


# ── Training ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch in loader:
        x = batch["input"].to(DEVICE)
        y = batch["target"].to(DEVICE)

        pred = model(x)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    psnr_list, ssim_list = [], []
    for batch in loader:
        x = batch["input"].to(DEVICE)
        y = batch["target"].to(DEVICE)
        pred = model(x).clamp(-1, 1)

        for i in range(pred.size(0)):
            psnr_list.append(compute_psnr(pred[i:i+1], y[i:i+1]))
            ssim_list.append(compute_ssim(pred[i:i+1], y[i:i+1]))

    return {
        "psnr": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "ssim": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "n_samples": len(psnr_list),
    }


@torch.no_grad()
def save_samples(model, loader, output_dir, n=5):
    """Save a few prediction samples for visual check."""
    model.eval()
    samples_dir = Path(output_dir) / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)

    saved = 0
    for batch in loader:
        if saved >= n:
            break
        x = batch["input"].to(DEVICE)
        y = batch["target"].to(DEVICE)
        pred = model(x).clamp(-1, 1)

        for i in range(pred.size(0)):
            if saved >= n:
                break
            # Convert [-1,1] → [0,1] → [0,255]
            inp_img = ((x[i, 0] + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
            tgt_img = ((y[i].permute(1, 2, 0) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
            pred_img = ((pred[i].permute(1, 2, 0) + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

            # Side-by-side: input | target | prediction
            inp_rgb = np.stack([inp_img] * 3, axis=-1)
            combined = np.concatenate([inp_rgb, tgt_img, pred_img], axis=1)
            Image.fromarray(combined).save(samples_dir / f"sample_{saved:03d}.png")
            saved += 1


# ── Main ─────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    print(f"\n{'='*60}")
    print(f"HiRISE → Mastcam-Z Image-to-Image Translation")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg["data_dir"])
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    train_ds = MarsI2IDataset(data_dir, "train")
    test_ds = MarsI2IDataset(data_dir, "test")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=1)

    model = UNetGenerator(in_ch=1, out_ch=3, n_feats=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: U-Net Generator ({n_params:,} params)\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))

    history = []
    best_psnr = 0.0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_epoch(model, train_loader, optimizer, epoch)
        elapsed = (time.time() - t0) / 60
        print(f"  Epoch {epoch:3d}/{cfg['epochs']}: loss={loss:.4f}  [{elapsed:.1f}min]")

        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            metrics = evaluate(model, test_loader)
            print(f"           Eval: PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}")

            history.append({"epoch": epoch, "loss": loss, **metrics})

            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save(model.state_dict(), output_dir / "best_model.pth")

    save_samples(model, test_loader, output_dir, n=8)

    elapsed = time.time() - t0
    results = {
        "experiment": cfg["experiment"],
        "model": "unet_pix2pix",
        "device": str(DEVICE),
        "params": n_params,
        "best_psnr": best_psnr,
        "training_time_min": round(elapsed / 60, 1),
        "history": history,
        "config": cfg,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Best PSNR: {best_psnr:.2f}, Time: {elapsed/60:.1f}min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
