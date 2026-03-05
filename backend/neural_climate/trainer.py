"""
Mars GCM Neural Emulator — Training Pipeline.

Trains the MLP emulator on data generated from the parametric climate model.
Features:
    - Cosine annealing learning rate schedule
    - Early stopping with patience
    - Per-output MSE tracking
    - Model checkpointing (best validation loss)
    - Normalization stats saved alongside model weights

Usage:
    python -m neural_climate.trainer [--epochs 200] [--hidden 256] [--lr 1e-3]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from neural_climate.model import (
    MarsClimateEmulator,
    create_model,
    OUTPUT_NAMES,
    INPUT_DIM,
    OUTPUT_DIM,
)
from neural_climate.dataset import (
    generate_training_data,
    create_dataloaders,
    save_dataset,
    load_dataset,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_MODULE_DIR, "checkpoints")
_BEST_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "best_model.pt")
_NORM_STATS_PATH = os.path.join(_CHECKPOINT_DIR, "norm_stats.npz")
_TRAINING_LOG_PATH = os.path.join(_CHECKPOINT_DIR, "training_log.json")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Trainer:
    """Handles training, validation, checkpointing, and logging."""

    def __init__(
        self,
        model: MarsClimateEmulator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        norm_stats: dict,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.norm_stats = norm_stats
        self.device = device

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.best_val_loss = float("inf")
        self.training_log = []

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, Y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, Y_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> tuple[float, dict]:
        """
        Run validation. Returns (mean_loss, per_output_mse_dict).

        Per-output MSE is computed in **physical units** (denormalized).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Accumulate per-output squared errors
        target_mean = torch.from_numpy(self.norm_stats["target_mean"]).float().to(self.device)
        target_std = torch.from_numpy(self.norm_stats["target_std"]).float().to(self.device)
        per_output_se = torch.zeros(OUTPUT_DIM, device=self.device)
        n_samples = 0

        for X_batch, Y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            predictions = self.model(X_batch)
            loss = self.criterion(predictions, Y_batch)
            total_loss += loss.item()
            n_batches += 1

            # Denormalize for per-output physical MSE
            pred_phys = predictions * target_std + target_mean
            true_phys = Y_batch * target_std + target_mean
            per_output_se += ((pred_phys - true_phys) ** 2).sum(dim=0)
            n_samples += X_batch.shape[0]

        mean_loss = total_loss / max(n_batches, 1)
        per_output_mse = (per_output_se / max(n_samples, 1)).cpu().numpy()

        per_output_dict = {
            name: float(per_output_mse[i])
            for i, name in enumerate(OUTPUT_NAMES)
        }

        return mean_loss, per_output_dict

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model weights + norm stats."""
        path = path or _BEST_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "hidden_dim": self.model.input_proj[0].in_features
                    if hasattr(self.model.input_proj[0], "in_features")
                    else 256,
                "n_blocks": len(self.model.blocks),
            },
            "best_val_loss": self.best_val_loss,
        }

        # Get hidden_dim from first linear layer
        for m in self.model.input_proj:
            if isinstance(m, nn.Linear):
                checkpoint["model_config"]["hidden_dim"] = m.out_features
                break

        torch.save(checkpoint, path)

        # Save normalization stats separately for easy loading
        norm_path = os.path.join(os.path.dirname(path), "norm_stats.npz")
        np.savez(
            norm_path,
            target_mean=self.norm_stats["target_mean"],
            target_std=self.norm_stats["target_std"],
        )

        logger.info(f"Checkpoint saved to {path}")

    def save_training_log(self, path: Optional[str] = None):
        """Save training log as JSON."""
        path = path or _TRAINING_LOG_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)

    def fit(
        self,
        epochs: int = 200,
        patience: int = 20,
        log_every: int = 10,
    ) -> dict:
        """
        Full training loop with early stopping.

        Args:
            epochs: maximum number of epochs
            patience: early stopping patience (epochs without improvement)
            log_every: print progress every N epochs

        Returns:
            dict with final metrics and training history summary
        """
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        no_improve = 0
        t_start = time.time()

        logger.info(
            f"Training {self.model.param_count:,} parameters "
            f"for up to {epochs} epochs (patience={patience})"
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, per_output_mse = self.validate()
            scheduler.step()

            lr = float(scheduler.get_last_lr()[0])

            entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": round(lr, 8),
                "per_output_mse": {k: round(v, 4) for k, v in per_output_mse.items()},
            }
            self.training_log.append(entry)

            # Check improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improve = 0
                self.save_checkpoint()
            else:
                no_improve += 1

            # Logging
            if epoch % log_every == 0 or epoch == 1 or no_improve == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train={train_loss:.6f} val={val_loss:.6f} | "
                    f"lr={lr:.2e} | best={self.best_val_loss:.6f} | "
                    f"no_improve={no_improve} | {elapsed:.0f}s"
                )

                # Per-output detail at milestones
                if epoch % (log_every * 5) == 0 or epoch == 1:
                    for name, mse in per_output_mse.items():
                        rmse = mse ** 0.5
                        logger.info(f"  {name:25s} RMSE={rmse:.4f}")

            # Early stopping
            if no_improve >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        elapsed_total = time.time() - t_start
        self.save_training_log()

        # Final validation with best model
        best_ckpt = torch.load(_BEST_MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(best_ckpt["model_state_dict"])
        final_val_loss, final_per_output = self.validate()

        result = {
            "epochs_trained": len(self.training_log),
            "best_val_loss": round(self.best_val_loss, 6),
            "final_val_loss": round(final_val_loss, 6),
            "training_time_s": round(elapsed_total, 1),
            "param_count": self.model.param_count,
            "per_output_rmse": {
                k: round(v ** 0.5, 4) for k, v in final_per_output.items()
            },
        }

        logger.info(f"Training complete: {result}")
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Command-line training interface."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train Mars GCM Neural Emulator")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--blocks", type=int, default=4, help="Number of residual blocks")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lat-step", type=float, default=5.0, help="Latitude grid step")
    parser.add_argument("--lon-step", type=float, default=10.0, help="Longitude grid step")
    parser.add_argument("--ls-step", type=float, default=10.0, help="Ls grid step")
    parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    parser.add_argument("--regenerate", action="store_true", help="Force regenerate dataset")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Generate or load dataset
    cached = None if args.regenerate else load_dataset()
    if cached is not None:
        inputs, targets, encoded, meta = cached
    else:
        logger.info("Generating training data from parametric model...")
        inputs, targets, encoded, meta = generate_training_data(
            lat_step=args.lat_step,
            lon_step=args.lon_step,
            ls_step=args.ls_step,
        )
        save_dataset(inputs, targets, encoded, meta)

    logger.info(f"Dataset: {len(inputs)} samples, {encoded.shape[1]} features → {targets.shape[1]} targets")

    # Create dataloaders
    train_loader, val_loader, norm_stats = create_dataloaders(
        encoded, targets,
        batch_size=args.batch_size,
    )

    # Create model
    model = create_model(
        hidden_dim=args.hidden,
        n_blocks=args.blocks,
    )
    logger.info(f"Model: {model.param_count:,} parameters")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        norm_stats=norm_stats,
        lr=args.lr,
        device=device,
    )

    result = trainer.fit(
        epochs=args.epochs,
        patience=args.patience,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for k, v in result.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    print(f"\nModel saved to: {_BEST_MODEL_PATH}")
    print(f"Norm stats at:  {_NORM_STATS_PATH}")


if __name__ == "__main__":
    main()
