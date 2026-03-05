from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import torch
import torch.optim as optim

from .forward import generate_synthetic_data
from .mars_model import get_reference_model
from .pinn_model import MarsInteriorPINN, compute_boundary_loss, compute_physics_loss

logger = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_DIR = os.path.join(_MODULE_DIR, "checkpoints")
_BEST_MODEL_PATH = os.path.join(_CHECKPOINT_DIR, "best_pinn_model.pt")


@dataclass
class TrainingResult:
    best_loss: float
    final_loss: float
    epochs: int
    checkpoint_path: str


class PINNTrainer:
    def __init__(
        self,
        model: MarsInteriorPINN,
        observed_distances: torch.Tensor,
        observed_times: torch.Tensor,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        bc_weight: float = 1.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.distances = observed_distances.float().to(device)
        self.times = observed_times.float().to(device)
        self.data_weight = float(data_weight)
        self.physics_weight = float(physics_weight)
        self.bc_weight = float(bc_weight)
        self.device = device
        self.best_loss = float("inf")

    def fit(self, epochs: int = 2000, lr: float = 1e-3, log_every: int = 100) -> TrainingResult:
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        final_loss = float("inf")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            pred_times = self.model.predict_travel_times(self.distances)
            data_loss = torch.mean((pred_times - self.times) ** 2)
            physics_loss = compute_physics_loss(self.model)
            bc_loss = compute_boundary_loss(self.model)

            loss = self.data_weight * data_loss + self.physics_weight * physics_loss + self.bc_weight * bc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            final_loss = float(loss.item())
            if final_loss < self.best_loss:
                self.best_loss = final_loss
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "hidden_dim": 64,
                        "best_loss": self.best_loss,
                    },
                    _BEST_MODEL_PATH,
                )

            if epoch == 1 or epoch % log_every == 0:
                logger.info(
                    "epoch=%d total=%.6f data=%.6f physics=%.6f bc=%.6f lr=%.2e",
                    epoch,
                    final_loss,
                    float(data_loss.item()),
                    float(physics_loss.item()),
                    float(bc_loss.item()),
                    float(scheduler.get_last_lr()[0]),
                )

        return TrainingResult(
            best_loss=self.best_loss,
            final_loss=final_loss,
            epochs=epochs,
            checkpoint_path=_BEST_MODEL_PATH,
        )


def train_default(epochs: int = 1200, lr: float = 1e-3, seed: int = 7) -> TrainingResult:
    torch.manual_seed(seed)
    ref = get_reference_model(n_layers=160)
    distances, travel_times = generate_synthetic_data(ref, n_events=20, seed=seed)

    model = MarsInteriorPINN(hidden_dim=64)
    trainer = PINNTrainer(
        model=model,
        observed_distances=torch.tensor(distances, dtype=torch.float32),
        observed_times=torch.tensor(travel_times, dtype=torch.float32),
        data_weight=1.0,
        physics_weight=0.1,
        bc_weight=1.0,
    )
    return trainer.fit(epochs=epochs, lr=lr)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train Mars interior PINN")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = train_default(epochs=args.epochs, lr=args.lr, seed=args.seed)
    print(
        {
            "best_loss": round(result.best_loss, 6),
            "final_loss": round(result.final_loss, 6),
            "epochs": result.epochs,
            "checkpoint_path": result.checkpoint_path,
        }
    )


if __name__ == "__main__":
    main()
