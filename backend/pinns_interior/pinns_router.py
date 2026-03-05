from __future__ import annotations

import logging
from typing import Any

import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .mars_model import get_reference_model
from .predictor import get_predictor, is_model_trained
from .trainer import train_default

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pinns", tags=["Mars PINNs Interior"])


class PredictRequest(BaseModel):
    depths_km: list[float] = Field(..., min_length=1, max_length=2000)


class TrainRequest(BaseModel):
    epochs: int = Field(1200, ge=100, le=5000)
    lr: float = Field(1e-3, ge=1e-5, le=1e-1)
    seed: int = Field(7, ge=0, le=99999)


class ProfileRequest(BaseModel):
    n_points: int = Field(200, ge=20, le=2000)


@router.get("/status")
async def status() -> dict[str, Any]:
    trained = is_model_trained()
    return {
        "status": "ready" if trained else "not_trained",
        "trained": trained,
    }


@router.post("/predict")
async def predict(req: PredictRequest) -> dict[str, Any]:
    if not is_model_trained():
        raise HTTPException(status_code=503, detail="PINNs model not trained. POST /api/pinns/train first.")
    predictor = get_predictor()
    vp = predictor.predict_velocity(np.asarray(req.depths_km, dtype=np.float32))
    return {
        "depths_km": req.depths_km,
        "vp_km_s": vp.tolist(),
        "source": "pinns_interior",
    }


@router.post("/profile")
async def profile(req: ProfileRequest) -> dict[str, Any]:
    if not is_model_trained():
        raise HTTPException(status_code=503, detail="PINNs model not trained. POST /api/pinns/train first.")
    predictor = get_predictor()
    result = predictor.profile(n_points=req.n_points)
    result["source"] = "pinns_interior"
    return result


@router.post("/train")
async def train(req: TrainRequest) -> dict[str, Any]:
    try:
        result = train_default(epochs=req.epochs, lr=req.lr, seed=req.seed)
        from . import predictor as pred_module

        pred_module._predictor_instance = None
        return {
            "status": "trained",
            "best_loss": result.best_loss,
            "final_loss": result.final_loss,
            "epochs": result.epochs,
            "checkpoint_path": result.checkpoint_path,
        }
    except Exception as exc:
        logger.error("PINNs training failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@router.post("/compare")
async def compare(req: ProfileRequest) -> dict[str, Any]:
    if not is_model_trained():
        raise HTTPException(status_code=503, detail="PINNs model not trained. POST /api/pinns/train first.")
    predictor = get_predictor()
    result = predictor.compare_with_reference(n_points=req.n_points)
    result["source"] = "pinns_interior"
    return result


@router.get("/reference-model")
async def reference_model() -> dict[str, Any]:
    radius, vp, vs, density = get_reference_model(n_layers=160)
    return {
        "radius_km": radius.tolist(),
        "depth_km": (3389.5 - radius).tolist(),
        "vp_km_s": vp.tolist(),
        "vs_km_s": vs.tolist(),
        "density_g_cm3": density.tolist(),
    }
