"""
FastAPI router for Mars GCM Neural Emulator endpoints.

POST /api/climate/neural/predict       — Single-point prediction
POST /api/climate/neural/predict/batch  — Batch prediction
POST /api/climate/neural/compare       — Compare neural vs parametric
POST /api/climate/neural/map           — Global climate map for given Ls
GET  /api/climate/neural/status        — Model status and info
POST /api/climate/neural/train         — Trigger model training
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/climate/neural", tags=["Neural Climate"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(..., ge=-180, le=360, description="Longitude in degrees")
    ls: float = Field(..., ge=0, le=360, description="Solar longitude (Ls) in degrees")
    elevation: Optional[float] = Field(
        None, description="Elevation in meters. If null, looked up from MOLA DEM."
    )


class PredictResponse(BaseModel):
    lat: float
    lon: float
    ls: float
    elevation_m: float
    temperature_mean_k: float
    temperature_max_k: float
    temperature_min_k: float
    pressure_pa: float
    dust_tau_mean: float
    wind_mean_ms: float
    frost_probability: float
    source: str


class BatchPredictRequest(BaseModel):
    points: List[PredictRequest] = Field(
        ..., min_length=1, max_length=10000,
        description="List of (lat, lon, Ls) points to predict"
    )


class CompareRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=360)
    ls: float = Field(..., ge=0, le=360)


class GlobalMapRequest(BaseModel):
    ls: float = Field(..., ge=0, le=360, description="Solar longitude for map")
    variable: str = Field(
        "temperature_mean_k",
        description="Climate variable to map"
    )
    lat_step: float = Field(2.0, ge=0.5, le=10.0, description="Latitude resolution")
    lon_step: float = Field(2.0, ge=0.5, le=10.0, description="Longitude resolution")


class TrainRequest(BaseModel):
    epochs: int = Field(200, ge=10, le=1000, description="Max training epochs")
    hidden_dim: int = Field(256, ge=64, le=1024, description="Hidden layer size")
    n_blocks: int = Field(4, ge=2, le=8, description="Number of residual blocks")
    lr: float = Field(1e-3, ge=1e-5, le=1e-1, description="Learning rate")
    lat_step: float = Field(5.0, ge=1.0, le=30.0, description="Training grid lat step")
    lon_step: float = Field(10.0, ge=1.0, le=30.0, description="Training grid lon step")
    ls_step: float = Field(10.0, ge=1.0, le=30.0, description="Training grid Ls step")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_model_status():
    """Check if neural emulator is trained and ready."""
    from neural_climate.predictor import is_model_trained

    trained = is_model_trained()

    if trained:
        try:
            from neural_climate.predictor import get_predictor
            predictor = get_predictor()
            info = predictor.model_info
            return {
                "status": "ready",
                "trained": True,
                "model_info": info,
            }
        except Exception as e:
            return {
                "status": "error",
                "trained": True,
                "error": str(e),
            }
    else:
        return {
            "status": "not_trained",
            "trained": False,
            "message": "Model has not been trained yet. POST /api/climate/neural/train to start training.",
        }


@router.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest):
    """Predict climate for a single (lat, lon, Ls) point using the neural emulator."""
    from neural_climate.predictor import is_model_trained, get_predictor

    if not is_model_trained():
        raise HTTPException(
            status_code=503,
            detail="Neural emulator not trained yet. POST /api/climate/neural/train first.",
        )

    try:
        predictor = get_predictor()
        result = predictor.predict(
            lat=req.lat, lon=req.lon, ls=req.ls,
            elevation=req.elevation,
        )
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Neural prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    """Batch prediction for multiple points."""
    import numpy as np
    from neural_climate.predictor import is_model_trained, get_predictor

    if not is_model_trained():
        raise HTTPException(
            status_code=503,
            detail="Neural emulator not trained yet.",
        )

    try:
        predictor = get_predictor()
        lats = np.array([p.lat for p in req.points])
        lons = np.array([p.lon for p in req.points])
        ls_vals = np.array([p.ls for p in req.points])
        elevations = None
        if all(p.elevation is not None for p in req.points):
            elevations = np.array([p.elevation for p in req.points])

        results = predictor.predict_batch(lats, lons, ls_vals, elevations)

        return {
            "predictions": results,
            "count": len(results),
            "source": "neural_emulator",
        }
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/compare")
async def compare_models(req: CompareRequest):
    """Compare neural emulator output with parametric model."""
    from neural_climate.predictor import is_model_trained, get_predictor

    if not is_model_trained():
        raise HTTPException(
            status_code=503,
            detail="Neural emulator not trained yet.",
        )

    try:
        predictor = get_predictor()
        comparison = predictor.compare_with_parametric(
            lat=req.lat, lon=req.lon, ls=req.ls,
        )
        return comparison
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/map")
async def generate_global_map(req: GlobalMapRequest):
    """Generate a global climate map for a given Ls and variable."""
    from neural_climate.predictor import is_model_trained, get_predictor
    from neural_climate.model import OUTPUT_NAMES

    if not is_model_trained():
        raise HTTPException(
            status_code=503,
            detail="Neural emulator not trained yet.",
        )

    if req.variable not in OUTPUT_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown variable '{req.variable}'. Choose from: {OUTPUT_NAMES}",
        )

    try:
        predictor = get_predictor()
        result = predictor.predict_global_map(
            ls=req.ls,
            lat_step=req.lat_step,
            lon_step=req.lon_step,
            output_var=req.variable,
        )
        return result
    except Exception as e:
        logger.error(f"Map generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Map generation failed: {str(e)}")


@router.post("/train")
async def train_model(req: TrainRequest):
    """
    Trigger model training (synchronous — may take several minutes).

    For production use, consider running training offline via CLI:
        python -m neural_climate.trainer --epochs 200
    """
    import torch
    from neural_climate.model import create_model
    from neural_climate.dataset import (
        generate_training_data,
        create_dataloaders,
        save_dataset,
        load_dataset,
    )
    from neural_climate.trainer import Trainer

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate or load dataset
        cached = load_dataset()
        if cached is not None:
            inputs, targets, encoded, meta = cached
        else:
            inputs, targets, encoded, meta = generate_training_data(
                lat_step=req.lat_step,
                lon_step=req.lon_step,
                ls_step=req.ls_step,
            )
            save_dataset(inputs, targets, encoded, meta)

        # Create dataloaders
        train_loader, val_loader, norm_stats = create_dataloaders(
            encoded, targets, batch_size=512,
        )

        # Create model
        model = create_model(
            hidden_dim=req.hidden_dim,
            n_blocks=req.n_blocks,
        )

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            norm_stats=norm_stats,
            lr=req.lr,
            device=device,
        )

        result = trainer.fit(epochs=req.epochs, patience=20)

        # Reset singleton predictor so next call loads new model
        import neural_climate.predictor as pred_module
        pred_module._predictor_instance = None

        return {
            "status": "trained",
            "device": device,
            **result,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
