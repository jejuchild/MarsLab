"""
HiRISE Landform Classification Pipeline — Real DINOv2 + MIL + VLM Agent.

Inference flow:
  1. Fetch HiRISE browse image → tile into 224×224 patches (filter black tiles)
  2. Transform tiles: Resize(224) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
  3. Run tiles through DINOv2-LoRA backbone → 768-dim CLS embedding per tile
  4. Extract MOLA features at image center (23 geomorphometric features)
  5. Feed into AttentionMILClassifier → logits + attention weights
  6. If confidence < 0.7 → invoke VLM ReACT agent for enhanced reasoning
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .heatmap import generate_heatmap, save_heatmap
from .models import (
    AgentReasoningResult,
    AgentReasoningStep,
    ClassifyRequest,
    ClassifyResult,
    PredictionResult,
    TileResult,
)
from .preprocessing import extract_mola_features, fetch_hirise_browse, tile_image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

# ── Paths ─────────────────────────────────────────────────────────────────────
MIL_CHECKPOINT = ROOT / "Data" / "HiRISE" / "v2_output" / "models" / "multihead_improved" / "best_mil_model.pt"
SSL_LORA_WEIGHTS = ROOT / "Data" / "HiRISE" / "v2_output" / "ssl_lora_weights" / "best_model.pt"

V2_CLASSES = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
CONFIDENCE_THRESHOLD = 0.7  # Below this → invoke VLM agent


# ── Image transform (ImageNet normalization, matching training) ───────────────
_tile_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


@dataclass(frozen=True)
class LoadedModels:
    """Container for lazy-loaded inference models."""
    backbone: Any  # DinoV2LoRA
    classifier: Any  # AttentionMILClassifier
    device: torch.device


class HiriseLandformPipeline:
    """Real MarsLandformNet V2 inference pipeline."""

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: LoadedModels | None = None
        self._loading = False

    @property
    def device(self) -> str:
        return str(self._device)

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _ensure_models(self) -> LoadedModels:
        """Lazy-load DINOv2-LoRA backbone + AttentionMILClassifier on first request."""
        if self._models is not None:
            return self._models

        if self._loading:
            raise RuntimeError("Model loading already in progress")

        self._loading = True
        try:
            models = self._load_models()
            self._models = models
            return models
        finally:
            self._loading = False

    def _load_models(self) -> LoadedModels:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from scripts.marslandform_v2.config import DINOv2Config, MILConfig
        from scripts.marslandform_v2.models.dinov2_lora import DinoV2LoRA
        from scripts.marslandform_v2.models.mil_classifier import AttentionMILClassifier

        device = self._device
        logger.info("Loading DINOv2-LoRA backbone on %s...", device)

        # Load DINOv2-LoRA backbone
        dinov2_cfg = DINOv2Config()
        backbone = DinoV2LoRA(dinov2_cfg, use_lora=True)

        # Load SSL LoRA weights if available
        if SSL_LORA_WEIGHTS.exists():
            logger.info("Loading SSL LoRA weights from %s", SSL_LORA_WEIGHTS)
            checkpoint = torch.load(SSL_LORA_WEIGHTS, map_location="cpu", weights_only=False)
            if "student_backbone" in checkpoint:
                backbone.load_state_dict(checkpoint["student_backbone"], strict=False)
                logger.info("Loaded SSL LoRA weights (student_backbone)")
            else:
                logger.warning("SSL checkpoint missing 'student_backbone' key, skipping")
        else:
            logger.warning("SSL LoRA weights not found at %s, using base DINOv2", SSL_LORA_WEIGHTS)

        backbone.eval()
        backbone = backbone.to(device)

        # Load MIL classifier
        if not MIL_CHECKPOINT.exists():
            raise FileNotFoundError(f"MIL checkpoint not found: {MIL_CHECKPOINT}")

        logger.info("Loading AttentionMILClassifier from %s", MIL_CHECKPOINT)
        mil_ckpt = torch.load(MIL_CHECKPOINT, map_location="cpu", weights_only=False)

        # Reconstruct MILConfig from checkpoint
        saved_config = mil_ckpt.get("mil_config", {})
        mil_cfg = MILConfig(
            embed_dim=saved_config.get("embed_dim", 768),
            mola_dim=saved_config.get("mola_dim", 23),
            hidden_dim=saved_config.get("hidden_dim", 256),
            attention_dim=saved_config.get("attention_dim", 128),
            num_attention_heads=saved_config.get("num_attention_heads", 4),
            num_classes=saved_config.get("num_classes", 5),
            dropout=saved_config.get("dropout", 0.3),
        )

        classifier = AttentionMILClassifier(mil_cfg)
        classifier.load_state_dict(mil_ckpt["model_state_dict"])
        classifier.eval()
        classifier = classifier.to(device)

        logger.info(
            "Models loaded — backbone: DINOv2-LoRA (768-dim), "
            "classifier: AttentionMIL (F1=%.4f, epoch=%d)",
            mil_ckpt.get("best_landform_macro_f1", 0.0),
            mil_ckpt.get("best_epoch", -1),
        )

        return LoadedModels(backbone=backbone, classifier=classifier, device=device)

    # ── Main classify entry point ─────────────────────────────────────────────

    def classify(self, request: ClassifyRequest) -> ClassifyResult:
        """Full inference: tiling → embedding → MIL → (optional VLM agent)."""
        t0 = time.time()

        models = self._ensure_models()

        # 1. Fetch browse image
        image = fetch_hirise_browse(request.product_id)
        img_w, img_h = image.size

        # 2. Tile with content filtering
        tiled = tile_image(image, tile_size=224, min_content=0.3)
        coords = [(x, y) for x, y, _ in tiled]
        tile_images = [tile for _, _, tile in tiled]
        num_tiles = len(tile_images)

        if num_tiles == 0:
            return self._empty_result(request, time.time() - t0)

        # 3. Compute tile lat/lon from image center
        lat = request.lat
        lon = request.lon
        tile_lat_lon = self._compute_tile_latlon(coords, img_w, img_h, lat, lon)

        # 4. Extract tile embeddings through DINOv2-LoRA
        embeddings = self._extract_embeddings(models, tile_images)

        # 5. Extract MOLA features at image center
        if lat is not None and lon is not None:
            mola_features = extract_mola_features(lat, lon)
        else:
            # Use approximate center — mid-latitude Mars as fallback
            mola_features = np.zeros((23,), dtype=np.float32)

        # 6. Run MIL classifier
        probabilities, attention_weights = self._run_mil_classifier(
            models, embeddings, mola_features
        )

        # 7. Build prediction
        top_idx = int(np.argmax(probabilities))
        top_class = V2_CLASSES[top_idx]
        confidence = float(probabilities[top_idx])

        prediction = PredictionResult(
            top_class=top_class,
            probabilities={name: float(probabilities[i]) for i, name in enumerate(V2_CLASSES)},
            confidence=confidence,
        )

        # 8. Build tile results
        tile_results = self._build_tile_results(coords, tile_lat_lon, attention_weights)

        # 9. Generate heatmap if requested
        heatmap_url: str | None = None
        if request.include_heatmap:
            heatmap = generate_heatmap(image=image, tiles=tile_results, tile_size=224)
            heatmap_url = save_heatmap(heatmap, request.product_id)

        # 10. VLM Agent — invoke if confidence is below threshold
        agent_reasoning: AgentReasoningResult | None = None
        if confidence < CONFIDENCE_THRESHOLD:
            agent_reasoning = self._run_vlm_agent(
                request.product_id,
                top_class,
                confidence,
                probabilities,
                attention_weights,
                mola_features,
                lat,
                lon,
            )
            # If agent provides a higher-confidence answer, use it
            if (
                agent_reasoning is not None
                and agent_reasoning.enabled
                and agent_reasoning.confidence is not None
                and agent_reasoning.confidence > confidence
                and agent_reasoning.landform_class is not None
                and agent_reasoning.landform_class in V2_CLASSES
            ):
                top_class = agent_reasoning.landform_class
                confidence = agent_reasoning.confidence
                prediction = PredictionResult(
                    top_class=top_class,
                    probabilities=prediction.probabilities,
                    confidence=confidence,
                )

        processing_time = time.time() - t0

        return ClassifyResult(
            product_id=request.product_id,
            model_used="v2",
            prediction=prediction,
            tiles=tile_results,
            heatmap_url=heatmap_url,
            processing_time_s=round(processing_time, 2),
            agent_reasoning=agent_reasoning,
            num_tiles=num_tiles,
            device=str(self._device),
        )

    # ── Embedding extraction ──────────────────────────────────────────────────

    @torch.no_grad()
    def _extract_embeddings(
        self, models: LoadedModels, tile_images: list[Image.Image],
    ) -> np.ndarray:
        """
        Run tiles through DINOv2-LoRA backbone → (num_tiles, 768) embeddings.
        Processes in batches to manage memory.
        """
        device = models.device
        backbone = models.backbone
        batch_size = 32

        tensors = []
        for tile in tile_images:
            rgb = tile.convert("RGB")
            tensor = _tile_transform(rgb)
            tensors.append(tensor)

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                emb = backbone(batch)  # (batch, 768)
            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)  # (num_tiles, 768)

    # ── MIL classifier ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_mil_classifier(
        self,
        models: LoadedModels,
        tile_embeddings: np.ndarray,
        mola_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run AttentionMILClassifier.
        Returns: (probabilities [5], attention_weights [num_tiles])
        """
        device = models.device
        classifier = models.classifier

        # Prepare tensors: (1, num_tiles, 768), (1, num_tiles) mask, (1, 23) mola
        num_tiles = tile_embeddings.shape[0]
        tile_emb_t = torch.from_numpy(tile_embeddings).unsqueeze(0).float().to(device)
        tile_mask_t = torch.ones(1, num_tiles, dtype=torch.bool, device=device)
        mola_t = torch.from_numpy(mola_features).unsqueeze(0).float().to(device)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits, att_weights = classifier(tile_emb_t, tile_mask_t, mola_t)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (5,)
        att = att_weights.squeeze(0).cpu().numpy()  # (num_tiles,)

        return probs, att

    # ── VLM Agent ─────────────────────────────────────────────────────────────

    def _run_vlm_agent(
        self,
        product_id: str,
        classifier_class: str,
        classifier_confidence: float,
        probabilities: np.ndarray,
        attention_weights: np.ndarray,
        mola_features: np.ndarray,
        lat: float | None,
        lon: float | None,
    ) -> AgentReasoningResult:
        """
        Invoke VLM ReACT agent for low-confidence classifications.
        Uses the agent framework from scripts/marslandform_v2/agent/.
        """
        try:
            import sys
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from scripts.marslandform_v2.config import AgentConfig
            from scripts.marslandform_v2.agent.react_agent import MarsLandformAgent

            # Build a lightweight classifier proxy for the agent's classify tool
            classifier_proxy = _ClassifierProxy(
                product_id=product_id,
                predicted_class=classifier_class,
                confidence=classifier_confidence,
                probabilities=probabilities.tolist(),
                attention_weights=attention_weights.tolist(),
            )

            # Build MOLA features dict for analyze_mola tool
            mola_dict: dict[str, dict[str, Any]] = {}
            if lat is not None and lon is not None:
                feature_names = []
                for scale in [1.0, 5.0, 20.0]:
                    for feat in ["slope_mean", "slope_std", "curvature_mean", "TPI", "TRI", "roughness", "lobateness"]:
                        feature_names.append(f"{feat}_{scale}km")
                feature_names.extend(["elevation_mean", "abs_latitude"])

                mola_entry: dict[str, Any] = {}
                for i, name in enumerate(feature_names):
                    if i < len(mola_features):
                        mola_entry[name] = float(mola_features[i])
                mola_entry["lat"] = lat
                mola_entry["lon"] = lon
                mola_dict[product_id] = mola_entry

            agent_cfg = AgentConfig(
                max_steps=3,  # Limit for API responsiveness
                confidence_threshold=CONFIDENCE_THRESHOLD,
                mode="agent",
            )

            agent = MarsLandformAgent(
                config=agent_cfg,
                classifier=classifier_proxy,
                rag=None,  # RAG not available in API mode
                mola_features=mola_dict,
                metadata={},
                vlm=None,  # Auto-detect: Groq > Claude > Mock
            )

            # Run async agent in sync context
            loop = asyncio.new_event_loop()
            try:
                agent_result = loop.run_until_complete(agent.classify_image(product_id))
            finally:
                loop.close()

            # Convert to API model
            reasoning_steps = []
            for step_data in agent_result.reasoning_chain:
                if isinstance(step_data, dict):
                    reasoning_steps.append(AgentReasoningStep(
                        step=step_data.get("step", 0),
                        action=step_data.get("action"),
                        action_input=step_data.get("action_input"),
                        observation=step_data.get("observation"),
                        thought=step_data.get("thought") or (step_data.get("parsed", {}) or {}).get("thought"),
                        vlm_response=step_data.get("vlm_response"),
                        error=step_data.get("error"),
                        forced_final=step_data.get("forced_final", False),
                    ))

            return AgentReasoningResult(
                enabled=True,
                mode=agent_result.mode,
                landform_class=agent_result.landform_class,
                confidence=agent_result.confidence,
                reasoning_chain=reasoning_steps,
                tools_used=agent_result.tools_used,
                num_steps=agent_result.num_steps,
                error=agent_result.error,
            )

        except Exception as exc:
            logger.warning("VLM agent failed: %s", exc, exc_info=True)
            return AgentReasoningResult(
                enabled=True,
                mode="agent",
                error=str(exc),
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_tile_latlon(
        self,
        coords: list[tuple[int, int]],
        img_w: int,
        img_h: int,
        center_lat: float | None,
        center_lon: float | None,
    ) -> list[tuple[float, float]]:
        """Compute approximate lat/lon for each tile from image center."""
        if center_lat is None or center_lon is None:
            return [(0.0, 0.0) for _ in coords]

        mars_circumference_m = 2 * np.pi * 3389500
        deg_per_meter_lat = 360.0 / mars_circumference_m
        deg_per_meter_lon = deg_per_meter_lat / max(np.cos(np.radians(center_lat)), 0.01)
        pixel_scale_m = 25.0  # HiRISE browse is ~25 m/px

        img_center_row = img_h / 2.0
        img_center_col = img_w / 2.0

        result = []
        for gx, gy in coords:
            tile_center_row = gy * 224 + 112
            tile_center_col = gx * 224 + 112
            dy_px = img_center_row - tile_center_row  # positive = north
            dx_px = tile_center_col - img_center_col  # positive = east
            t_lat = center_lat + dy_px * pixel_scale_m * deg_per_meter_lat
            t_lon = center_lon + dx_px * pixel_scale_m * deg_per_meter_lon
            result.append((float(t_lat), float(t_lon)))
        return result

    def _build_tile_results(
        self,
        coords: list[tuple[int, int]],
        tile_lat_lon: list[tuple[float, float]],
        attention_weights: np.ndarray,
    ) -> list[TileResult]:
        """Build TileResult list with normalized attention weights."""
        # Normalize attention to [0, 1]
        att_min = float(attention_weights.min())
        att_max = float(attention_weights.max())
        att_range = att_max - att_min if att_max > att_min else 1.0

        tile_results: list[TileResult] = []
        for idx, (gx, gy) in enumerate(coords):
            lat, lon = tile_lat_lon[idx]
            raw_weight = float(attention_weights[idx]) if idx < len(attention_weights) else 0.0
            norm_weight = (raw_weight - att_min) / att_range
            norm_weight = max(0.0, min(1.0, norm_weight))
            tile_results.append(
                TileResult(x=gx, y=gy, attention_weight=norm_weight, lat=lat, lon=lon)
            )
        return tile_results

    def _empty_result(self, request: ClassifyRequest, elapsed: float) -> ClassifyResult:
        """Return when no tiles are extractable."""
        return ClassifyResult(
            product_id=request.product_id,
            model_used="v2",
            prediction=PredictionResult(
                top_class="BACKGROUND",
                probabilities={name: 0.2 for name in V2_CLASSES},
                confidence=0.2,
            ),
            tiles=[],
            heatmap_url=None,
            processing_time_s=round(elapsed, 2),
            num_tiles=0,
            device=str(self._device),
        )

    def status(self) -> dict[str, Any]:
        """Runtime status for the /status endpoint."""
        models_loaded = []
        if self._models is not None:
            models_loaded = ["v2"]

        memory_mb = 0.0
        if self._device.type == "cuda":
            try:
                memory_mb = torch.cuda.memory_allocated(self._device) / (1024 * 1024)
            except Exception:
                pass

        return {
            "models_loaded": models_loaded,
            "device": str(self._device),
            "memory_mb": round(memory_mb, 1),
        }


class _ClassifierProxy:
    """
    Lightweight proxy that feeds pre-computed classifier results
    into the agent's ClassifyTool without re-running inference.
    """

    def __init__(
        self,
        product_id: str,
        predicted_class: str,
        confidence: float,
        probabilities: list[float],
        attention_weights: list[float],
    ) -> None:
        self._product_id = product_id
        self._predicted_class = predicted_class
        self._confidence = confidence
        self._probabilities = probabilities
        self._attention_weights = attention_weights
        self.is_trained = True

    def predict_image(self, image_id: str) -> dict[str, Any]:
        return {
            "class": self._predicted_class,
            "confidence": self._confidence,
            "probabilities": self._probabilities,
            "attention_weights": self._attention_weights,
        }
