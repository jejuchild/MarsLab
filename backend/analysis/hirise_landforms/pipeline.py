from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .heatmap import generate_class_map, save_heatmap
from .models import (
    AgentReasoningResult,
    AgentReasoningStep,
    ClassSummary,
    ClassifyRequest,
    ClassifyResult,
    TilePrediction,
)
from .preprocessing import extract_mola_features, extract_mola_features_batch, fetch_hirise_browse, tile_image
from .models import CLASS_THRESHOLDS, UNCERTAIN_CLASS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
V3_TILE_CHECKPOINT = ROOT / "Data" / "HiRISE" / "v3_output" / "models" / "best_tile_classifier.pt"
V4B_FILM_CHECKPOINT = ROOT / "Data" / "HiRISE" / "v3_output" / "models" / "marslandform_v4b_deploy.pt"
V5_FILM_CHECKPOINT = ROOT / "Data" / "HiRISE" / "v3_output" / "models" / "marslandform_v5_deploy.pt"
SSL_LORA_WEIGHTS = ROOT / "Data" / "HiRISE" / "v2_output" / "ssl_lora_weights" / "best_model.pt"

V3_CLASSES = ["LDA", "LVF", "CCF", "OTHER", "SCT"]
CONFIDENCE_THRESHOLD = 0.7

_tile_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


@dataclass(frozen=True)
class LoadedModels:
    backbone: Any
    classifier: Any
    device: torch.device
    model_version: str  # 'v3-concat' or 'v4b-film'


class HiriseLandformPipeline:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: LoadedModels | None = None
        self._loading = False

    @property
    def device(self) -> str:
        return str(self._device)

    def _ensure_models(self) -> LoadedModels:
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

        config_module = importlib.import_module("scripts.marslandform_v2.config")
        dinov2_module = importlib.import_module("scripts.marslandform_v2.models.dinov2_lora")

        DINOv2Config = getattr(config_module, "DINOv2Config")
        DinoV2LoRA = getattr(dinov2_module, "DinoV2LoRA")

        device = self._device

        # --- Load backbone ---
        dinov2_cfg = DINOv2Config()
        backbone = DinoV2LoRA(dinov2_cfg, use_lora=True)

        # Priority: load fine-tuned backbone from best checkpoint (V5 > V4b)
        v4b_backbone_loaded = False
        ckpt_path = V5_FILM_CHECKPOINT if V5_FILM_CHECKPOINT.exists() else V4B_FILM_CHECKPOINT
        if ckpt_path.exists():
            v4b_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            backbone_state = {
                k: v for k, v in v4b_ckpt.get("model_state_dict", {}).items()
                if k.startswith("backbone.")
            }
            if backbone_state:
                result = backbone.load_state_dict(backbone_state, strict=False)
                if not result.unexpected_keys and not result.missing_keys:
                    logger.info("Loaded V4b fine-tuned backbone weights (%d tensors)", len(backbone_state))
                    v4b_backbone_loaded = True
                else:
                    logger.warning(
                        "V4b backbone partial load: %d unexpected, %d missing",
                        len(result.unexpected_keys), len(result.missing_keys),
                    )
                    v4b_backbone_loaded = True  # Still better than SSL-only

        if not v4b_backbone_loaded:
            # Fallback: load SSL pretrained LoRA weights
            if SSL_LORA_WEIGHTS.exists():
                checkpoint = torch.load(SSL_LORA_WEIGHTS, map_location="cpu", weights_only=False)
                if "student_backbone" in checkpoint:
                    backbone.load_state_dict(checkpoint["student_backbone"], strict=False)
                    logger.info("Loaded SSL LoRA backbone weights")
                else:
                    logger.warning("SSL checkpoint missing 'student_backbone' key: %s", SSL_LORA_WEIGHTS)
            else:
                logger.warning("No backbone weights found — using pretrained DINOv2 only")

        backbone.eval()
        backbone = backbone.to(device)

        # --- Load classifier (auto-detect V4b FiLM vs V3 concat) ---
        classifier, model_version = self._load_classifier(device)

        logger.info("Loaded landform model: %s on %s", model_version, device)
        return LoadedModels(backbone=backbone, classifier=classifier, device=device, model_version=model_version)

    def _load_classifier(self, device: torch.device) -> tuple[Any, str]:
        """Auto-detect and load the best available classifier checkpoint."""
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        # Priority 1: V5 FiLM model (5-class, SCT expansion)
        if V5_FILM_CHECKPOINT.exists():
            return self._load_film_classifier(V5_FILM_CHECKPOINT, device)

        # Priority 2: V4b FiLM model (4-class, best pre-expansion F1)
        if V4B_FILM_CHECKPOINT.exists():
            return self._load_film_classifier(V4B_FILM_CHECKPOINT, device)

        # Priority 2: V3 concat model (check if checkpoint is actually FiLM)
        if V3_TILE_CHECKPOINT.exists():
            tile_ckpt = torch.load(V3_TILE_CHECKPOINT, map_location="cpu", weights_only=False)
            state_dict = tile_ckpt.get("model_state_dict", {})

            # Auto-detect: FiLM checkpoints have 'film.' keys
            has_film_keys = any(k.startswith("film.") for k in state_dict)
            # Also detect from cfg/version
            cfg_data = tile_ckpt.get("cfg", tile_ckpt.get("config", {}))
            version = tile_ckpt.get("version", "")
            is_film = has_film_keys or "film" in version.lower()

            if is_film:
                return self._load_film_classifier(V3_TILE_CHECKPOINT, device)
            else:
                return self._load_concat_classifier(tile_ckpt, device)

        logger.warning(
            "No tile classifier checkpoint found. Checked: %s, %s",
            V4B_FILM_CHECKPOINT, V3_TILE_CHECKPOINT,
        )
        return None, "none"

    def _load_film_classifier(self, ckpt_path: Path, device: torch.device) -> tuple[Any, str]:
        """Load V4b FiLM classifier from checkpoint."""
        film_module = importlib.import_module("scripts.marslandform_v2.models.film_classifier")
        FiLMClassifier = getattr(film_module, "FiLMClassifier")

        tile_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = tile_ckpt.get("model_state_dict", {})
        cfg_data = tile_ckpt.get("cfg", tile_ckpt.get("config", {}))

        # Extract FiLM + classifier weights (filter out backbone.* keys)
        head_state = {}
        for k, v in state_dict.items():
            if k.startswith("film.") or k.startswith("classifier."):
                head_state[k] = v

        classifier = FiLMClassifier(
            visual_dim=int(cfg_data.get("hidden_dim", 768)),
            mola_dim=int(cfg_data.get("mola_dim", 25)),
            film_hidden=64,
            head_hidden=int(cfg_data.get("head_hidden", 128)),
            num_classes=int(cfg_data.get("num_classes", 4)),
            dropout=float(cfg_data.get("dropout", 0.4)),
        )

        result = classifier.load_state_dict(head_state, strict=False)
        if result.unexpected_keys:
            logger.warning("Unexpected keys in FiLM checkpoint: %s", result.unexpected_keys)
        if result.missing_keys:
            logger.warning("Missing keys in FiLM checkpoint: %s", result.missing_keys)

        classifier.eval()
        classifier = classifier.to(device)
        epoch = tile_ckpt.get("epoch", "?")
        val_f1 = tile_ckpt.get("val_f1", tile_ckpt.get("test_f1", "?"))
        logger.info("Loaded V4b FiLM classifier from %s (epoch=%s, F1=%s)", ckpt_path.name, epoch, val_f1)
        return classifier, "v4b-film"

    def _load_concat_classifier(self, tile_ckpt: dict[str, Any], device: torch.device) -> tuple[Any, str]:
        """Load V3 concat classifier from checkpoint dict."""
        config_module = importlib.import_module("scripts.marslandform_v2.config")
        tile_module = importlib.import_module("scripts.marslandform_v2.models.tile_classifier")

        TileClassifierConfig = getattr(config_module, "TileClassifierConfig")
        TileLandformClassifier = getattr(tile_module, "TileLandformClassifier")

        cfg_data = tile_ckpt.get("config", {})
        tile_cfg = TileClassifierConfig(
            embed_dim=cfg_data.get("embed_dim", 768),
            mola_dim=cfg_data.get("mola_dim", 25),
            hidden_dim=cfg_data.get("hidden_dim", 256),
            num_classes=cfg_data.get("num_classes", 4),
            dropout=cfg_data.get("dropout", 0.3),
        )
        classifier = TileLandformClassifier(tile_cfg)
        classifier.load_state_dict(tile_ckpt["model_state_dict"])
        classifier.eval()
        classifier = classifier.to(device)
        epoch = tile_ckpt.get("epoch", "?")
        f1 = tile_ckpt.get("best_landform_macro_f1", "?")
        logger.info("Loaded V3 concat classifier (epoch=%s, F1=%s)", epoch, f1)
        return classifier, "v3-concat"

    def classify(self, request: ClassifyRequest) -> ClassifyResult:
        t0 = time.time()
        models = self._ensure_models()

        if models.classifier is None:
            return self._empty_result(request, time.time() - t0)

        image = fetch_hirise_browse(request.product_id)
        img_w, img_h = image.size

        tiled = tile_image(image, tile_size=224, min_content=0.3)
        coords = [(x, y) for x, y, _ in tiled]
        tile_images = [tile for _, _, tile in tiled]
        num_tiles = len(tile_images)

        if num_tiles == 0:
            return self._empty_result(request, time.time() - t0)

        tile_lat_lon = self._compute_tile_latlon(coords, img_w, img_h, request.lat, request.lon)
        embeddings = self._extract_embeddings(models, tile_images)
        tile_mola = self._extract_tile_mola_features(tile_lat_lon, request.lat, request.lon)
        probabilities, predictions = self._run_tile_classifier(models, embeddings, tile_mola)

        # Optional CRF spatial smoothing on the tile grid
        if request.use_crf and num_tiles >= 2:
            probabilities, predictions = self._apply_crf_smoothing(
                coords, probabilities, smoothing_weight=1.5, n_iterations=10,
            )

        tile_predictions = self._build_tile_predictions(
            coords, tile_lat_lon, probabilities, predictions, request.confidence_threshold,
        )
        class_summary = self._build_class_summary(tile_predictions)
        dominant_class, dominant_confidence = self._dominant_from_summary(class_summary)

        heatmap_url: str | None = None
        if request.include_heatmap:
            heatmap = generate_class_map(image=image, tile_predictions=tile_predictions, tile_size=224)
            heatmap_url = save_heatmap(heatmap, request.product_id)

        agent_reasoning: AgentReasoningResult | None = None
        if dominant_confidence < CONFIDENCE_THRESHOLD:
            class_distribution = self._class_distribution_from_summary(class_summary)
            tile_confidences = [float(tp.confidence) for tp in tile_predictions]
            center_mola = tile_mola.mean(axis=0) if tile_mola.size > 0 else np.zeros((25,), dtype=np.float32)
            agent_reasoning = self._run_vlm_agent(
                product_id=request.product_id,
                dominant_class=dominant_class,
                dominant_confidence=dominant_confidence,
                class_distribution=class_distribution,
                tile_confidences=tile_confidences,
                mola_features=center_mola,
                lat=request.lat,
                lon=request.lon,
            )
            if (
                agent_reasoning is not None
                and agent_reasoning.enabled
                and agent_reasoning.confidence is not None
                and agent_reasoning.confidence > dominant_confidence
                and agent_reasoning.landform_class is not None
                and agent_reasoning.landform_class in V3_CLASSES
            ):
                dominant_class = agent_reasoning.landform_class
                dominant_confidence = float(agent_reasoning.confidence)

        return ClassifyResult(
            product_id=request.product_id,
            model_used=models.model_version,
            tile_predictions=tile_predictions,
            class_summary=class_summary,
            dominant_class=dominant_class,
            dominant_confidence=float(dominant_confidence),
            heatmap_url=heatmap_url,
            processing_time_s=round(time.time() - t0, 2),
            agent_reasoning=agent_reasoning,
            num_tiles=num_tiles,
            device=str(self._device),
        )

    @property
    def model_version(self) -> str:
        """Return the currently loaded model version string."""
        if self._models is not None:
            return self._models.model_version
        return "not-loaded"

    @property
    def is_loaded(self) -> bool:
        """Return True if models are loaded."""
        return self._models is not None and self._models.classifier is not None
    @torch.no_grad()
    def _extract_embeddings(self, models: LoadedModels, tile_images: list[Image.Image]) -> np.ndarray:
        device = models.device
        backbone = models.backbone
        batch_size = 32
        tensors: list[torch.Tensor] = []
        for tile in tile_images:
            transformed = _tile_transform(tile.convert("RGB"))
            if isinstance(transformed, torch.Tensor):
                tensors.append(transformed)
            else:
                tensors.append(transforms.ToTensor()(tile.convert("RGB")))

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                emb = backbone(batch)
            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def _extract_tile_mola_features(
        self,
        tile_lat_lon: list[tuple[float | None, float | None]],
        center_lat: float | None,
        center_lon: float | None,
    ) -> np.ndarray:
        n_tiles = len(tile_lat_lon)
        if n_tiles == 0:
            return np.zeros((0, 25), dtype=np.float32)

        # Collect tiles with valid coordinates for batch extraction
        valid_coords: list[tuple[float, float]] = []
        valid_indices: list[int] = []
        for i, (lat, lon) in enumerate(tile_lat_lon):
            if lat is not None and lon is not None:
                valid_coords.append((lat, lon))
                valid_indices.append(i)

        # Batch-extract MOLA: single DEM I/O for all tiles
        if valid_coords:
            batch_features = extract_mola_features_batch(valid_coords)  # (N_valid, 23)
        else:
            batch_features = np.zeros((0, 23), dtype=np.float32)

        # Fallback for tiles without coords
        if center_lat is not None and center_lon is not None and len(valid_coords) < n_tiles:
            center_base = extract_mola_features(center_lat, center_lon)
        else:
            center_base = np.zeros((23,), dtype=np.float32)

        # Assemble full array
        base_arr = np.zeros((n_tiles, 23), dtype=np.float32)
        for j, orig_i in enumerate(valid_indices):
            base_arr[orig_i] = batch_features[j]
        # Fill missing tiles with center MOLA
        for i in range(n_tiles):
            if i not in valid_indices:
                base_arr[i] = center_base[:23]

        # Relative features: per-tile deviation from image mean
        mean_elev = float(np.mean(base_arr[:, 21]))
        mean_slope = float(np.mean(base_arr[:, 0]))
        rel_elev = (base_arr[:, 21] - mean_elev).reshape(-1, 1)
        rel_slope = (base_arr[:, 0] - mean_slope).reshape(-1, 1)
        return np.concatenate([base_arr, rel_elev, rel_slope], axis=1).astype(np.float32)

    @torch.no_grad()
    def _run_tile_classifier(
        self,
        models: LoadedModels,
        embeddings: np.ndarray,
        mola_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if models.classifier is None or embeddings.size == 0:
            probs = np.zeros((embeddings.shape[0], len(V3_CLASSES)), dtype=np.float32)
            if probs.shape[0] > 0:
                probs[:, -1] = 1.0
            preds = np.full((embeddings.shape[0],), len(V3_CLASSES) - 1, dtype=np.int64)
            return probs, preds

        device = models.device
        emb_t = torch.from_numpy(embeddings).float().to(device)
        mola_t = torch.from_numpy(mola_features).float().to(device)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = models.classifier(emb_t, mola_t)

        probs_t = torch.softmax(logits, dim=1)
        probs = probs_t.cpu().numpy()
        preds = np.argmax(probs, axis=1).astype(np.int64)
        return probs, preds

    def _apply_crf_smoothing(
        self,
        coords: list[tuple[int, int]],
        probabilities: np.ndarray,
        smoothing_weight: float = 1.5,
        n_iterations: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean-field CRF on the tile grid for spatial consistency.

        Uses a compatibility matrix that allows LDA↔LVF glacial transitions
        while penalising unlikely neighbour combinations.
        """
        # Compatibility matrix: how likely class_i neighbours class_j
        #                   LDA   LVF   CCF   OTHER
        compat = np.array([
            [1.0, 0.8, 0.3, 0.4],  # LDA
            [0.8, 1.0, 0.3, 0.4],  # LVF
            [0.3, 0.3, 1.0, 0.5],  # CCF
            [0.4, 0.4, 0.5, 1.0],  # OTHER
        ], dtype=np.float32)

        # Build position → local index mapping
        pos_to_idx: dict[tuple[int, int], int] = {}
        for i, (gx, gy) in enumerate(coords):
            pos_to_idx[(gy, gx)] = i  # (row, col)

        smoothed = probabilities.copy()
        n_tiles = len(coords)

        for _ in range(n_iterations):
            new_logits = np.log(smoothed + 1e-10)  # unary
            for i, (gx, gy) in enumerate(coords):
                neighbour_msg = np.zeros(len(V3_CLASSES), dtype=np.float32)
                n_nbrs = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    key = (gy + dr, gx + dc)
                    if key in pos_to_idx:
                        nidx = pos_to_idx[key]
                        neighbour_msg += compat @ smoothed[nidx]
                        n_nbrs += 1
                if n_nbrs > 0:
                    new_logits[i] += smoothing_weight * neighbour_msg / n_nbrs
            # Stable softmax
            exp_x = np.exp(new_logits - np.max(new_logits, axis=1, keepdims=True))
            smoothed = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        preds = np.argmax(smoothed, axis=1).astype(np.int64)
        logger.debug("CRF smoothing: %d tiles, %d iterations", n_tiles, n_iterations)
        return smoothed, preds

    def _run_vlm_agent(
        self,
        product_id: str,
        dominant_class: str,
        dominant_confidence: float,
        class_distribution: np.ndarray,
        tile_confidences: list[float],
        mola_features: np.ndarray,
        lat: float | None,
        lon: float | None,
    ) -> AgentReasoningResult:
        try:
            import os
            import sys

            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            agent_module = importlib.import_module("scripts.marslandform_v2.agent.react_agent")
            config_module = importlib.import_module("scripts.marslandform_v2.config")
            MarsLandformAgent = getattr(agent_module, "MarsLandformAgent")
            AgentConfig = getattr(config_module, "AgentConfig")

            classifier_proxy = _ClassifierProxy(
                product_id=product_id,
                predicted_class=dominant_class,
                confidence=dominant_confidence,
                probabilities=class_distribution.tolist(),
                attention_weights=tile_confidences,
            )

            mola_dict: dict[str, dict[str, Any]] = {}
            if lat is not None and lon is not None:
                feature_names = []
                for scale in [1.0, 5.0, 20.0]:
                    for feat in ["slope_mean", "slope_std", "curvature_mean", "TPI", "TRI", "roughness", "lobateness"]:
                        feature_names.append(f"{feat}_{scale}km")
                feature_names.extend(["elevation_mean", "abs_latitude", "elevation_relative_to_image_mean", "slope_relative_to_image_mean"])

                mola_entry: dict[str, Any] = {}
                for i, name in enumerate(feature_names):
                    if i < len(mola_features):
                        mola_entry[name] = float(mola_features[i])
                mola_entry["lat"] = lat
                mola_entry["lon"] = lon
                mola_dict[product_id] = mola_entry

            if os.getenv("GROQ_API_KEY"):
                vlm_model = "llama-3.3-70b-versatile"
            elif os.getenv("ANTHROPIC_API_KEY"):
                vlm_model = "claude-sonnet-4-20250514"
            else:
                vlm_model = ""

            agent_cfg = AgentConfig(
                max_steps=3,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                mode="agent",
                vlm_model=vlm_model,
            )

            agent = MarsLandformAgent(
                config=agent_cfg,
                classifier=classifier_proxy,
                rag=None,
                mola_features=mola_dict,
                metadata={},
                vlm=None,
            )

            loop = asyncio.new_event_loop()
            try:
                agent_result = loop.run_until_complete(agent.classify_image(product_id))
            finally:
                loop.close()

            reasoning_steps: list[AgentReasoningStep] = []
            for step_data in agent_result.reasoning_chain:
                if isinstance(step_data, dict):
                    reasoning_steps.append(
                        AgentReasoningStep(
                            step=step_data.get("step", 0),
                            action=step_data.get("action"),
                            action_input=step_data.get("action_input"),
                            observation=step_data.get("observation"),
                            thought=step_data.get("thought") or (step_data.get("parsed", {}) or {}).get("thought"),
                            vlm_response=step_data.get("vlm_response"),
                            error=step_data.get("error"),
                            forced_final=step_data.get("forced_final", False),
                        )
                    )

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
            return AgentReasoningResult(enabled=True, mode="agent", error=str(exc))

    def _compute_tile_latlon(
        self,
        coords: list[tuple[int, int]],
        img_w: int,
        img_h: int,
        center_lat: float | None,
        center_lon: float | None,
    ) -> list[tuple[float | None, float | None]]:
        if center_lat is None or center_lon is None:
            return [(None, None) for _ in coords]

        mars_circumference_m = 2 * np.pi * 3389500
        deg_per_meter_lat = 360.0 / mars_circumference_m
        deg_per_meter_lon = deg_per_meter_lat / max(np.cos(np.radians(center_lat)), 0.01)
        pixel_scale_m = 25.0

        img_center_row = img_h / 2.0
        img_center_col = img_w / 2.0

        result: list[tuple[float | None, float | None]] = []
        for gx, gy in coords:
            tile_center_row = gy * 224 + 112
            tile_center_col = gx * 224 + 112
            dy_px = img_center_row - tile_center_row
            dx_px = tile_center_col - img_center_col
            t_lat = center_lat + dy_px * pixel_scale_m * deg_per_meter_lat
            t_lon = center_lon + dx_px * pixel_scale_m * deg_per_meter_lon
            lat_val: float | None = float(t_lat)
            lon_val: float | None = float(t_lon)
            result.append((lat_val, lon_val))
        return result

    def _build_tile_predictions(
        self,
        coords: list[tuple[int, int]],
        tile_lat_lon: list[tuple[float | None, float | None]],
        probabilities: np.ndarray,
        predictions: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[TilePrediction]:
        out: list[TilePrediction] = []
        for idx, (gx, gy) in enumerate(coords):
            pred_idx = int(predictions[idx]) if idx < len(predictions) else (len(V3_CLASSES) - 1)
            pred_idx = max(0, min(pred_idx, len(V3_CLASSES) - 1))
            probs = probabilities[idx] if idx < len(probabilities) else np.zeros((len(V3_CLASSES),), dtype=np.float32)
            confidence = float(probs[pred_idx]) if pred_idx < len(probs) else 0.0
            confidence = max(0.0, min(1.0, confidence))
            raw_class = V3_CLASSES[pred_idx]
            thresh = confidence_threshold if confidence_threshold is not None else CLASS_THRESHOLDS.get(raw_class, 0.6)
            display_class = raw_class if confidence >= thresh else UNCERTAIN_CLASS
            lat, lon = tile_lat_lon[idx]
            out.append(
                TilePrediction(
                    x=gx,
                    y=gy,
                    predicted_class=display_class,
                    raw_class=raw_class,
                    confidence=confidence,
                    probabilities={name: float(probs[i]) for i, name in enumerate(V3_CLASSES) if i < len(probs)},
                    lat=float(lat if lat is not None else 0.0),
                    lon=float(lon if lon is not None else 0.0),
                )
            )
        return out

    def _build_class_summary(self, tile_predictions: list[TilePrediction]) -> list[ClassSummary]:
        total = max(len(tile_predictions), 1)
        summary: list[ClassSummary] = []
        all_classes = V3_CLASSES + [UNCERTAIN_CLASS]
        for class_name in all_classes:
            class_tiles = [tp for tp in tile_predictions if tp.predicted_class == class_name]
            count = len(class_tiles)
            mean_conf = float(np.mean([tp.confidence for tp in class_tiles])) if class_tiles else 0.0
            summary.append(
                ClassSummary(
                    class_name=class_name,
                    tile_count=count,
                    percentage=(100.0 * count / total) if tile_predictions else 0.0,
                    mean_confidence=mean_conf,
                )
            )
        return summary

    def _dominant_from_summary(self, class_summary: list[ClassSummary]) -> tuple[str, float]:
        # Only consider real classes (not Uncertain) for dominant
        real = [s for s in class_summary if s.class_name not in ('OTHER', UNCERTAIN_CLASS)]
        real = sorted(real, key=lambda s: s.tile_count, reverse=True)
        if real and real[0].tile_count > 0:
            return real[0].class_name, float(real[0].mean_confidence)

        other = next((s for s in class_summary if s.class_name == 'OTHER'), None)
        if other and other.tile_count > 0:
            return 'OTHER', float(other.mean_confidence)

        uncertain = next((s for s in class_summary if s.class_name == UNCERTAIN_CLASS), None)
        if uncertain and uncertain.tile_count > 0:
            return UNCERTAIN_CLASS, float(uncertain.mean_confidence)
        return 'OTHER', 0.0

    def _class_distribution_from_summary(self, class_summary: list[ClassSummary]) -> np.ndarray:
        percentages = {entry.class_name: float(entry.percentage) / 100.0 for entry in class_summary}
        return np.array([percentages.get(name, 0.0) for name in V3_CLASSES], dtype=np.float32)

    def _empty_result(self, request: ClassifyRequest, elapsed: float) -> ClassifyResult:
        class_summary = [
            ClassSummary(class_name=name, tile_count=0, percentage=0.0, mean_confidence=0.0)
            for name in V3_CLASSES
        ]
        return ClassifyResult(
            product_id=request.product_id,
            model_used="v3",
            tile_predictions=[],
            class_summary=class_summary,
            dominant_class="OTHER",
            dominant_confidence=0.0,
            heatmap_url=None,
            processing_time_s=round(elapsed, 2),
            num_tiles=0,
            device=str(self._device),
        )

    def status(self) -> dict[str, Any]:
        models_loaded = []
        if self._models is not None:
            models_loaded = ["v3"]

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
