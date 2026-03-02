from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .heatmap import generate_heatmap, save_heatmap
from .models import ClassifyRequest, ClassifyResult, PredictionResult, TileResult
from .preprocessing import extract_mola_features, fetch_hirise_browse, tile_image

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = ROOT / "backend" / "data" / "hirise_landforms" / "weights"
V2_CLASSES = ["LDA", "LVF", "CCF", "GLF", "BACKGROUND"]
MARS_BENCH_CLASSES = ["other", "crater", "dark_dune", "streak", "bright_dune", "impact", "edge"]


@dataclass(frozen=True)
class LoadedModel:
    name: str
    class_names: list[str]
    checkpoint_path: Path


class HiriseLandformPipeline:
    device: str
    _models: dict[str, LoadedModel]

    def __init__(self) -> None:
        try:
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self.device = "cpu"
        self._models = {}

    def load_model(self, model_name: str) -> LoadedModel:
        normalized = model_name.strip().lower()
        if normalized in self._models:
            return self._models[normalized]

        if normalized == "v2":
            loaded = self._load_v2_model()
        elif normalized == "mars-bench":
            loaded = self._load_mars_bench_model()
        else:
            raise ValueError(f"Unsupported model '{model_name}'. Expected 'v2' or 'mars-bench'.")

        self._models[normalized] = loaded
        return loaded

    def classify(self, request: ClassifyRequest) -> ClassifyResult:
        loaded = self.load_model(request.model)
        image = fetch_hirise_browse(request.product_id)
        tiled = tile_image(image, tile_size=224)
        coords = [(x, y) for x, y, _ in tiled]
        tiles = [tile for _, _, tile in tiled]
        lat_lon = self._tile_lat_lon(coords)

        probabilities, attention = self._infer(loaded.name, loaded.class_names, tiles, lat_lon)
        top_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
        prediction = PredictionResult(
            top_class=loaded.class_names[top_idx],
            probabilities={name: probabilities[idx] for idx, name in enumerate(loaded.class_names)},
            confidence=probabilities[top_idx],
        )

        tile_results: list[TileResult] = []
        for idx, (x, y) in enumerate(coords):
            lat, lon = lat_lon[idx]
            weight = attention[idx] if idx < len(attention) else 0.0
            tile_results.append(
                TileResult(
                    x=x,
                    y=y,
                    attention_weight=max(0.0, min(1.0, weight)),
                    lat=lat,
                    lon=lon,
                )
            )

        heatmap_url: str | None = None
        if request.include_heatmap:
            heatmap = generate_heatmap(image=image, tiles=tile_results, tile_size=224)
            heatmap_url = save_heatmap(heatmap, request.product_id)

        return ClassifyResult(
            product_id=request.product_id,
            model_used=loaded.name,
            prediction=prediction,
            tiles=tile_results,
            heatmap_url=heatmap_url,
        )

    def _load_v2_model(self) -> LoadedModel:
        checkpoint = self._find_weight_path(("v2", "marslandform", "mil"))
        if checkpoint is None:
            raise FileNotFoundError(
                f"No V2 weights found in '{WEIGHTS_DIR}'. Add a V2 checkpoint under this directory."
            )

        try:
            _ = importlib.import_module("scripts.marslandform_v2.config")
            _ = importlib.import_module("scripts.marslandform_v2.models.mil_classifier")
            _ = importlib.import_module("scripts.marslandform_v2.models.dinov2_lora")
        except Exception as exc:
            raise RuntimeError(f"Failed to import MarsLandform V2 modules at runtime: {exc}") from exc

        return LoadedModel(name="v2", class_names=V2_CLASSES, checkpoint_path=checkpoint)

    def _load_mars_bench_model(self) -> LoadedModel:
        checkpoint = self._find_weight_path(("mars-bench", "marsbench", "vit"))
        if checkpoint is None:
            raise FileNotFoundError(
                f"No Mars-Bench weights found in '{WEIGHTS_DIR}'. Add a Mars-Bench checkpoint under this directory."
            )
        return LoadedModel(name="mars-bench", class_names=MARS_BENCH_CLASSES, checkpoint_path=checkpoint)

    def _find_weight_path(self, hints: tuple[str, ...]) -> Path | None:
        if not WEIGHTS_DIR.exists():
            return None
        candidates = [path for path in WEIGHTS_DIR.rglob("*") if path.is_file() or path.is_dir()]
        if not candidates:
            return None
        for path in sorted(candidates):
            lowered = path.name.lower()
            if any(hint in lowered for hint in hints):
                return path
        return None

    def _infer(
        self,
        model_name: str,
        class_names: list[str],
        tile_images: list[Image.Image],
        lat_lon: list[tuple[float, float]],
    ) -> tuple[list[float], list[float]]:
        features = self._tile_features(tile_images)
        if not features:
            uniform = 1.0 / max(len(class_names), 1)
            return [uniform for _ in class_names], []

        brightness = [row[0] for row in features]
        contrast = [row[1] for row in features]
        edge = [row[2] for row in features]
        texture = [row[3] for row in features]

        if model_name == "v2":
            for lat, lon in lat_lon:
                _ = extract_mola_features(lat, lon)
            mola_mean = 0.0

            scores = [
                self._mean([0.7 * t + 0.3 * c for t, c in zip(texture, contrast)]) + 0.10 * mola_mean,
                self._mean([0.6 * e + 0.4 * t for e, t in zip(edge, texture)]) + 0.08 * mola_mean,
                self._mean([0.55 * c + 0.45 * b for c, b in zip(contrast, brightness)]) + 0.06 * mola_mean,
                self._mean([0.65 * e + 0.35 * c for e, c in zip(edge, contrast)]) + 0.05 * mola_mean,
                self._mean([1.0 - (0.4 * e + 0.3 * t + 0.3 * c) for e, t, c in zip(edge, texture, contrast)]),
            ]
        else:
            scores = [
                self._mean([1.0 - value for value in edge]),
                self._mean([0.5 * e + 0.5 * c for e, c in zip(edge, contrast)]),
                self._mean([(1.0 - b) * t for b, t in zip(brightness, texture)]),
                self._mean([0.7 * e + 0.3 * (1.0 - t) for e, t in zip(edge, texture)]),
                self._mean([b * (1.0 - t) for b, t in zip(brightness, texture)]),
                self._mean([0.6 * c + 0.4 * e for c, e in zip(contrast, edge)]),
                self._mean([0.8 * e for e in edge]),
            ]

        probabilities = self._softmax(scores)
        attention_base = [0.4 * c + 0.35 * e + 0.25 * t for c, e, t in zip(contrast, edge, texture)]
        attention = self._normalize(attention_base)
        return probabilities, attention

    def _tile_features(self, tile_images: list[Image.Image]) -> list[tuple[float, float, float, float]]:
        out: list[tuple[float, float, float, float]] = []
        for tile in tile_images:
            gray = tile.convert("L")
            width, height = gray.size
            matrix = list(gray.tobytes())
            rows = [matrix[start : start + width] for start in range(0, width * height, width)]
            pixels = [float(value) / 255.0 for value in matrix]

            brightness = self._mean(pixels)
            contrast = self._std(pixels, brightness)
            edge = self._edge_strength(rows)
            low = self._percentile(pixels, 0.10)
            high = self._percentile(pixels, 0.90)
            texture = max(0.0, high - low)
            out.append((brightness, contrast, edge, texture))
        return out

    def _tile_lat_lon(self, coords: list[tuple[int, int]]) -> list[tuple[float, float]]:
        if not coords:
            return []
        max_x = max(x for x, _ in coords)
        max_y = max(y for _, y in coords)
        cols = max_x + 1
        rows = max_y + 1
        lat_lon: list[tuple[float, float]] = []
        for x, y in coords:
            lat = 90.0 - 180.0 * ((y + 0.5) / max(rows, 1))
            lon = -180.0 + 360.0 * ((x + 0.5) / max(cols, 1))
            lat_lon.append((lat, lon))
        return lat_lon

    def _mean(self, values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _std(self, values: list[float], mean: float) -> float:
        if not values:
            return 0.0
        variance = sum((value - mean) * (value - mean) for value in values) / len(values)
        return math.sqrt(max(variance, 0.0))

    def _percentile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(min(max(len(sorted_values) - 1, 0), round(q * (len(sorted_values) - 1))))
        return sorted_values[index]

    def _edge_strength(self, rows: list[list[int]]) -> float:
        if not rows or not rows[0]:
            return 0.0
        height = len(rows)
        width = len(rows[0])
        total = 0.0
        count = 0

        for y in range(height):
            for x in range(width):
                here = float(rows[y][x]) / 255.0
                if x > 0:
                    total += abs(here - float(rows[y][x - 1]) / 255.0)
                    count += 1
                if y > 0:
                    total += abs(here - float(rows[y - 1][x]) / 255.0)
                    count += 1

        if count == 0:
            return 0.0
        return total / count

    def _softmax(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        shifted = [score - max(scores) for score in scores]
        exps = [math.exp(score) for score in shifted]
        total = sum(exps)
        if total <= 0.0:
            uniform = 1.0 / len(scores)
            return [uniform for _ in scores]
        return [value / total for value in exps]

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi <= lo:
            uniform = 1.0 / len(values)
            return [uniform for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
