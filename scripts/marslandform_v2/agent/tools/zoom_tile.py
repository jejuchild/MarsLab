from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image


class ZoomTileTool:
    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata

    def run(self, image_id: str, tile_indices: List[int] | str = "top_attention") -> Dict[str, Any]:
        record = self._get_record(image_id)
        if not record:
            return {
                "tile_descriptions": [],
                "tile_paths": [],
                "attention_weights": [],
                "error": f"Metadata for image_id={image_id} not found.",
            }

        attention = self._as_float_list(record.get("attention_weights") or record.get("tile_attention") or [])
        paths = record.get("tile_paths") or []
        embeddings = record.get("tile_embeddings") or []

        selected_indices = self._resolve_indices(tile_indices=tile_indices, attention=attention, max_len=max(len(paths), len(embeddings), len(attention)))

        class_centroids = self._class_centroids()
        tile_descriptions: List[str] = []
        tile_paths: List[str] = []
        selected_weights: List[float] = []

        for idx in selected_indices:
            tile_path = str(paths[idx]) if idx < len(paths) else ""
            emb = embeddings[idx] if idx < len(embeddings) else None
            weight = float(attention[idx]) if idx < len(attention) else 0.0

            stats = self._image_stats(tile_path)
            sim_text = self._centroid_similarity_text(emb, class_centroids)
            description = (
                f"tile={idx}, mean_brightness={stats['mean_brightness']:.3f}, "
                f"contrast={stats['contrast']:.3f}, edge_density={stats['edge_density']:.3f}, {sim_text}"
            )

            tile_descriptions.append(description)
            tile_paths.append(tile_path)
            selected_weights.append(weight)

        return {
            "tile_descriptions": tile_descriptions,
            "tile_paths": tile_paths,
            "attention_weights": selected_weights,
        }

    def _get_record(self, image_id: str) -> Dict[str, Any]:
        if isinstance(self.metadata, dict):
            if image_id in self.metadata and isinstance(self.metadata[image_id], dict):
                return self.metadata[image_id]
            images = self.metadata.get("images")
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, dict) and str(item.get("image_id")) == image_id:
                        return item
        elif isinstance(self.metadata, list):
            for item in self.metadata:
                if isinstance(item, dict) and str(item.get("image_id")) == image_id:
                    return item
        return {}

    def _class_centroids(self) -> Dict[str, np.ndarray]:
        if isinstance(self.metadata, dict):
            centroids = self.metadata.get("class_centroids")
            if isinstance(centroids, dict):
                result: Dict[str, np.ndarray] = {}
                for key, value in centroids.items():
                    arr = np.asarray(value, dtype=float)
                    if arr.ndim == 1 and arr.size > 0:
                        result[str(key).upper()] = arr
                return result
        return {}

    @staticmethod
    def _as_float_list(values: Any) -> List[float]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, list):
            return []
        output: List[float] = []
        for value in values:
            try:
                output.append(float(value))
            except (TypeError, ValueError):
                output.append(0.0)
        return output

    @staticmethod
    def _resolve_indices(tile_indices: List[int] | str, attention: List[float], max_len: int) -> List[int]:
        if max_len <= 0:
            return []

        if isinstance(tile_indices, str) and tile_indices == "top_attention":
            if attention:
                ranked = sorted(enumerate(attention), key=lambda pair: pair[1], reverse=True)
                return [idx for idx, _ in ranked[:5]]
            return list(range(min(5, max_len)))

        if isinstance(tile_indices, list):
            valid = []
            for idx in tile_indices:
                if isinstance(idx, int) and 0 <= idx < max_len:
                    valid.append(idx)
            return valid

        return []

    @staticmethod
    def _image_stats(tile_path: str) -> Dict[str, float]:
        default = {"mean_brightness": 0.0, "contrast": 0.0, "edge_density": 0.0}
        if not tile_path:
            return default

        path = Path(tile_path)
        if not path.exists() or not path.is_file():
            return default

        try:
            image = Image.open(path).convert("L")
            arr = np.asarray(image, dtype=float) / 255.0
            mean_brightness = float(arr.mean())
            contrast = float(arr.std())

            gx = np.gradient(arr, axis=1)
            gy = np.gradient(arr, axis=0)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            edge_density = float((grad_mag > 0.1).mean())
            return {
                "mean_brightness": mean_brightness,
                "contrast": contrast,
                "edge_density": edge_density,
            }
        except Exception:
            return default

    @staticmethod
    def _centroid_similarity_text(embedding: Any, class_centroids: Dict[str, np.ndarray]) -> str:
        if embedding is None or not class_centroids:
            return "centroid_similarity=unavailable"

        emb = np.asarray(embedding, dtype=float)
        if emb.ndim != 1 or emb.size == 0:
            return "centroid_similarity=unavailable"

        emb_norm = float(np.linalg.norm(emb))
        if math.isclose(emb_norm, 0.0):
            return "centroid_similarity=unavailable"

        best_class = "UNKNOWN"
        best_score = -1.0
        for cls_name, centroid in class_centroids.items():
            if centroid.shape != emb.shape:
                continue
            denom = emb_norm * float(np.linalg.norm(centroid))
            if math.isclose(denom, 0.0):
                continue
            score = float(np.dot(emb, centroid) / denom)
            if score > best_score:
                best_score = score
                best_class = cls_name

        if best_score < 0.0:
            return "centroid_similarity=unavailable"
        return f"nearest_centroid={best_class} (cos={best_score:.3f})"
