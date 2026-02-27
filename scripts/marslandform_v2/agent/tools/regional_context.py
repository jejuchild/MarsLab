from __future__ import annotations

import math
from typing import Any, Dict, List


class RegionalContextTool:
    def __init__(self, metadata: Any, classifier: Any) -> None:
        self.metadata = metadata
        self.classifier = classifier

    def run(self, image_id: str, radius_km: float = 100.0) -> Dict[str, Any]:
        records = self._records()
        target = self._find_record(records, image_id)
        if not target:
            return {
                "neighbors": [],
                "cluster_stats": {},
                "latitude_context": "Target image metadata not found.",
                "error": f"Metadata for image_id={image_id} not found.",
            }

        t_lat = self._get_float(target, "latitude", "lat")
        t_lon = self._get_float(target, "longitude", "lon", "lng")
        if t_lat is None or t_lon is None:
            return {
                "neighbors": [],
                "cluster_stats": {},
                "latitude_context": "Latitude/longitude missing for target image.",
                "error": "Target metadata does not include coordinates.",
            }

        neighbors = []
        class_counts: Dict[str, int] = {}

        for record in records:
            rid = str(record.get("image_id", ""))
            if not rid or rid == image_id:
                continue

            lat = self._get_float(record, "latitude", "lat")
            lon = self._get_float(record, "longitude", "lon", "lng")
            if lat is None or lon is None:
                continue

            distance = self._haversine_km(t_lat, t_lon, lat, lon)
            if distance > float(radius_km):
                continue

            pred = self._prediction_label(record)
            class_counts[pred] = class_counts.get(pred, 0) + 1
            neighbors.append({"image_id": rid, "distance_km": round(distance, 2), "prediction": pred})

        neighbors = sorted(neighbors, key=lambda item: item["distance_km"])
        cluster_stats = {
            "num_neighbors": len(neighbors),
            "class_distribution": class_counts,
        }

        latitude_context = self._latitude_context(t_lat)
        return {
            "neighbors": neighbors,
            "cluster_stats": cluster_stats,
            "latitude_context": latitude_context,
        }

    def _records(self) -> List[Dict[str, Any]]:
        if isinstance(self.metadata, list):
            return [item for item in self.metadata if isinstance(item, dict)]
        if isinstance(self.metadata, dict):
            images = self.metadata.get("images")
            if isinstance(images, list):
                return [item for item in images if isinstance(item, dict)]
            return [value for value in self.metadata.values() if isinstance(value, dict) and "image_id" in value]
        return []

    @staticmethod
    def _find_record(records: List[Dict[str, Any]], image_id: str) -> Dict[str, Any]:
        for record in records:
            if str(record.get("image_id", "")) == image_id:
                return record
        return {}

    @staticmethod
    def _get_float(record: Dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in record and record[key] is not None:
                try:
                    return float(record[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _prediction_label(record: Dict[str, Any]) -> str:
        for key in ("predicted_class", "prediction", "class", "label"):
            value = record.get(key)
            if value:
                return str(value).upper()
        return "UNCLASSIFIED"

    @staticmethod
    def _latitude_context(latitude: float) -> str:
        abs_lat = abs(latitude)
        if abs_lat < 25.0:
            return "Low-mid latitude band: glacial/periglacial classes are less common; BACKGROUND and non-glacial terrains are more likely."
        if 25.0 <= abs_lat <= 35.0:
            return "Transitional latitude band: mixed signatures possible, with increasing incidence of LVF and GLF features."
        if 35.0 < abs_lat <= 50.0:
            return "Core mid-latitude glacial belt: LDA and LVF are expected frequently, with localized CCF and GLF occurrences."
        return "High latitude margin: periglacial textures may dominate; evaluate context carefully against non-target patterned ground."

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        mars_radius_km = 3389.5
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lam = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return mars_radius_km * c
