from __future__ import annotations

from typing import Any, Dict


class AnalyzeMOLATool:
    def __init__(self, mola_features: Dict[str, Dict[str, Any]]) -> None:
        self.mola_features = mola_features or {}

    def run(self, image_id: str) -> Dict[str, Any]:
        features = self._get_features(image_id)
        if not features:
            return {
                "elevation": None,
                "slope_mean": None,
                "slope_std": None,
                "TPI": None,
                "TRI": None,
                "roughness": None,
                "lobateness": None,
                "curvature": None,
                "interpretation": "No precomputed MOLA features found for this image.",
                "error": f"Missing MOLA features for image_id={image_id}",
            }

        elevation = self._pick(features, "elevation", "elevation_mean")
        slope_mean = self._pick(features, "slope_mean")
        slope_std = self._pick(features, "slope_std")
        tpi = self._pick(features, "TPI", "tpi")
        tri = self._pick(features, "TRI", "tri")
        roughness = self._pick(features, "roughness")
        lobateness = self._pick(features, "lobateness")
        curvature = self._pick(features, "curvature", "curvature_mean")
        latitude = self._pick(features, "latitude", "lat")
        if latitude is None:
            abs_lat = self._pick(features, "abs_latitude")
            if abs_lat is not None:
                latitude = abs_lat

        interpretation = self._interpret(
            slope_mean=slope_mean,
            tpi=tpi,
            tri=tri,
            roughness=roughness,
            latitude=latitude,
            elevation=elevation,
            features=features,
        )

        return {
            "elevation": elevation,
            "slope_mean": slope_mean,
            "slope_std": slope_std,
            "TPI": tpi,
            "TRI": tri,
            "roughness": roughness,
            "lobateness": lobateness,
            "curvature": curvature,
            "interpretation": interpretation,
        }

    def _get_features(self, image_id: str) -> Dict[str, Any]:
        if image_id in self.mola_features:
            entry = self.mola_features[image_id]
            return entry if isinstance(entry, dict) else {}
        return {}

    @staticmethod
    def _pick(features: Dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in features and features[key] is not None:
                try:
                    return float(features[key])
                except (TypeError, ValueError):
                    continue
        return None

    def _interpret(
        self,
        slope_mean: float | None,
        tpi: float | None,
        tri: float | None,
        roughness: float | None,
        latitude: float | None,
        elevation: float | None,
        features: Dict[str, Any],
    ) -> str:
        notes = []

        if slope_mean is not None:
            if slope_mean > 20.0:
                notes.append(f"Steep mean slope ({slope_mean:.1f} deg) indicates headwall terrain, often consistent with GLF.")
            elif slope_mean >= 6.0:
                notes.append(f"Moderate slope ({slope_mean:.1f} deg) supports valley/apron flow morphology.")
            else:
                notes.append(f"Low slope ({slope_mean:.1f} deg) suggests subdued relief.")

        if tpi is not None and latitude is not None and 30.0 <= abs(latitude) <= 50.0 and tpi > 20.0:
            notes.append(f"High TPI ({tpi:.1f}) in mid-latitudes ({latitude:.1f} deg) is compatible with LDA scarp-margin settings.")
        elif tpi is not None:
            notes.append(f"TPI={tpi:.1f} provides local relief context for class discrimination.")

        crater_context = bool(features.get("crater_context") or features.get("is_crater"))
        if crater_context and tri is not None and tri < 8.0:
            notes.append("Low-relief crater setting supports possible CCF.")

        valley_confined = bool(features.get("valley_confined") or features.get("in_valley"))
        if valley_confined and slope_mean is not None and 5.0 <= slope_mean <= 20.0:
            notes.append("Valley-confined morphology with moderate slope favors LVF.")

        if roughness is not None:
            notes.append(f"Surface roughness={roughness:.2f} refines confidence between smooth ice-rich and blocky units.")
        if elevation is not None:
            notes.append(f"Elevation {elevation:.0f} m provides regional climate-band context.")

        if not notes:
            return "MOLA features present but insufficient for a strong geomorphic interpretation."
        return " ".join(notes)
