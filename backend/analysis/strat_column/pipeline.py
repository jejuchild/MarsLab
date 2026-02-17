"""
StratigraphicColumnBuilder — multi-instrument vertical stratigraphic column.

Algorithm:
  1. Run CraterStratigraphyAnalyzer to get terrace depths + εr estimates
  2. Build surface layers from terrace depth boundaries
  3. If include_crism: find nearby CRISM obs, load cached CNN, assign minerals per layer
  4. If include_sharad: add subsurface layers from εr estimates
  5. Assign colors per layer by geochemical group or material class
  6. Return assembled vertical column
"""

import logging
from typing import Dict, List, Optional

from analysis.shared.base import AnalysisModule
from analysis.shared.coordinates import haversine_km
from analysis.shared.overlay import GEOCHEM_COLORS, NO_DATA_COLOR
from .models import (
    ColumnLayer,
    ColumnSummary,
    ColumnParameters,
    StratColumnResult,
)

logger = logging.getLogger(__name__)

# ── Material class from εr ────────────────────────────────────────
EPSILON_MATERIALS = [
    (2.0, "Dry regolith / vacuum"),
    (3.5, "Ice-rich regolith"),
    (5.0, "Dusty ice / porous rock"),
    (8.0, "Basalt / dense rock"),
    (float("inf"), "Water-bearing / briny"),
]


def _material_from_epsilon(epsilon_r: float) -> str:
    """Classify material from dielectric constant."""
    for threshold, label in EPSILON_MATERIALS:
        if epsilon_r < threshold:
            return label
    return EPSILON_MATERIALS[-1][1]


class StratigraphicColumnBuilder(AnalysisModule):
    """Build a composite vertical stratigraphic column for a crater."""

    def __init__(self):
        self._result: Optional[StratColumnResult] = None

    # ────────────────────────────────────────────────────────────────
    # AnalysisModule interface
    # ────────────────────────────────────────────────────────────────

    def run(
        self,
        crater_lat: float,
        crater_lon: float,
        diameter_km: float,
        buffer_km: float = 30.0,
        include_crism: bool = True,
        include_sharad: bool = True,
    ) -> StratColumnResult:
        """Execute the stratigraphic column builder."""
        try:
            self._result = self._run_impl(
                crater_lat, crater_lon, diameter_km,
                buffer_km, include_crism, include_sharad,
            )
        except Exception as exc:
            logger.exception("Strat column pipeline failed")
            self._result = StratColumnResult(success=False, error=str(exc))
        return self._result

    def generate_profile(self) -> List[Dict]:
        if not self._result or not self._result.layers:
            return []
        return [layer.model_dump() for layer in self._result.layers]

    def generate_overlay(self) -> List[Dict]:
        # Strat column doesn't have a map overlay
        return []

    def generate_summary(self) -> Dict:
        if not self._result:
            return {"success": False, "error": "Not run yet"}
        d: Dict = {"success": self._result.success, "error": self._result.error}
        if self._result.summary:
            d.update(self._result.summary.model_dump())
        return d

    # ────────────────────────────────────────────────────────────────
    # Core implementation
    # ────────────────────────────────────────────────────────────────

    def _run_impl(
        self,
        crater_lat: float,
        crater_lon: float,
        diameter_km: float,
        buffer_km: float,
        include_crism: bool,
        include_sharad: bool,
    ) -> StratColumnResult:
        from analysis.epsilon_terrace.pipeline import CraterStratigraphyAnalyzer

        logger.info(
            "Strat column: lat=%.3f lon=%.3f d=%.1fkm buffer=%.0fkm crism=%s sharad=%s",
            crater_lat, crater_lon, diameter_km, buffer_km,
            include_crism, include_sharad,
        )

        # ── Step 1: Run crater stratigraphy (terraces + εr) ───────
        strat = CraterStratigraphyAnalyzer()
        strat_result = strat.run(
            crater_lat=crater_lat,
            crater_lon=crater_lon,
            diameter_km=diameter_km,
            buffer_km=buffer_km,
        )

        crater_info = strat_result.crater_info
        rim_elevation_m = None
        dtm_source = "none"

        if crater_info:
            dtm_source = crater_info.dtm_source
            # Estimate rim elevation from radial profile (first point)
            if strat_result.radial_profile:
                rim_elevation_m = strat_result.radial_profile[0].elevation_m

        # ── Step 2: Build surface layers from terrace depths ──────
        layers: List[ColumnLayer] = []
        layer_idx = 0
        instruments_used: List[str] = []

        terrace_depths: List[Dict] = []
        if crater_info and crater_info.terrace_depths:
            terrace_depths = sorted(
                crater_info.terrace_depths,
                key=lambda t: t.get("depth_m", 0),
            )

        if terrace_depths:
            dtm_instrument = "HiRISE" if dtm_source == "HiRISE" else "MOLA"
            if dtm_instrument not in instruments_used:
                instruments_used.append(dtm_instrument)

            # First layer: rim to first terrace
            prev_depth = 0.0
            for td in terrace_depths:
                depth_m = td.get("depth_m", 0)
                if depth_m <= prev_depth:
                    continue
                layers.append(ColumnLayer(
                    layer_idx=layer_idx,
                    depth_top_m=round(prev_depth, 1),
                    depth_bottom_m=round(depth_m, 1),
                    thickness_m=round(depth_m - prev_depth, 1),
                    source="DTM_terrace",
                    instrument=dtm_instrument,
                    color=list(NO_DATA_COLOR),
                ))
                layer_idx += 1
                prev_depth = depth_m

            # Layer from deepest terrace to crater floor
            floor_depth = crater_info.depth_m if crater_info and crater_info.depth_m else None
            if floor_depth and floor_depth > prev_depth:
                layers.append(ColumnLayer(
                    layer_idx=layer_idx,
                    depth_top_m=round(prev_depth, 1),
                    depth_bottom_m=round(floor_depth, 1),
                    thickness_m=round(floor_depth - prev_depth, 1),
                    source="DTM_terrace",
                    instrument=dtm_instrument,
                    color=list(NO_DATA_COLOR),
                ))
                layer_idx += 1
        elif crater_info and crater_info.depth_m:
            # No terraces — single layer from rim to floor
            dtm_instrument = "HiRISE" if dtm_source == "HiRISE" else "MOLA"
            if dtm_instrument not in instruments_used:
                instruments_used.append(dtm_instrument)
            layers.append(ColumnLayer(
                layer_idx=layer_idx,
                depth_top_m=0.0,
                depth_bottom_m=round(crater_info.depth_m, 1),
                thickness_m=round(crater_info.depth_m, 1),
                source="DTM_terrace",
                instrument=dtm_instrument,
                color=list(NO_DATA_COLOR),
            ))
            layer_idx += 1

        # ── Step 3: CRISM mineral assignment ──────────────────────
        has_crism = False
        if include_crism:
            crism_minerals = self._find_crism_minerals(crater_lat, crater_lon, buffer_km)
            if crism_minerals:
                has_crism = True
                if "CRISM" not in instruments_used:
                    instruments_used.append("CRISM")
                # Assign dominant mineral to each layer
                for layer in layers:
                    if crism_minerals:
                        # Use the most common mineral across all CRISM obs
                        mineral_name, geochem_group, color = crism_minerals[0]
                        layer.mineral_name = mineral_name
                        layer.geochem_group = geochem_group
                        layer.color = list(color)

        # ── Step 4: SHARAD subsurface layers from εr ──────────────
        has_sharad = False
        if include_sharad and strat_result.epsilon_estimates:
            has_sharad = True
            if "SHARAD" not in instruments_used:
                instruments_used.append("SHARAD")

            # Get the deepest surface layer bottom
            deepest_surface = max((l.depth_bottom_m for l in layers), default=0.0)

            for est in strat_result.epsilon_estimates:
                depth_m = est.depth_m
                epsilon_r = est.epsilon_r
                material = _material_from_epsilon(epsilon_r)
                interpretation = est.interpretation

                # Subsurface layer extends below the deepest surface layer
                sub_top = deepest_surface
                sub_bottom = deepest_surface + depth_m
                if sub_bottom <= sub_top:
                    continue

                # Color by εr interpretation
                layer_color = [120, 120, 120, 180]  # default gray
                if epsilon_r < 3.5:
                    layer_color = [135, 206, 250, 200]  # ice-like (sky blue)
                elif epsilon_r < 5.0:
                    layer_color = [210, 180, 140, 200]  # mixed (tan)
                else:
                    layer_color = [139, 90, 43, 200]    # dense rock (brown)

                layers.append(ColumnLayer(
                    layer_idx=layer_idx,
                    depth_top_m=round(sub_top, 1),
                    depth_bottom_m=round(sub_bottom, 1),
                    thickness_m=round(sub_bottom - sub_top, 1),
                    source="SHARAD_reflector",
                    instrument="SHARAD",
                    epsilon_r=round(epsilon_r, 2),
                    material_class=material,
                    mineral_name=interpretation,
                    color=layer_color,
                    confidence=None,
                ))
                layer_idx += 1
                deepest_surface = sub_bottom  # Stack subsequent reflectors

        # ── Step 5: Compute summary ───────────────────────────────
        total_depth = max((l.depth_bottom_m for l in layers), default=0.0)
        dominant_material = None
        if layers:
            # Most thick layer's material
            thickest = max(layers, key=lambda l: l.thickness_m)
            dominant_material = thickest.material_class or thickest.mineral_name or thickest.geochem_group

        summary = ColumnSummary(
            crater_lat=crater_lat,
            crater_lon=crater_lon,
            diameter_km=diameter_km,
            n_layers=len(layers),
            total_depth_m=round(total_depth, 1),
            instruments_used=instruments_used,
            dtm_source=dtm_source,
            has_crism=has_crism,
            has_sharad_subsurface=has_sharad,
            dominant_material=dominant_material,
        )

        params = ColumnParameters(
            crater_lat=crater_lat,
            crater_lon=crater_lon,
            diameter_km=diameter_km,
            buffer_km=buffer_km,
            include_crism=include_crism,
            include_sharad=include_sharad,
        )

        logger.info(
            "Strat column: done — %d layers, %.1fm total depth, instruments=%s",
            len(layers), total_depth, instruments_used,
        )

        return StratColumnResult(
            success=True,
            summary=summary,
            layers=layers,
            rim_elevation_m=rim_elevation_m,
            parameters=params,
        )

    # ────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_crism_minerals(
        lat: float,
        lon: float,
        buffer_km: float,
    ) -> List[tuple]:
        """Find dominant minerals from cached CRISM CNN results near crater.

        Returns list of (mineral_name, geochem_group, color_rgba) tuples,
        sorted by frequency (most common first).
        """
        from api.mineral_cnn.pipeline import has_cached_result, load_cached_result
        from api.mineral_cnn.constants import CLASS_NAME
        from analysis.mineral_sequence.taxonomy import group_for_class

        import numpy as np

        # Search CRISM_TRR3 GeoJSON index for observations near crater
        try:
            from app import _geojson_cache
            crism_geojson = _geojson_cache.get("crism_trr3")
            if not crism_geojson:
                return []
        except Exception:
            return []

        nearby_obs: List[str] = []
        for feat in crism_geojson.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])

            # CRISM_TRR3 uses Point geometry with 0-360 lon
            if geom.get("type") == "Point" and len(coords) >= 2:
                obs_lon = coords[0] - 360 if coords[0] > 180 else coords[0]
                obs_lat = coords[1]
            else:
                continue

            dist = haversine_km(lat, lon, obs_lat, obs_lon)
            if dist <= buffer_km:
                obs_id = props.get("product_id", "")
                if obs_id:
                    # Strip suffix for CNN cache lookup
                    clean_id = obs_id.upper().replace("-", "_")
                    nearby_obs.append(clean_id)

        if not nearby_obs:
            return []

        # Find which have cached CNN results
        mineral_counts: Dict[int, int] = {}
        for obs_id in nearby_obs[:5]:  # Check up to 5 nearest
            if has_cached_result(obs_id):
                try:
                    result = load_cached_result(obs_id)
                    # Count minerals (exclude -1 and 100)
                    for mid in np.unique(result.mineral_map):
                        mid_int = int(mid)
                        if mid_int > 0 and mid_int != 100:
                            pixel_count = int((result.mineral_map == mid_int).sum())
                            mineral_counts[mid_int] = mineral_counts.get(mid_int, 0) + pixel_count
                except Exception:
                    continue

        if not mineral_counts:
            return []

        # Sort by count, return with group + color
        sorted_minerals = sorted(mineral_counts.items(), key=lambda x: x[1], reverse=True)
        results = []
        for mid, _ in sorted_minerals[:5]:
            name = CLASS_NAME.get(mid, f"Class {mid}")
            group = group_for_class(mid)
            color = GEOCHEM_COLORS.get(group or "Unknown", [120, 120, 120, 180])
            results.append((name, group, color))

        return results
