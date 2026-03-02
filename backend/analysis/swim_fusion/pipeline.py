from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Set
from typing import Literal, Protocol, runtime_checkable, cast

from ..swim_common import SwimGeoTIFF, load_swim_geotiff
from ..swim_common.coord_utils import clamp_to_swim_region, validate_region_size, validate_swim_bounds
from .models import ConsistencyPointResponse, ConsistencyRegionResponse
from .weights import ALL_SWIM_METHODS, SWIM_WEIGHTS

logger = logging.getLogger(__name__)


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self) -> dict[str, object]:
        ...


@runtime_checkable
class _RegionSampler(Protocol):
    def sample_region(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
    ) -> dict[str, object]:
        ...


class SwimFusionPipeline:
    def __init__(self) -> None:
        self._precomputed: dict[str, SwimGeoTIFF | None] = {
            "0-1m": self._load_precomputed_depth(
                "0-1m",
                ("consistency_0_1m.tif", "SWIM4MIM_Ci_0_1.tif"),
            ),
            "1-5m": self._load_precomputed_depth(
                "1-5m",
                ("consistency_1_5m.tif", "SWIM4MIM_Ci_1_5.tif"),
            ),
            ">5m": self._load_precomputed_depth(
                ">5m",
                ("consistency_5m_plus.tif", "SWIM4MIM_Ci_5.tif"),
            ),
        }
        self._live_pipelines: dict[str, object | None] = {}

    def query_point(self, lat: float, lon: float, mode: str = "precomputed") -> ConsistencyPointResponse:
        if mode == "live":
            return self._query_point_live(lat, lon)
        return self._query_point_precomputed(lat, lon)

    def custom_query(
        self,
        lat: float,
        lon: float,
        enabled_methods: Iterable[str],
        custom_weights: dict[str, float] | None = None,
    ) -> ConsistencyPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return self._empty_point_response(lat, lon, mode="live")

        enabled_set = set(enabled_methods)
        if not enabled_set:
            enabled_set = set(ALL_SWIM_METHODS)

        unknown = sorted(enabled_set - set(ALL_SWIM_METHODS))
        if unknown:
            raise ValueError(f"Unknown methods in enabled_methods: {unknown}")

        if custom_weights is not None and sum(custom_weights.values()) <= 0:
            raise ValueError("custom_weights sum must be > 0")

        all_scores = self._get_live_method_scores(lat, lon)
        filtered_scores: dict[str, float | None] = {
            method: (all_scores.get(method) if method in enabled_set else None)
            for method in ALL_SWIM_METHODS
        }

        c_0_1 = self._fuse_depth(filtered_scores, "0-1m", enabled_set, custom_weights)
        c_1_5 = self._fuse_depth(filtered_scores, "1-5m", enabled_set, custom_weights)
        c_5 = self._fuse_depth(filtered_scores, ">5m", enabled_set, custom_weights)

        return ConsistencyPointResponse(
            lat=lat,
            lon=lon,
            consistency_0_1m=c_0_1,
            consistency_1_5m=c_1_5,
            consistency_5m_plus=c_5,
            method_scores=filtered_scores,
            mode="live",
            depth_to_ice_estimate_m=self._estimate_depth_to_ice(c_0_1, c_1_5, c_5),
        )

    def query_region(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
    ) -> ConsistencyRegionResponse:
        ok, msg = validate_region_size(north, south, east, west)
        if not ok:
            raise ValueError(msg)

        north, south, east, west = clamp_to_swim_region(north, south, east, west)

        return ConsistencyRegionResponse(
            bounds={"north": north, "south": south, "east": east, "west": west},
            stats_0_1m=self._region_stats_for_depth("0-1m", north, south, east, west),
            stats_1_5m=self._region_stats_for_depth("1-5m", north, south, east, west),
            stats_5m_plus=self._region_stats_for_depth(">5m", north, south, east, west),
            tile_urls={
                "0-1m": "/api/swim-ice/consistency/tile/{z}/{x}/{y}.png?depth=0-1m",
                "1-5m": "/api/swim-ice/consistency/tile/{z}/{x}/{y}.png?depth=1-5m",
                "5m-plus": "/api/swim-ice/consistency/tile/{z}/{x}/{y}.png?depth=5m-plus",
            },
        )

    def _query_point_precomputed(self, lat: float, lon: float) -> ConsistencyPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return self._empty_point_response(lat, lon, mode="precomputed")

        c_0_1 = self._sample_precomputed("0-1m", lat, lon)
        c_1_5 = self._sample_precomputed("1-5m", lat, lon)
        c_5 = self._sample_precomputed(">5m", lat, lon)

        return ConsistencyPointResponse(
            lat=lat,
            lon=lon,
            consistency_0_1m=c_0_1,
            consistency_1_5m=c_1_5,
            consistency_5m_plus=c_5,
            method_scores={method: None for method in ALL_SWIM_METHODS},
            mode="precomputed",
            depth_to_ice_estimate_m=self._estimate_depth_to_ice(c_0_1, c_1_5, c_5),
        )

    def _query_point_live(self, lat: float, lon: float) -> ConsistencyPointResponse:
        valid, _ = validate_swim_bounds(lat, lon)
        if not valid:
            return self._empty_point_response(lat, lon, mode="live")

        method_scores = self._get_live_method_scores(lat, lon)
        c_0_1 = self._fuse_depth(method_scores, "0-1m")
        c_1_5 = self._fuse_depth(method_scores, "1-5m")
        c_5 = self._fuse_depth(method_scores, ">5m")

        return ConsistencyPointResponse(
            lat=lat,
            lon=lon,
            consistency_0_1m=c_0_1,
            consistency_1_5m=c_1_5,
            consistency_5m_plus=c_5,
            method_scores=method_scores,
            mode="live",
            depth_to_ice_estimate_m=self._estimate_depth_to_ice(c_0_1, c_1_5, c_5),
        )

    def _empty_point_response(
        self,
        lat: float,
        lon: float,
        mode: Literal["precomputed", "live"],
    ) -> ConsistencyPointResponse:
        return ConsistencyPointResponse(
            lat=lat,
            lon=lon,
            consistency_0_1m=None,
            consistency_1_5m=None,
            consistency_5m_plus=None,
            method_scores={method: None for method in ALL_SWIM_METHODS},
            mode=mode,
            depth_to_ice_estimate_m=None,
        )

    def _load_precomputed_depth(self, depth_label: str, filenames: Iterable[str]) -> SwimGeoTIFF | None:
        loaded: SwimGeoTIFF | None = None
        for filename in filenames:
            geotiff = load_swim_geotiff(filename, f"SWIM consistency {depth_label}")
            loaded = geotiff
            if geotiff.loaded:
                return geotiff
        return loaded

    def _sample_precomputed(self, depth: str, lat: float, lon: float) -> float | None:
        geotiff = self._precomputed.get(depth)
        if geotiff is None or not geotiff.loaded:
            return None
        value = geotiff.sample_point(lat, lon)
        if value is None:
            return None
        return round(float(value), 4)

    def _region_stats_for_depth(
        self,
        depth: str,
        north: float,
        south: float,
        east: float,
        west: float,
    ) -> dict[str, object]:
        geotiff = self._precomputed.get(depth)
        if geotiff is None or not geotiff.loaded:
            return {"available": False, "error": f"Precomputed layer unavailable: {depth}"}

        sampler = cast(_RegionSampler, geotiff)
        result = sampler.sample_region(north, south, east, west)
        available_obj = result.get("available")
        if isinstance(available_obj, bool) and available_obj:
            stats_obj = result.get("stats")
            if isinstance(stats_obj, dict):
                return cast(dict[str, object], stats_obj)

        return result

    def _init_live_pipelines(self) -> None:
        if self._live_pipelines:
            return

        specs: dict[str, tuple[str, str]] = {
            "neutron": ("analysis.swim_neutron.pipeline", "SwimNeutronPipeline"),
            "thermal": ("analysis.swim_thermal.pipeline", "SwimThermalPipeline"),
            "radar_surface": ("analysis.swim_sharad_surface.pipeline", "SwimSharadSurfacePipeline"),
            "radar_dielectric": ("analysis.swim_sharad_dielectric.pipeline", "SwimSharadDielectricPipeline"),
            "geomorphic": ("analysis.swim_geomorphic.pipeline", "SwimGeomorphicPipeline"),
        }

        for method_key, (module_path, class_name) in specs.items():
            try:
                module = importlib.import_module(module_path)
                cls_obj: object = getattr(module, class_name, None)
                if cls_obj is None or not callable(cls_obj):
                    self._live_pipelines[method_key] = None
                    continue
                factory = cast(Callable[[], object], cls_obj)
                self._live_pipelines[method_key] = factory()
            except Exception as exc:
                logger.warning("Unable to initialize %s pipeline: %s", method_key, exc)
                self._live_pipelines[method_key] = None

    def _get_live_method_scores(self, lat: float, lon: float) -> dict[str, float | None]:
        self._init_live_pipelines()
        scores: dict[str, float | None] = {method: None for method in ALL_SWIM_METHODS}

        neutron_payload = self._query_pipeline_payload(self._live_pipelines.get("neutron"), lat, lon)
        scores["neutron"] = self._extract_method_score(neutron_payload, "neutron")

        thermal_payload = self._query_pipeline_payload(self._live_pipelines.get("thermal"), lat, lon)
        scores["thermal"] = self._extract_method_score(thermal_payload, "thermal")

        radar_surface_payload = self._query_pipeline_payload(self._live_pipelines.get("radar_surface"), lat, lon)
        scores["radar_surface"] = self._extract_method_score(radar_surface_payload, "radar_surface")

        radar_dielectric_payload = self._query_pipeline_payload(self._live_pipelines.get("radar_dielectric"), lat, lon)
        scores["radar_dielectric"] = self._extract_method_score(radar_dielectric_payload, "radar_dielectric")

        geomorphic_payload = self._query_pipeline_payload(self._live_pipelines.get("geomorphic"), lat, lon)
        scores["geomorphic_shallow"] = self._extract_method_score(geomorphic_payload, "geomorphic_shallow")
        scores["geomorphic_deep"] = self._extract_method_score(geomorphic_payload, "geomorphic_deep")

        return scores

    def _query_pipeline_payload(self, pipeline: object | None, lat: float, lon: float) -> object | None:
        if pipeline is None:
            return None

        query_fn_obj = getattr(pipeline, "query_point", None)
        if callable(query_fn_obj):
            for kwargs in ({"lat": lat, "lon": lon}, {}):
                try:
                    if kwargs:
                        return query_fn_obj(**kwargs)
                    return query_fn_obj(lat, lon)
                except TypeError:
                    continue
                except Exception:
                    return None

        run_fn_obj = getattr(pipeline, "run", None)
        if callable(run_fn_obj):
            for kwargs in ({"lat": lat, "lon": lon}, {}):
                try:
                    if kwargs:
                        return run_fn_obj(**kwargs)
                    return run_fn_obj(lat, lon)
                except TypeError:
                    continue
                except Exception:
                    return None

        return None

    def _fuse_depth(
        self,
        method_scores: dict[str, float | None],
        depth: str,
        enabled_methods: Set[str] | None = None,
        custom_weights: dict[str, float] | None = None,
    ) -> float | None:
        numerator = 0.0
        denominator = 0.0

        depth_weights = SWIM_WEIGHTS[depth]
        for method, default_weight in depth_weights.items():
            if enabled_methods is not None and method not in enabled_methods:
                continue

            score = method_scores.get(method)
            if score is None:
                continue

            weight = custom_weights.get(method, default_weight) if custom_weights is not None else default_weight
            if weight <= 0:
                continue

            numerator += weight * score
            denominator += weight

        if denominator <= 0:
            return None
        return round(numerator / denominator, 4)

    def _extract_method_score(self, payload: object | None, method: str) -> float | None:
        if payload is None:
            return None

        normalized = self._normalize_payload(payload)
        if isinstance(normalized, (int, float)):
            return self._to_score(float(normalized))
        if not isinstance(normalized, dict):
            return None
        data = cast(dict[str, object], normalized)

        method_scores_obj = data.get("method_scores")
        if isinstance(method_scores_obj, dict):
            method_scores_dict = cast(dict[str, object], method_scores_obj)
            direct = self._to_score(method_scores_dict.get(method))
            if direct is not None:
                return direct

        direct = self._to_score(data.get(method))
        if direct is not None:
            return direct

        key_candidates: dict[str, tuple[str, ...]] = {
            "neutron": ("consistency", "score", "value", "consistency_score", "neutron_score"),
            "thermal": ("consistency", "score", "value", "consistency_score", "thermal_score"),
            "radar_surface": (
                "radar_surface",
                "surface_power_consistency",
                "consistency",
                "score",
                "value",
            ),
            "radar_dielectric": (
                "radar_dielectric",
                "dielectric_consistency",
                "consistency",
                "score",
                "value",
            ),
            "geomorphic_shallow": (
                "geomorphic_shallow",
                "consistency_0_1m",
                "consistency_1_5m",
                "shallow_consistency",
            ),
            "geomorphic_deep": (
                "geomorphic_deep",
                "consistency_5m_plus",
                "deep_consistency",
            ),
        }

        for key in key_candidates.get(method, ()):
            candidate = self._to_score(data.get(key))
            if candidate is not None:
                return candidate
        return None

    def _normalize_payload(self, payload: object) -> object:
        if isinstance(payload, dict):
            return cast(dict[str, object], payload)
        if isinstance(payload, _ModelDumpable):
            return payload.model_dump()
        return payload

    def _to_score(self, value: object) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return round(max(-1.0, min(1.0, float(value))), 4)
        return None

    def _estimate_depth_to_ice(
        self,
        consistency_0_1m: float | None,
        consistency_1_5m: float | None,
        consistency_5m_plus: float | None,
    ) -> float | None:
        weighted_sum = 0.0
        positive_weight = 0.0

        for consistency, center_m in (
            (consistency_0_1m, 0.5),
            (consistency_1_5m, 3.0),
            (consistency_5m_plus, 7.5),
        ):
            if consistency is None:
                continue
            weight = max(consistency, 0.0)
            if weight <= 0:
                continue
            weighted_sum += weight * center_m
            positive_weight += weight

        if positive_weight <= 0:
            return None
        return round(weighted_sum / positive_weight, 3)
