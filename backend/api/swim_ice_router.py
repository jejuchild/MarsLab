from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from analysis.swim_common.coord_utils import validate_region_size, validate_swim_bounds
from analysis.swim_fusion.models import CustomFusionRequest

router = APIRouter(prefix="/api/swim-ice", tags=["SWIM Ice Detection"])

_neutron_pipeline: object | None = None
_thermal_pipeline: object | None = None
_surface_pipeline: object | None = None
_dielectric_pipeline: object | None = None
_geomorphic_pipeline: object | None = None
_fusion_pipeline: object | None = None


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {"value": value}


def _call_method(target: object, name: str, **kwargs: object) -> object:
    method = getattr(target, name, None)
    if not callable(method):
        raise HTTPException(status_code=503, detail=f"SWIM endpoint unavailable: {name}")
    return method(**kwargs)


def _validate_point(lat: float, lon: float) -> None:
    valid, error = validate_swim_bounds(lat, lon)
    if not valid:
        raise HTTPException(status_code=422, detail=error)


def _validate_region(north: float, south: float, east: float, west: float) -> None:
    for lat, lon in ((north, west), (north, east), (south, west), (south, east)):
        valid, error = validate_swim_bounds(lat, lon)
        if not valid:
            raise HTTPException(status_code=422, detail=error)
    valid_size, size_error = validate_region_size(north=north, south=south, east=east, west=west)
    if not valid_size:
        raise HTTPException(status_code=422, detail=size_error)


def _ensure_loaded(geotiff: object | None, detail: str) -> None:
    if geotiff is None or not getattr(geotiff, "loaded", False):
        raise HTTPException(status_code=503, detail=detail)


def _get_neutron() -> object:
    global _neutron_pipeline
    if _neutron_pipeline is None:
        from analysis.swim_neutron.pipeline import SwimNeutronPipeline

        _neutron_pipeline = SwimNeutronPipeline()
    return _neutron_pipeline


def _get_thermal() -> object:
    global _thermal_pipeline
    if _thermal_pipeline is None:
        from analysis.swim_thermal.pipeline import SwimThermalPipeline

        _thermal_pipeline = SwimThermalPipeline()
    return _thermal_pipeline


def _get_surface() -> object:
    global _surface_pipeline
    if _surface_pipeline is None:
        try:
            from analysis.swim_sharad_surface.pipeline import SwimSharadSurfacePipeline
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"SWIM radar-surface module unavailable: {exc}")
        _surface_pipeline = SwimSharadSurfacePipeline()
    return _surface_pipeline


def _get_dielectric() -> object:
    global _dielectric_pipeline
    if _dielectric_pipeline is None:
        from analysis.swim_sharad_dielectric.pipeline import SwimSharadDielectricPipeline

        _dielectric_pipeline = SwimSharadDielectricPipeline()
    return _dielectric_pipeline


def _get_geomorphic() -> object:
    global _geomorphic_pipeline
    if _geomorphic_pipeline is None:
        try:
            from analysis.swim_geomorphic.pipeline import SwimGeomorphicPipeline
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"SWIM geomorphic module unavailable: {exc}")
        _geomorphic_pipeline = SwimGeomorphicPipeline()
    return _geomorphic_pipeline


def _get_fusion() -> object:
    global _fusion_pipeline
    if _fusion_pipeline is None:
        try:
            from analysis.swim_fusion.pipeline import SwimFusionPipeline
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"SWIM consistency module unavailable: {exc}")
        _fusion_pipeline = SwimFusionPipeline()
    return _fusion_pipeline


@router.get("/neutron/point")
def get_neutron_point(lat: float = Query(...), lon: float = Query(...)):
    _validate_point(lat, lon)
    pipeline = _get_neutron()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM neutron data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon)))


@router.get("/neutron/region")
def get_neutron_region(north: float, south: float, east: float, west: float):
    _validate_region(north, south, east, west)
    pipeline = _get_neutron()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM neutron data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_region", north=north, south=south, east=east, west=west)))


@router.get("/neutron/tile/{z}/{x}/{y}.png")
def get_neutron_tile(z: int, x: int, y: int):
    pipeline = _get_neutron()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM neutron data not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM neutron tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/thermal/point")
def get_thermal_point(lat: float = Query(...), lon: float = Query(...)):
    _validate_point(lat, lon)
    pipeline = _get_thermal()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM thermal data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon)))


@router.get("/thermal/tile/{z}/{x}/{y}.png")
def get_thermal_tile(z: int, x: int, y: int):
    pipeline = _get_thermal()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM thermal data not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM thermal tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/thermal/region")
def get_thermal_region(north: float, south: float, east: float, west: float):
    _validate_region(north, south, east, west)
    pipeline = _get_thermal()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM thermal data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_region", north=north, south=south, east=east, west=west)))


@router.get("/radar-surface/point")
def get_radar_surface_point(lat: float = Query(...), lon: float = Query(...)):
    _validate_point(lat, lon)
    pipeline = _get_surface()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM radar-surface data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon)))


@router.get("/radar-surface/tile/{z}/{x}/{y}.png")
def get_radar_surface_tile(z: int, x: int, y: int):
    pipeline = _get_surface()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM radar-surface data not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM radar-surface tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/radar-surface/region")
def get_radar_surface_region(north: float, south: float, east: float, west: float):
    _validate_region(north, south, east, west)
    pipeline = _get_surface()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_geotiff", None), "SWIM radar-surface data not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_region", north=north, south=south, east=east, west=west)))


@router.get("/radar-dielectric/point")
def get_radar_dielectric_point(lat: float = Query(...), lon: float = Query(...)):
    _validate_point(lat, lon)
    pipeline = _get_dielectric()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, "_dielectric_1_5m", None), "SWIM radar-dielectric data (1-5m) not loaded")
    _ensure_loaded(getattr(pipeline, "_dielectric_5m_plus", None), "SWIM radar-dielectric data (5m-plus) not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon)))


@router.get("/radar-dielectric/tile/{z}/{x}/{y}.png")
def get_radar_dielectric_tile(z: int, x: int, y: int, depth: Literal["1-5m", "5m-plus"] = Query(...)):
    pipeline = _get_dielectric()
    geotiff = _call_method(pipeline, "_get_geotiff_for_depth", depth=depth)
    _ensure_loaded(geotiff, f"SWIM radar-dielectric data ({depth}) not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y, depth=depth)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM radar-dielectric tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/radar-dielectric/region")
def get_radar_dielectric_region(
    north: float,
    south: float,
    east: float,
    west: float,
    depth: Literal["1-5m", "5m-plus"] = Query(...),
):
    _validate_region(north, south, east, west)
    pipeline = _get_dielectric()
    geotiff = _call_method(pipeline, "_get_geotiff_for_depth", depth=depth)
    _ensure_loaded(geotiff, f"SWIM radar-dielectric data ({depth}) not loaded")
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_region", north=north, south=south, east=east, west=west, depth=depth)))


@router.get("/geomorphic/point")
def get_geomorphic_point(lat: float = Query(...), lon: float = Query(...)):
    _validate_point(lat, lon)
    pipeline = _get_geomorphic()
    ensure_loaded = getattr(pipeline, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    _ensure_loaded(getattr(pipeline, '_geomorphology_0_1m', None), 'SWIM geomorphic data not loaded')
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon)))


@router.get("/geomorphic/tile/{z}/{x}/{y}.png")
def get_geomorphic_tile(z: int, x: int, y: int, depth: Literal["0-1m", "1-5m", "5m-plus"] = Query(...)):
    pipeline = _get_geomorphic()
    geotiff = _call_method(pipeline, "_get_geotiff_for_depth", depth=depth)
    _ensure_loaded(geotiff, f"SWIM geomorphic data ({depth}) not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y, depth=depth)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM geomorphic tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/geomorphic/landforms")
def get_geomorphic_landforms(lat: float = Query(...), lon: float = Query(...), radius_km: float = Query(50.0, ge=0.1, le=500.0)):
    _validate_point(lat, lon)
    pipeline = _get_geomorphic()
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_landforms", lat=lat, lon=lon, radius_km=radius_km)))


@router.get("/consistency/point")
def get_consistency_point(lat: float = Query(...), lon: float = Query(...), mode: Literal["precomputed", "live"] = Query("precomputed")):
    _validate_point(lat, lon)
    pipeline = _get_fusion()
    result = _as_dict(_call_method(pipeline, "query_point", lat=lat, lon=lon, mode=mode))
    if mode == "precomputed" and result.get("consistency_0_1m") is None and result.get("consistency_1_5m") is None and result.get("consistency_5m_plus") is None:
        raise HTTPException(status_code=503, detail="SWIM consistency data not loaded")
    return JSONResponse(content=result)


@router.get("/consistency/tile/{z}/{x}/{y}.png")
def get_consistency_tile(z: int, x: int, y: int, depth: Literal["0-1m", "1-5m", "5m-plus"] = Query(...)):
    pipeline = _get_fusion()
    geotiff = _call_method(pipeline, "_get_geotiff_for_depth", depth=depth)
    _ensure_loaded(geotiff, f"SWIM consistency data ({depth}) not loaded")
    png_bytes = _call_method(pipeline, "get_tile", z=z, x=x, y=y, depth=depth)
    if not isinstance(png_bytes, (bytes, bytearray)):
        raise HTTPException(status_code=404, detail="SWIM consistency tile unavailable")
    return Response(content=bytes(png_bytes), media_type="image/png")


@router.get("/consistency/region")
def get_consistency_region(north: float, south: float, east: float, west: float):
    _validate_region(north, south, east, west)
    pipeline = _get_fusion()
    return JSONResponse(content=_as_dict(_call_method(pipeline, "query_region", north=north, south=south, east=east, west=west)))


@router.post("/consistency/custom")
def post_custom_consistency(request: CustomFusionRequest):
    _validate_point(request.lat, request.lon)
    pipeline = _get_fusion()
    method = getattr(pipeline, "query_custom", None)
    if not callable(method):
        method = getattr(pipeline, "custom_fusion", None)
    if not callable(method):
        raise HTTPException(status_code=503, detail="SWIM consistency custom fusion endpoint unavailable")
    return JSONResponse(content=_as_dict(method(request)))
