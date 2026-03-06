"""
FastAPI router for MarsLab Integration Modules.

Exposes all four cross-system analysis modules:
  A. /api/integration/landing-site     — Landing Site Suitability Scorer
  B. /api/integration/mineral-stability — Climate-Mineral Stability Map
  C. /api/integration/ice-evolution     — Subsurface Ice Evolution Model
  D. /api/integration/seismic-surface   — Seismic Risk + Surface Correlation
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/integration", tags=["Integration Modules"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PointQuery(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(..., ge=-360, le=360, description="Longitude in degrees")
    ls: float = Field(0.0, ge=0, le=360, description="Solar longitude (season)")


class SiteEntry(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-360, le=360)
    name: str = ""


class MultiSiteQuery(BaseModel):
    sites: List[SiteEntry]
    ls: float = Field(0.0, ge=0, le=360)


# ---------------------------------------------------------------------------
# A. Landing Site Scorer
# ---------------------------------------------------------------------------

@router.post("/landing-site/score")
async def score_landing_site(query: PointQuery) -> JSONResponse:
    """Score a single landing site for suitability (0-100)."""
    from .landing_site_scorer import score_landing_site as _score

    result = _score(lat=query.lat, lon=query.lon, ls=query.ls)
    return JSONResponse(content=_dataclass_to_dict(result))


@router.post("/landing-site/compare")
async def compare_landing_sites(query: MultiSiteQuery) -> JSONResponse:
    """Score and rank multiple landing sites."""
    from .landing_site_scorer import compare_sites as _compare

    sites = [{"lat": s.lat, "lon": s.lon} for s in query.sites]
    results = _compare(sites, ls=query.ls)
    return JSONResponse(content=[_dataclass_to_dict(r) for r in results])


# ---------------------------------------------------------------------------
# B. Mineral Stability
# ---------------------------------------------------------------------------

@router.post("/mineral-stability/assess")
async def assess_mineral_stability(query: PointQuery) -> JSONResponse:
    """Assess mineral phase stability at a location."""
    from .mineral_stability import assess_mineral_stability as _assess

    result = _assess(lat=query.lat, lon=query.lon, ls=query.ls)
    return JSONResponse(content=_dataclass_to_dict(result))


@router.post("/mineral-stability/seasonal")
async def seasonal_stability(query: PointQuery) -> JSONResponse:
    """Compute mineral stability across a full Mars year."""
    from .mineral_stability import seasonal_stability_profile as _seasonal

    results = _seasonal(lat=query.lat, lon=query.lon, n_seasons=12)
    return JSONResponse(content=[_dataclass_to_dict(r) for r in results])


@router.post("/mineral-stability/brine-window")
async def brine_window(query: PointQuery) -> JSONResponse:
    """Determine seasonal brine habitability window."""
    from .mineral_stability import brine_habitability_window as _brine

    result = _brine(lat=query.lat, lon=query.lon, n_seasons=36)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# C. Ice Evolution
# ---------------------------------------------------------------------------

@router.post("/ice-evolution/stability")
async def ice_stability(query: PointQuery) -> JSONResponse:
    """Compute subsurface ice stability depth."""
    from .ice_evolution import compute_ice_stability as _compute

    result = _compute(lat=query.lat, lon=query.lon, ls=query.ls)
    return JSONResponse(content=_dataclass_to_dict(result))


@router.post("/ice-evolution/assess")
async def assess_ice_evolution(query: PointQuery) -> JSONResponse:
    """Full ice evolution assessment with observational validation."""
    from .ice_evolution import assess_ice_evolution as _assess

    result = _assess(lat=query.lat, lon=query.lon, ls=query.ls)
    return JSONResponse(content=_dataclass_to_dict(result))


@router.post("/ice-evolution/seasonal")
async def ice_seasonal(query: PointQuery) -> JSONResponse:
    """Compute ice stability depth across a full Mars year."""
    from .ice_evolution import depth_stability_map as _seasonal

    result = _seasonal(lat=query.lat, lon=query.lon, n_seasons=12)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# D. Seismic-Surface Correlation
# ---------------------------------------------------------------------------

@router.post("/seismic-surface/assess")
async def assess_seismic_surface(query: PointQuery) -> JSONResponse:
    """Assess seismic risk and surface feature correlation."""
    from .seismic_surface import assess_seismic_surface as _assess

    result = _assess(lat=query.lat, lon=query.lon)
    return JSONResponse(content=_dataclass_to_dict(result))


@router.post("/seismic-surface/compare")
async def compare_seismic_risk(query: MultiSiteQuery) -> JSONResponse:
    """Rank multiple sites by seismic-geological risk (safest first)."""
    from .seismic_surface import compare_seismic_risk as _compare

    sites = [{"lat": s.lat, "lon": s.lon} for s in query.sites]
    results = _compare(sites)
    return JSONResponse(content=[_dataclass_to_dict(r) for r in results])


# ---------------------------------------------------------------------------
# Combined endpoint
# ---------------------------------------------------------------------------

@router.post("/full-assessment")
async def full_assessment(query: PointQuery) -> JSONResponse:
    """Run ALL four integration modules for a single location.

    Returns a comprehensive assessment combining landing site scoring,
    mineral stability, ice evolution, and seismic risk.
    """
    from .landing_site_scorer import score_landing_site as _score_landing
    from .mineral_stability import assess_mineral_stability as _assess_mineral
    from .ice_evolution import assess_ice_evolution as _assess_ice
    from .seismic_surface import assess_seismic_surface as _assess_seismic

    landing = _score_landing(lat=query.lat, lon=query.lon, ls=query.ls)
    mineral = _assess_mineral(lat=query.lat, lon=query.lon, ls=query.ls)
    ice = _assess_ice(lat=query.lat, lon=query.lon, ls=query.ls)
    seismic = _assess_seismic(lat=query.lat, lon=query.lon)

    return JSONResponse(content={
        "lat": query.lat,
        "lon": query.lon,
        "ls": query.ls,
        "landing_site": _dataclass_to_dict(landing),
        "mineral_stability": _dataclass_to_dict(mineral),
        "ice_evolution": _dataclass_to_dict(ice),
        "seismic_surface": _dataclass_to_dict(seismic),
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataclass_to_dict(obj) -> dict:
    """Recursively convert dataclass to dict, handling nested dataclasses."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _convert_value(value)
        return result
    return obj


def _convert_value(value):
    """Convert a value for JSON serialization."""
    if hasattr(value, "__dataclass_fields__"):
        return _dataclass_to_dict(value)
    if isinstance(value, list):
        return [_convert_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value
