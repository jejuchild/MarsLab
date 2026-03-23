"""
AI-assisted natural language search for Mars data.

Uses Google Gemini API for natural language understanding.
Gemini is used ONLY for parsing intent - it does NOT download files.

Parses user queries to identify:
- Target instrument(s)
- Geographic region or coordinates
- Optional constraints

Then translates to existing search primitives.
"""

import re
import asyncio
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import aiohttp

from .mars_regions import find_region, MarsRegion, get_region_by_id, MARS_REGIONS
from .ode_client import (
    Instrument,
    search_ode_spatial,
    search_sharad_spatial,
    search_sharad_highres_spatial,
)
from .download_manager import check_local_existence_detailed
from .gemini_parser import (
    parse_with_gemini,
    is_gemini_available,
    get_gemini_status,
    GeminiPlan,
    GeminiCrossFilter,
    validate_plan,
    normalize_longitude,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai-search"])


# =============================================================================
# Models
# =============================================================================

class AISearchRequest(BaseModel):
    """Request for AI-assisted search."""
    query: str
    max_results: int = 10


class ParsedIntent(BaseModel):
    """Parsed intent from natural language query."""
    instruments: List[str]  # crism, hirise, sharad, sharad_highres
    region_name: Optional[str] = None
    region_lat: Optional[float] = None
    region_lon: Optional[float] = None
    radius_deg: float = 2.0
    raw_query: str = ""


class AISearchResult(BaseModel):
    """Single search result."""
    product_id: str
    instrument: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_km: Optional[float] = None
    exists: bool = False
    has_core: bool = False
    has_browse: bool = False
    missing_files: List[str] = []


class AISearchResponse(BaseModel):
    """Response from AI search."""
    parsed_intent: ParsedIntent
    total_count: int
    shown_count: int
    truncated: bool
    results: List[AISearchResult]
    message: str = ""


# =============================================================================
# Natural Language Parser
# =============================================================================

# Instrument keywords mapping
INSTRUMENT_KEYWORDS = {
    "crism": ["crism", "spectrometer", "spectral", "mineral", "mineralogy", "composition", "hyperspectral"],
    "hirise": ["hirise", "high resolution", "high-resolution", "camera", "image", "imagery", "photo", "photograph", "surface detail"],
    "sharad": ["sharad", "radar", "subsurface", "sounder", "sounding", "ice", "radargram"],
    "sharad_highres": ["sharad high-res", "sharad highres", "sharad hi-res", "high-res radar", "full radar", "rdr"],
    "ctx": ["ctx", "context camera", "context", "wide-angle", "wide angle"],
    "hirise_dtm": ["dtm", "digital terrain", "elevation model", "terrain model", "hirise dtm", "topography"],
}

# Action keywords that help identify intent
ACTION_KEYWORDS = {
    "download": ["download", "get", "fetch", "retrieve", "obtain", "grab"],
    "search": ["search", "find", "look for", "locate", "show", "list"],
}


def parse_natural_language(query: str) -> ParsedIntent:
    """
    Parse natural language query to extract search intent.

    Args:
        query: User's natural language query

    Returns:
        ParsedIntent with extracted instruments, region, etc.
    """
    query_lower = query.lower()
    intent = ParsedIntent(instruments=[], raw_query=query)

    # 1) Identify instruments
    for instrument, keywords in INSTRUMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                if instrument not in intent.instruments:
                    intent.instruments.append(instrument)
                break

    # If no instrument specified, default to all
    if not intent.instruments:
        intent.instruments = ["crism", "hirise", "sharad", "sharad_highres"]

    # 2) Identify region
    # Try to find named regions from our lookup table
    region = None

    # Check for coordinate patterns first (e.g., "at 18.5, 77.4" or "lat 18.5 lon 77.4")
    coord_patterns = [
        r"at\s+(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)",  # at 18.5, 77.4
        r"lat(?:itude)?\s*[:=]?\s*(-?\d+\.?\d*)\s*[,\s]*lon(?:gitude)?\s*[:=]?\s*(-?\d+\.?\d*)",  # lat 18.5 lon 77.4
        r"(-?\d+\.?\d*)°?\s*[NS]?\s*[,\s]\s*(-?\d+\.?\d*)°?\s*[EW]?",  # 18.5° N, 77.4° E
        r"coordinates?\s*[:=]?\s*\(?(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\)?",  # coordinates: (18.5, 77.4)
    ]

    for pattern in coord_patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                intent.region_lat = float(match.group(1))
                intent.region_lon = float(match.group(2))
                intent.region_name = f"Coordinates ({intent.region_lat}, {intent.region_lon})"
                break
            except (ValueError, IndexError):
                continue

    # If no coordinates found, look for named regions
    if intent.region_lat is None:
        # Extract potential region names (remove common words)
        words_to_remove = set(ACTION_KEYWORDS["download"] + ACTION_KEYWORDS["search"] +
                              ["data", "over", "around", "near", "in", "at", "the", "from", "for", "and", "or", "with"])
        for instrument_keywords in INSTRUMENT_KEYWORDS.values():
            words_to_remove.update(instrument_keywords)

        # Try to find region in the remaining text
        # Split by common delimiters and try each phrase
        potential_regions = []

        # Try the whole query first (minus known keywords)
        remaining = query_lower
        for word in words_to_remove:
            remaining = remaining.replace(word, " ")
        remaining = " ".join(remaining.split())  # Normalize whitespace

        if remaining.strip():
            potential_regions.append(remaining.strip())

        # Also try specific phrases
        region_patterns = [
            r"(?:over|around|near|in|at)\s+(.+?)(?:\s+(?:region|area|crater|planitia|terra|mons|vallis|chasma))?(?:\s*$|\s+(?:and|with|for))",
            r"(.+?)\s+(?:region|area|crater|planitia|terra|mons|vallis|chasma)",
        ]

        for pattern in region_patterns:
            matches = re.findall(pattern, query_lower)
            potential_regions.extend(matches)

        # Try to match each potential region
        for potential in potential_regions:
            region = find_region(potential)
            if region:
                break

    if region:
        intent.region_name = region.display_name
        intent.region_lat = region.center_lat
        # Use center_lon in -180/180 for internal, convert for ODE later
        lon_180 = region.center_lon
        if lon_180 < 0:
            lon_180 += 360
        intent.region_lon = lon_180
        intent.radius_deg = region.radius_deg

    # 3) Look for radius specification
    radius_match = re.search(r"radius\s*[:=]?\s*(\d+\.?\d*)\s*(?:deg|degrees?|°)?", query_lower)
    if radius_match:
        try:
            intent.radius_deg = float(radius_match.group(1))
        except ValueError:
            pass

    return intent


# =============================================================================
# Search Execution
# =============================================================================

async def execute_ai_search(intent: ParsedIntent, max_results: int = 10) -> AISearchResponse:
    """
    Execute search based on parsed intent.

    Args:
        intent: Parsed search intent
        max_results: Maximum number of results to return

    Returns:
        AISearchResponse with results
    """
    results: List[AISearchResult] = []

    # If no region specified, return error
    if intent.region_lat is None or intent.region_lon is None:
        return AISearchResponse(
            parsed_intent=intent,
            total_count=0,
            shown_count=0,
            truncated=False,
            results=[],
            message="Could not identify a region from your query. Please specify a Mars region (e.g., 'Jezero Crater', 'Arcadia Planitia') or coordinates."
        )

    # Calculate bounding box from center point and radius
    lat = intent.region_lat
    lon = intent.region_lon
    radius = intent.radius_deg

    minlat = max(-90, lat - radius)
    maxlat = min(90, lat + radius)
    westernlon = lon - radius
    easternlon = lon + radius

    # Normalize longitudes to 0-360
    if westernlon < 0:
        westernlon += 360
    if easternlon < 0:
        easternlon += 360

    # Search each requested instrument
    async with aiohttp.ClientSession() as session:
        for instrument in intent.instruments:
            try:
                products = []

                if instrument == "crism":
                    products = await search_ode_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        Instrument.CRISM,
                        max_results=max_results * 2,  # Get extra to account for filtering
                        session=session
                    )
                elif instrument == "hirise":
                    products = await search_ode_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        Instrument.HIRISE,
                        max_results=max_results * 2,
                        session=session
                    )
                elif instrument == "sharad":
                    products = await search_sharad_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        max_results=max_results * 2,
                        session=session
                    )
                elif instrument in ("sharad_highres", "ctx", "hirise_dtm"):
                    # Local-index instruments: convert lon to -180..180 for local search
                    lon_min_180 = westernlon if westernlon <= 180 else westernlon - 360
                    lon_max_180 = easternlon if easternlon <= 180 else easternlon - 360
                    local_results = _search_local_index(
                        instrument.upper(), minlat, maxlat,
                        lon_min_180, lon_max_180,
                        lat, lon if lon <= 180 else lon - 360,
                        max_results * 2,
                    )
                    for lr in local_results:
                        results.append(AISearchResult(**lr.model_dump()))
                    continue

                # Convert to AISearchResult
                for p in products:
                    # Check local existence
                    inst_enum = Instrument(instrument)
                    existence = check_local_existence_detailed(p.product_id, inst_enum)

                    # Calculate distance if we have coordinates
                    distance_km = None
                    if p.lat is not None and p.lon is not None:
                        distance_km = haversine_distance_km(lat, lon, p.lat, p.lon)

                    results.append(AISearchResult(
                        product_id=p.product_id,
                        instrument=instrument,
                        lat=p.lat,
                        lon=p.lon,
                        distance_km=round(distance_km, 2) if distance_km else None,
                        exists=existence.exists,
                        has_core=existence.has_core,
                        has_browse=existence.has_browse,
                        missing_files=existence.missing_files,
                    ))

            except Exception as e:
                print(f"AI search error for {instrument}: {e}")
                continue

    # Sort by distance
    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float('inf'))

    # Apply max_results limit
    total_count = len(results)
    truncated = total_count > max_results
    results = results[:max_results]

    # Generate message
    if total_count == 0:
        message = f"No datasets found in {intent.region_name or 'the specified area'}."
    elif truncated:
        message = f"Found {total_count} datasets. Showing first {max_results}."
    else:
        message = f"Found {total_count} datasets in {intent.region_name or 'the specified area'}."

    return AISearchResponse(
        parsed_intent=intent,
        total_count=total_count,
        shown_count=len(results),
        truncated=truncated,
        results=results,
        message=message,
    )


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points on Mars using Haversine formula."""
    import math

    MARS_RADIUS_KM = 3389.5

    # Normalize longitudes
    if lon1 > 180:
        lon1 -= 360
    if lon2 > 180:
        lon2 -= 360

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    lon1_r = math.radians(lon1)
    lon2_r = math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(min(a, 1.0)))

    return MARS_RADIUS_KM * c


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/search", response_model=AISearchResponse)
async def ai_search(request: AISearchRequest):
    """
    AI-assisted natural language search.

    Parses the query to identify instruments and regions,
    then executes appropriate searches.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if request.max_results < 1:
        raise HTTPException(status_code=400, detail="max_results must be at least 1")

    if request.max_results > 100:
        request.max_results = 100  # Cap at 100

    # Parse the natural language query
    intent = parse_natural_language(request.query)

    # Execute search
    response = await execute_ai_search(intent, request.max_results)

    return response


@router.get("/regions")
async def list_regions():
    """List all known Mars regions with bounding boxes."""
    from .mars_regions import list_all_regions

    regions = list_all_regions()
    return {
        "count": len(regions),
        "regions": [
            {
                "region_id": r.region_id,
                "name": r.display_name,
                "lat_min": r.lat_min,
                "lat_max": r.lat_max,
                "lon_min": r.lon_min,
                "lon_max": r.lon_max,
                "center_lat": r.center_lat,
                "center_lon": r.center_lon,
                "radius_deg": r.radius_deg,
                "description": r.description,
                "tags": r.tags,
            }
            for r in regions
        ]
    }


@router.post("/parse")
async def parse_query(request: AISearchRequest):
    """
    Parse a natural language query without executing search.
    Useful for debugging/testing the parser.
    """
    intent = parse_natural_language(request.query)
    return intent


# =============================================================================
# Gemini-based AI Search (NEW)
# =============================================================================

class GeminiSearchRequest(BaseModel):
    """Request for Gemini-based AI search."""
    query: str
    max_results: int = 20
    plan: Optional[Dict[str, Any]] = None  # Reuse parsed plan from preview


class GeminiSearchPreview(BaseModel):
    """Preview of parsed intent before execution."""
    success: bool
    instruments: List[str]
    region_type: str
    region_name: Optional[str] = None
    region_lat: Optional[float] = None
    region_lon: Optional[float] = None
    region_bbox: Optional[Dict[str, float]] = None
    radius_km: float = 100.0
    max_results: int = 20
    spatial_predicate: str = "any"
    distribution: str = "any"
    terrain_hint: Optional[str] = None
    cross_instrument: Optional[str] = None  # Reference instrument name
    cross_title_filter: Optional[str] = None  # Title filter on reference
    message: str = ""
    error: Optional[str] = None
    raw_plan: Optional[Dict[str, Any]] = None


class GeminiSearchResult(BaseModel):
    """Single result from Gemini search."""
    product_id: str
    instrument: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    distance_km: Optional[float] = None
    exists: bool = False
    has_core: bool = False
    has_browse: bool = False
    missing_files: List[str] = []


class GeminiSearchResponse(BaseModel):
    """Full response from Gemini search execution."""
    preview: GeminiSearchPreview
    total_count: int
    shown_count: int
    truncated: bool
    results: List[GeminiSearchResult]


@router.get("/gemini/status")
async def gemini_status():
    """Check Gemini API availability."""
    return get_gemini_status()


@router.post("/gemini/preview", response_model=GeminiSearchPreview)
async def gemini_preview(request: GeminiSearchRequest):
    """
    Parse query with Gemini and return preview WITHOUT executing search.
    User must confirm before actual search/download.

    This is the first step in the AI search flow:
    1. User submits query → this endpoint
    2. Show preview to user
    3. User confirms → /gemini/execute
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Cap max_results
    max_results = min(request.max_results, 50)
    if max_results < 1:
        max_results = 1

    logger.info(f"Gemini preview request: {request.query[:100]}...")

    # Parse with Gemini
    plan = await parse_with_gemini(request.query, max_results)

    if plan.error:
        return GeminiSearchPreview(
            success=False,
            instruments=[],
            region_type="global",
            max_results=max_results,
            message="Failed to parse your request.",
            error=plan.error,
            raw_plan=plan.to_dict(),
        )

    # Resolve region bbox from our lookup table if named_region
    _resolve_region_bbox(plan)
    # Auto-detect cross-instrument pattern if Gemini didn't use crossInstrument
    _auto_detect_cross_instrument(plan)

    # Build preview response
    preview = _build_preview(plan)
    return preview


@router.post("/gemini/execute", response_model=GeminiSearchResponse)
async def gemini_execute(request: GeminiSearchRequest):
    """
    Parse query with Gemini AND execute the search.

    This should only be called after user confirms the preview.
    If `plan` is provided (from preview), reuse it to avoid re-parsing.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Cap max_results
    max_results = min(request.max_results, 50)
    if max_results < 1:
        max_results = 1

    logger.info(f"Gemini execute request: {request.query[:100]}...")

    # Reuse plan from preview if provided, otherwise re-parse with Gemini
    if request.plan:
        plan = _reconstruct_plan(request.plan, max_results)
        logger.info("Reusing plan from preview (skipping Gemini re-parse)")
    else:
        plan = await parse_with_gemini(request.query, max_results)

    # Resolve region bbox from our lookup table if named_region
    _resolve_region_bbox(plan)
    # Auto-detect cross-instrument pattern if Gemini didn't use crossInstrument
    _auto_detect_cross_instrument(plan)

    # Build preview
    preview = _build_preview(plan)

    if plan.error:
        return GeminiSearchResponse(
            preview=preview,
            total_count=0,
            shown_count=0,
            truncated=False,
            results=[],
        )

    # Execute search based on plan
    results = await _execute_gemini_plan(plan)

    # Sort by distance
    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float('inf'))

    # Apply limit
    total_count = len(results)
    truncated = total_count > plan.filters.maxResults
    results = results[:plan.filters.maxResults]

    return GeminiSearchResponse(
        preview=preview,
        total_count=total_count,
        shown_count=len(results),
        truncated=truncated,
        results=results,
    )


def _reconstruct_plan(raw_plan: Dict[str, Any], max_results: int) -> GeminiPlan:
    """Reconstruct a GeminiPlan from the raw_plan dict saved during preview."""
    from .gemini_parser import GeminiRegion, GeminiFilters, GeminiCrossFilter

    region_data = raw_plan.get("region", {})
    region = GeminiRegion(
        type=region_data.get("type", "global"),
        name=region_data.get("name"),
        bbox=region_data.get("bbox"),
        point=region_data.get("point"),
        radiusKm=region_data.get("radiusKm", 100.0),
    )

    filters_data = raw_plan.get("filters", {})
    cross_data = filters_data.get("crossInstrument")
    cross_filter = None
    if cross_data and isinstance(cross_data, dict) and cross_data.get("instrument"):
        cross_filter = GeminiCrossFilter(
            instrument=cross_data["instrument"],
            titleContains=cross_data.get("titleContains"),
        )

    filters = GeminiFilters(
        maxResults=min(filters_data.get("maxResults", max_results), max_results),
        spatialPredicate=filters_data.get("spatialPredicate", "any"),
        distribution=filters_data.get("distribution", "any"),
        terrainHint=filters_data.get("terrainHint"),
        crossInstrument=cross_filter,
    )

    return GeminiPlan(
        intent=raw_plan.get("intent", "download"),
        targets=raw_plan.get("targets", []),
        region=region,
        filters=filters,
    )


def _auto_detect_cross_instrument(plan: GeminiPlan) -> None:
    """
    Auto-detect cross-instrument queries when Gemini doesn't use the
    crossInstrument field explicitly.

    Pattern: 2 instruments in targets, spatialPredicate is "intersects",
    no crossInstrument set.
    → Convert to cross-instrument: first target is primary, second is reference.
    → Move terrainHint to titleContains if present.
    """
    if plan.filters.crossInstrument:
        return  # already set
    if plan.filters.spatialPredicate not in ("intersects", "crosses"):
        return
    if len(plan.targets) != 2:
        return

    primary, reference = plan.targets[0], plan.targets[1]
    title_filter = plan.filters.terrainHint

    logger.info(f"Auto-detected cross-instrument: primary={primary}, reference={reference}, title={title_filter}")

    plan.targets = [primary]
    plan.filters.crossInstrument = GeminiCrossFilter(
        instrument=reference,
        titleContains=title_filter,
    )
    plan.filters.terrainHint = None


def _resolve_region_bbox(plan: GeminiPlan) -> None:
    """
    If plan has a named_region, resolve its bbox from the lookup table.
    This ensures all region-based queries use authoritative bounding boxes.
    """
    region = plan.region
    if region.type == "named_region" and region.name:
        lookup = find_region(region.name)
        if lookup:
            # Override with lookup table bbox
            region.bbox = {
                "minLat": lookup.lat_min,
                "maxLat": lookup.lat_max,
                "minLon": lookup.lon_min,
                "maxLon": lookup.lon_max,
            }
            # Also set point to center if not already set
            if not region.point:
                region.point = {
                    "lat": lookup.center_lat,
                    "lon": lookup.center_lon,
                }
            # Update region name to canonical name
            region.name = lookup.display_name


def _build_preview(plan: GeminiPlan) -> GeminiSearchPreview:
    """Build a GeminiSearchPreview from a validated plan."""
    region = plan.region
    cross = plan.filters.crossInstrument
    return GeminiSearchPreview(
        success=not plan.error,
        instruments=plan.targets,
        region_type=region.type,
        region_name=region.name,
        region_lat=region.point.get("lat") if region.point else None,
        region_lon=region.point.get("lon") if region.point else None,
        region_bbox=region.bbox,
        radius_km=region.radiusKm,
        max_results=plan.filters.maxResults,
        spatial_predicate=plan.filters.spatialPredicate,
        distribution=plan.filters.distribution,
        terrain_hint=plan.filters.terrainHint,
        cross_instrument=cross.instrument if cross else None,
        cross_title_filter=cross.titleContains if cross else None,
        message=_build_preview_message(plan) if not plan.error else "Failed to parse request.",
        error=plan.error,
        raw_plan=plan.to_dict(),
    )


def _build_preview_message(plan: GeminiPlan) -> str:
    """Build a human-readable preview message."""
    parts = []

    # Instruments
    instruments = ", ".join(plan.targets)
    parts.append(f"Instruments: {instruments}")

    # Cross-instrument filter
    cross = plan.filters.crossInstrument
    if cross:
        label = f"Intersect with: {cross.instrument}"
        if cross.titleContains:
            label += f' (title: "{cross.titleContains}")'
        parts.append(label)

    # Region
    region = plan.region
    if region.type == "named_region" and region.name:
        parts.append(f"Region: {region.name}")
        if region.point:
            lat, lon = region.point.get("lat"), region.point.get("lon")
            if lat is not None and lon is not None:
                parts.append(f"Center: {lat:.2f}°, {lon:.2f}°")
    elif region.type == "point" and region.point:
        lat, lon = region.point.get("lat"), region.point.get("lon")
        parts.append(f"Point: {lat:.2f}°, {lon:.2f}°")
    elif region.type == "bbox" and region.bbox:
        bbox = region.bbox
        parts.append(f"Bounding box: {bbox.get('minLat'):.1f}° to {bbox.get('maxLat'):.1f}° lat, "
                     f"{bbox.get('minLon'):.1f}° to {bbox.get('maxLon'):.1f}° lon")
    elif region.type == "global" and not cross:
        # Only show "Global" when there's no cross-instrument (which provides its own spatial scope)
        parts.append("Region: Global (no specific area)")

    # Radius
    if region.radiusKm and region.type in ("named_region", "point"):
        parts.append(f"Search radius: {region.radiusKm:.0f} km")

    # Spatial predicate (skip for cross-instrument since it's already implied)
    if plan.filters.spatialPredicate and plan.filters.spatialPredicate != "any" and not cross:
        parts.append(f"Spatial: {plan.filters.spatialPredicate}")

    # Distribution
    if plan.filters.distribution and plan.filters.distribution != "any":
        parts.append(f"Distribution: {plan.filters.distribution}")

    # Terrain hint
    if plan.filters.terrainHint:
        parts.append(f"Terrain: {plan.filters.terrainHint}")

    # Max results
    parts.append(f"Max results: {plan.filters.maxResults}")

    return " • ".join(parts)


def _search_local_index(instrument_id: str, minlat: float, maxlat: float,
                        lon_min: float, lon_max: float,
                        center_lat: float, center_lon: float,
                        max_results: int) -> List[GeminiSearchResult]:
    """Search local GeoJSON index for instruments not queryable via ODE (CTX, HIRISE_DTM)."""
    from .registry import get_registry
    from .proximity_router import _bbox_from_geometry, _centroid_from_geometry, haversine_km
    from .search_router import _geometry_intersects_bbox

    registry = get_registry()
    try:
        index = registry.load_index(instrument_id)
    except Exception:
        return []

    if not index:
        return []

    results = []
    for feature in index.get("features", []):
        props = feature.get("properties", {})
        pid = (props.get("product_id") or props.get("ProductId") or
               props.get("id") or props.get("PRODUCT_ID") or "")
        if not pid:
            continue

        geom = feature.get("geometry")
        if not geom:
            continue

        # Use proper geometry intersection (same as overlay filter)
        if not _geometry_intersects_bbox(geom, minlat, maxlat, lon_min, lon_max):
            continue

        centroid = _centroid_from_geometry(geom)
        if not centroid:
            continue
        clat, clon = centroid["lat"], centroid["lon"]

        dist = haversine_km(center_lat, center_lon, clat, clon)

        results.append(GeminiSearchResult(
            product_id=pid,
            instrument=instrument_id.upper(),
            lat=clat,
            lon=clon,
            distance_km=round(dist, 2),
            exists=True,  # If in local index, it's available locally
            has_core=True,
            has_browse=True,
            missing_files=[],
        ))

    results.sort(key=lambda r: r.distance_km if r.distance_km is not None else float("inf"))
    return results[:max_results]


async def _execute_gemini_plan(plan: GeminiPlan) -> List[GeminiSearchResult]:
    """
    Execute search based on Gemini plan.

    Uses region bbox from lookup table when available (more accurate than radius).
    Supports cross-instrument spatial joins via plan.filters.crossInstrument.

    Args:
        plan: Validated Gemini plan

    Returns:
        List of search results
    """
    # Cross-instrument spatial join: delegate to specialised handler
    if plan.filters.crossInstrument:
        return await _execute_cross_instrument(plan)

    results: List[GeminiSearchResult] = []
    region = plan.region

    # Determine search bbox - prefer explicit bbox over point+radius
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    minlat: Optional[float] = None
    maxlat: Optional[float] = None
    westernlon: Optional[float] = None
    easternlon: Optional[float] = None

    if region.bbox:
        # Use bbox directly (from region lookup table or explicit)
        minlat = region.bbox.get("minLat", -90)
        maxlat = region.bbox.get("maxLat", 90)
        lon_min_180 = region.bbox.get("minLon", -180)
        lon_max_180 = region.bbox.get("maxLon", 180)
        center_lat = (minlat + maxlat) / 2
        if lon_min_180 <= lon_max_180:
            center_lon = (lon_min_180 + lon_max_180) / 2
        else:
            # Wraps around antimeridian
            c = (lon_min_180 + lon_max_180 + 360) / 2
            center_lon = c if c <= 180 else c - 360
        # Convert to ODE's 0-360
        westernlon = lon_min_180
        easternlon = lon_max_180
        if westernlon < 0:
            westernlon += 360
        if easternlon < 0:
            easternlon += 360
    elif region.point:
        center_lat = region.point.get("lat")
        center_lon = region.point.get("lon")
        radius_deg = region.radiusKm / 59.0 if region.radiusKm else 2.0
        if center_lat is not None:
            minlat = max(-90, center_lat - radius_deg)
            maxlat = min(90, center_lat + radius_deg)
            westernlon = center_lon - radius_deg
            easternlon = center_lon + radius_deg
            if westernlon < 0:
                westernlon += 360
            if easternlon < 0:
                easternlon += 360

    if center_lat is None or minlat is None:
        # No region specified — try to derive bbox from local instrument data.
        local_instruments = [t for t in plan.targets
                             if t.upper() in ("CTX", "HIRISE_DTM", "SHARAD_HIGHRES")]
        if local_instruments:
            from .registry import get_registry
            from .proximity_router import _bbox_from_geometry
            registry = get_registry()
            all_west, all_south, all_east, all_north = 180.0, 90.0, -180.0, -90.0
            found_any = False
            for inst in local_instruments:
                try:
                    index = registry.load_index(inst.upper())
                except Exception:
                    continue
                for feat in index.get("features", []):
                    geom = feat.get("geometry")
                    if not geom:
                        continue
                    bbox = _bbox_from_geometry(geom)
                    if not bbox:
                        continue
                    found_any = True
                    all_west = min(all_west, bbox["lon_min"])
                    all_east = max(all_east, bbox["lon_max"])
                    all_south = min(all_south, bbox["lat_min"])
                    all_north = max(all_north, bbox["lat_max"])
            if found_any:
                margin = 1.0
                minlat = max(-90, all_south - margin)
                maxlat = min(90, all_north + margin)
                center_lat = (minlat + maxlat) / 2
                center_lon = (all_west + all_east) / 2
                westernlon = all_west - margin
                easternlon = all_east + margin
                if westernlon < 0:
                    westernlon += 360
                if easternlon < 0:
                    easternlon += 360
                logger.info(f"Derived bbox from local data: [{minlat:.1f}, {maxlat:.1f}] lon [{all_west - margin:.1f}, {all_east + margin:.1f}]")
            else:
                logger.warning("No region specified and no local data found, returning empty results")
                return []
        else:
            logger.warning("No region specified in plan, returning empty results")
            return []

    # For local index search, keep lon in -180/180
    if region.bbox:
        lon_min_180 = region.bbox.get("minLon", center_lon - 5)
        lon_max_180 = region.bbox.get("maxLon", center_lon + 5)
    else:
        lon_min_180 = westernlon - 360 if westernlon > 180 else westernlon
        lon_max_180 = easternlon - 360 if easternlon > 180 else easternlon

    logger.info(f"Executing search: bbox=[{minlat:.1f}, {maxlat:.1f}, {westernlon:.1f}, {easternlon:.1f}]")

    # Search each requested instrument — local-first, ODE as fallback, all in parallel
    max_per_inst = plan.filters.maxResults * 2

    # Map of instrument key to ODE-queryable instruments
    _ODE_INSTRUMENTS = {"crism", "hirise", "sharad", "sharad_highres"}

    async def _search_one_instrument(target: str) -> List[GeminiSearchResult]:
        """Search one instrument: local index first, ODE fallback if needed."""
        instrument = target.lower()
        inst_results: List[GeminiSearchResult] = []

        # Step 1: Always try local index first (instant, no network)
        local_results = _search_local_index(
            instrument.upper(), minlat, maxlat,
            lon_min_180, lon_max_180,
            center_lat, center_lon,
            max_per_inst,
        )
        inst_results.extend(local_results)

        # Step 2: If local results exist, skip slow ODE call
        if inst_results or instrument not in _ODE_INSTRUMENTS:
            return inst_results

        # Step 3: ODE fallback for instruments with remote data
        try:
            products = []
            async with aiohttp.ClientSession() as session:
                if instrument == "crism":
                    products = await search_ode_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        Instrument.CRISM,
                        max_results=max_per_inst,
                        session=session
                    )
                elif instrument == "hirise":
                    products = await search_ode_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        Instrument.HIRISE,
                        max_results=max_per_inst,
                        session=session
                    )
                elif instrument == "sharad":
                    products = await search_sharad_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        max_results=max_per_inst,
                        session=session
                    )
                elif instrument == "sharad_highres":
                    products = await search_sharad_highres_spatial(
                        minlat, maxlat, westernlon, easternlon,
                        max_results=max_per_inst,
                        session=session
                    )

            # Deduplicate: skip products already found in local index
            local_pids = {r.product_id.upper() for r in inst_results}

            for p in products:
                if p.product_id.upper() in local_pids:
                    continue
                inst_enum = Instrument(instrument)
                existence = check_local_existence_detailed(p.product_id, inst_enum)

                distance_km = None
                if p.lat is not None and p.lon is not None and center_lat is not None:
                    distance_km = haversine_distance_km(center_lat, center_lon, p.lat, p.lon)

                inst_results.append(GeminiSearchResult(
                    product_id=p.product_id,
                    instrument=instrument.upper(),
                    lat=p.lat,
                    lon=p.lon,
                    distance_km=round(distance_km, 2) if distance_km else None,
                    exists=existence.exists,
                    has_core=existence.has_core,
                    has_browse=existence.has_browse,
                    missing_files=existence.missing_files,
                ))
        except Exception as e:
            logger.error(f"ODE search error for {instrument}: {e}")

        return inst_results

    # Run all instrument searches in parallel
    tasks = [_search_one_instrument(target) for target in plan.targets]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in all_results:
        if isinstance(r, Exception):
            logger.error(f"Instrument search failed: {r}")
            continue
        results.extend(r)

    return results


async def _execute_cross_instrument(plan: GeminiPlan) -> List[GeminiSearchResult]:
    """
    Cross-instrument spatial join.

    1. Load the reference instrument's local index.
    2. Optionally filter reference features by title substring.
    3. Compute the combined bbox of matching reference features.
    4. Search each target instrument within that bbox.
    5. Post-filter: only keep target products that actually spatially
       intersect at least one matching reference feature.
    """
    from .registry import get_registry
    from .proximity_router import (
        _bbox_from_geometry,
        _centroid_from_geometry,
        _geometries_overlap,
        haversine_km,
    )

    cross = plan.filters.crossInstrument
    ref_instrument = cross.instrument.upper()
    title_filter = (cross.titleContains or "").lower().strip()
    # Keyword-based matching: split into significant words (len > 2)
    # Handles cases like "related to terraced crater" → ["terraced", "crater"]
    _STOP_WORDS = {"the", "and", "for", "with", "related", "about", "that", "which", "from", "into"}
    title_keywords = [w for w in title_filter.split() if len(w) > 2 and w not in _STOP_WORDS] if title_filter else []

    registry = get_registry()

    # 1. Load reference index & filter
    try:
        ref_index = registry.load_index(ref_instrument)
    except Exception:
        logger.warning(f"Cross-instrument: cannot load index for {ref_instrument}")
        return []

    ref_features = []  # list of (geometry, bbox_dict)
    for feat in ref_index.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry")
        if not geom:
            continue

        # Title filter: all keywords must appear in the title
        if title_keywords:
            title = (props.get("title") or props.get("Title") or "").lower()
            if not all(kw in title for kw in title_keywords):
                continue

        bbox = _bbox_from_geometry(geom)
        if not bbox:
            continue
        ref_features.append((geom, bbox))

    if not ref_features:
        logger.info(f"Cross-instrument: no reference features matched (ref={ref_instrument}, title_filter={title_filter!r})")
        return []

    logger.info(f"Cross-instrument: {len(ref_features)} reference features from {ref_instrument}")

    # 2. Combined bbox of reference features
    all_west = min(b["lon_min"] for _, b in ref_features)
    all_east = max(b["lon_max"] for _, b in ref_features)
    all_south = min(b["lat_min"] for _, b in ref_features)
    all_north = max(b["lat_max"] for _, b in ref_features)
    margin = 1.0
    minlat = max(-90, all_south - margin)
    maxlat = min(90, all_north + margin)
    lon_min_180 = all_west - margin
    lon_max_180 = all_east + margin
    center_lat = (minlat + maxlat) / 2
    center_lon = (lon_min_180 + lon_max_180) / 2
    westernlon = lon_min_180 + 360 if lon_min_180 < 0 else lon_min_180
    easternlon = lon_max_180 + 360 if lon_max_180 < 0 else lon_max_180

    logger.info(f"Cross-instrument combined bbox: lat [{minlat:.1f}, {maxlat:.1f}] lon [{lon_min_180:.1f}, {lon_max_180:.1f}]")

    # 3. Search target instruments within combined bbox
    candidate_results: List[GeminiSearchResult] = []
    candidate_geoms: List[Any] = []  # parallel list — geometry per candidate

    async with aiohttp.ClientSession() as session:
        for target in plan.targets:
            instrument = target.lower()
            try:
                if instrument in ("ctx", "hirise_dtm", "sharad_highres"):
                    # Local index: collect features with geometry for post-filter
                    try:
                        tgt_index = registry.load_index(instrument.upper())
                    except Exception:
                        continue
                    for feat in tgt_index.get("features", []):
                        props = feat.get("properties", {})
                        pid = (props.get("product_id") or props.get("ProductId")
                               or props.get("id") or props.get("PRODUCT_ID") or "")
                        if not pid:
                            continue
                        geom = feat.get("geometry")
                        if not geom:
                            continue
                        # Use proper geometry intersection (same as overlay filter)
                        if not _geometry_intersects_bbox(geom, minlat, maxlat,
                                                         lon_min_180, lon_max_180):
                            continue

                        centroid = _centroid_from_geometry(geom)
                        if not centroid:
                            continue
                        clat, clon = centroid["lat"], centroid["lon"]

                        dist = haversine_km(center_lat, center_lon, clat, clon)
                        candidate_results.append(GeminiSearchResult(
                            product_id=pid,
                            instrument=instrument.upper(),
                            lat=clat,
                            lon=clon,
                            distance_km=round(dist, 2),
                            exists=True,
                            has_core=True,
                            has_browse=True,
                            missing_files=[],
                        ))
                        candidate_geoms.append(geom)
                else:
                    # ODE instruments
                    products = []
                    if instrument == "crism":
                        products = await search_ode_spatial(
                            minlat, maxlat, westernlon, easternlon,
                            Instrument.CRISM,
                            max_results=plan.filters.maxResults * 4,
                            session=session,
                        )
                    elif instrument == "hirise":
                        products = await search_ode_spatial(
                            minlat, maxlat, westernlon, easternlon,
                            Instrument.HIRISE,
                            max_results=plan.filters.maxResults * 4,
                            session=session,
                        )
                    elif instrument == "sharad":
                        products = await search_sharad_spatial(
                            minlat, maxlat, westernlon, easternlon,
                            max_results=plan.filters.maxResults * 4,
                            session=session,
                        )

                    for p in products:
                        inst_enum = Instrument(instrument)
                        existence = check_local_existence_detailed(p.product_id, inst_enum)
                        distance_km = None
                        if p.lat is not None and p.lon is not None:
                            distance_km = haversine_distance_km(center_lat, center_lon, p.lat, p.lon)
                        # Build a simple Point geometry for spatial intersection test
                        point_geom = None
                        if p.lat is not None and p.lon is not None:
                            point_geom = {"type": "Point", "coordinates": [p.lon, p.lat]}
                        candidate_results.append(GeminiSearchResult(
                            product_id=p.product_id,
                            instrument=instrument.upper(),
                            lat=p.lat,
                            lon=p.lon,
                            distance_km=round(distance_km, 2) if distance_km else None,
                            exists=existence.exists,
                            has_core=existence.has_core,
                            has_browse=existence.has_browse,
                            missing_files=existence.missing_files,
                        ))
                        candidate_geoms.append(point_geom)

            except Exception as e:
                logger.error(f"Cross-instrument search error for {instrument}: {e}")
                continue

    # 4. Post-filter: keep only candidates that intersect a reference feature
    results: List[GeminiSearchResult] = []
    for candidate, cand_geom in zip(candidate_results, candidate_geoms):
        if cand_geom is None:
            continue
        cand_bbox = _bbox_from_geometry(cand_geom)
        if not cand_bbox:
            continue
        for ref_geom, ref_bbox in ref_features:
            if _geometries_overlap(cand_geom, cand_bbox, ref_geom, ref_bbox):
                results.append(candidate)
                break

    logger.info(f"Cross-instrument: {len(candidate_results)} candidates → {len(results)} after spatial intersection")
    return results
