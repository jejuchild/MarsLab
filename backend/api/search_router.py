"""
Search & Download API Router

Endpoints:
- GET /api/search - Search ODE by product ID (typeahead)
- GET /api/search/spatial - Search ODE by coordinates
- GET /api/exists/{instrument}/{product_id} - Check local existence
- POST /api/download - Start a download task
- GET /api/download/{task_id} - Get download status
- GET /api/download - List all download tasks
"""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import os

from .ode_client import (
    Instrument,
    ODEProduct,
    search_ode_products,
    search_ode_spatial,
    search_sharad_products,
    search_sharad_highres_products,
    parse_crism_base_key,
    is_crism_product_id,
    is_hirise_product_id,
    is_sharad_product_id,
    is_sharad_highres_product_id,
    is_hirise_dtm_product_id,
    _parse_ode_rest_response,
    ODE_REST_BASE,
)
import json
from pathlib import Path
from .download_manager import (
    download_manager,
    check_local_existence,
    check_local_existence_detailed,
    DownloadStatus,
)
from .validation import validate_product_id
from .rate_limit import limiter


import aiohttp
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Search & Download"])


def _ode_products_to_trr3_results(ode_products: list, limit: int) -> List["SearchResult"]:
    """Convert ODE products to TRR3 SearchResults, deduplicated by base_key."""
    results = []
    seen = set()
    for p in ode_products:
        pid = p.product_id.lower()
        # Filter to FRT (Full Resolution Targeted) only
        if not pid.startswith("frt"):
            continue
        # Base key = first two underscore-separated tokens (e.g. frt0000951f_07)
        parts = pid.split("_")
        base_key = "_".join(parts[:2]) if len(parts) >= 2 else pid
        if base_key in seen:
            continue
        seen.add(base_key)
        lon = p.lon
        if lon is not None and lon > 180:
            lon -= 360
        results.append(SearchResult(
            product_id=base_key,
            instrument="crism_trr3",
            base_key=base_key,
            lat=p.lat, lon=lon,
            exists=False, has_core=False, has_browse=False,
            missing_files=[],
        ))
        if len(results) >= limit:
            break
    return results


async def _search_ode_trr3_by_id(query: str, limit: int = 10) -> List["SearchResult"]:
    """Search ODE for CRISM TRDR (TRR3) products by product ID."""
    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&ihid=mro&iid=crism&pt=TRDR&"
        f"productid={query}*&output=json&results=pmf&limit={max(limit * 10, 100)}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    ode_prods = _parse_ode_rest_response(data, Instrument.CRISM)
                    return _ode_products_to_trr3_results(ode_prods, limit)
    except Exception as e:
        logger.warning(f"ODE TRR3 ID search failed: {e}")
    return []


async def _search_ode_trr3_spatial(
    minlat: float, maxlat: float, westernlon: float, easternlon: float, limit: int = 10
) -> List["SearchResult"]:
    """Search ODE for CRISM TRDR (TRR3) products within a bounding box."""
    # ODE works best with 0-360 longitude
    wlon = westernlon if westernlon >= 0 else westernlon + 360
    elon = easternlon if easternlon >= 0 else easternlon + 360
    url = (
        f"{ODE_REST_BASE}?"
        f"target=mars&ihid=mro&iid=crism&pt=TRDR&"
        f"minlat={minlat}&maxlat={maxlat}&"
        f"westernlon={wlon}&easternlon={elon}&"
        f"output=json&results=pmf&limit={max(limit * 20, 400)}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    ode_prods = _parse_ode_rest_response(data, Instrument.CRISM)
                    logger.info(f"ODE TRR3 spatial: {len(ode_prods)} raw, filtering to targeted")
                    return _ode_products_to_trr3_results(ode_prods, limit)
                else:
                    logger.warning(f"ODE TRR3 spatial returned {resp.status}")
    except Exception as e:
        logger.warning(f"ODE TRR3 spatial search failed: {e}")
    return []

# SHARAD index path
SHARAD_INDEX_PATH = Path(__file__).parent.parent / "sharad_data" / "index.geojson"

# All local index paths for full-database search
_BACKEND_DIR = Path(__file__).parent.parent
_LOCAL_INDICES = {
    "CRISM":          _BACKEND_DIR / "crism_data"          / "index.geojson",
    "HIRISE":         _BACKEND_DIR / "hirise_data"         / "index.geojson",
    "SHARAD":         _BACKEND_DIR / "sharad_data"         / "index.geojson",
    "SHARAD_HIGHRES": _BACKEND_DIR / "sharad_highres_data" / "index.geojson",
    "CTX":            _BACKEND_DIR / "ctx_data"            / "index.geojson",
    "HIRISE_DTM":     _BACKEND_DIR / "hirise_dtm_data"     / "index.geojson",
    "CRISM_TRR3":     _BACKEND_DIR / "mineral_cnn_data"    / "index.geojson",
}

# Cached product list (loaded once on first search)
_all_products_cache: list | None = None


def invalidate_products_cache():
    """Reset the all-products cache so the next search re-reads index files."""
    global _all_products_cache
    _all_products_cache = None


def _load_all_products() -> list:
    """Load all products from all index.geojson files. Cached after first call."""
    global _all_products_cache
    if _all_products_cache is not None:
        return _all_products_cache

    products = []
    for instrument, path in _LOCAL_INDICES.items():
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                pid = props.get("product_id", "")
                if not pid:
                    continue
                # Extract centroid lat/lon from geometry or properties
                lat = props.get("center_latitude") or props.get("start_lat")
                lon = props.get("center_longitude") or props.get("start_lon")
                # Try geometry centroid as fallback
                if lat is None or lon is None:
                    geom = feature.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    if geom.get("type") == "Polygon" and coords:
                        ring = coords[0]
                        lat = sum(c[1] for c in ring) / len(ring)
                        lon = sum(c[0] for c in ring) / len(ring)
                    elif geom.get("type") == "LineString" and coords:
                        lat = coords[0][1]
                        lon = coords[0][0]
                products.append({
                    "product_id": pid,
                    "instrument": instrument,
                    "title": props.get("title", ""),
                    "lat": lat,
                    "lon": lon,
                })
        except Exception:
            continue

    _all_products_cache = products
    return products


# =============================================================================
# Response Models
# =============================================================================

class SearchResult(BaseModel):
    """Single search result."""
    product_id: str
    instrument: str  # "crism", "hirise", or "sharad"
    base_key: str  # For CRISM, the observation-level key
    lat: Optional[float] = None
    lon: Optional[float] = None
    exists: bool  # True if ALL required files are downloaded
    has_core: bool = False  # True if core files (.img, .lbl) exist
    has_browse: bool = False  # True if at least one browse file exists
    missing_files: List[str] = []  # List of missing file types


def search_local_sharad(query: str, limit: int = 10) -> List[SearchResult]:
    """
    Search local SHARAD index.geojson for matching products.
    """
    if not SHARAD_INDEX_PATH.exists():
        return []

    try:
        with open(SHARAD_INDEX_PATH) as f:
            data = json.load(f)
    except Exception:
        return []

    results = []
    query_upper = query.upper()

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        product_id = props.get("product_id", "")

        # Match query against product_id
        if query_upper in product_id.upper():
            results.append(SearchResult(
                product_id=product_id,
                instrument="sharad",
                base_key=product_id,
                lat=props.get("start_lat"),
                lon=props.get("start_lon"),
                exists=True,  # Local data always exists
                has_core=True,
                has_browse=True,
                missing_files=[],
            ))

        if len(results) >= limit:
            break

    # Sort: exact prefix matches first
    results.sort(key=lambda r: (
        0 if r.product_id.upper().startswith(query_upper) else 1,
        r.product_id
    ))

    return results[:limit]


def search_local_index(instrument_key: str, instrument_value: str, query: str, limit: int = 10) -> List[SearchResult]:
    """Search a local GeoJSON index for matching products by product ID substring."""
    index_path = _LOCAL_INDICES.get(instrument_key)
    if not index_path or not index_path.exists():
        return []

    try:
        with open(index_path) as f:
            data = json.load(f)
    except Exception:
        return []

    results = []
    query_upper = query.upper()

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        product_id = (props.get("product_id") or props.get("ProductId") or
                      props.get("id") or props.get("PRODUCT_ID") or "")
        if not product_id:
            continue

        if query_upper in product_id.upper():
            # Extract lat/lon from properties or geometry centroid
            lat = props.get("center_latitude") or props.get("start_lat")
            lon = props.get("center_longitude") or props.get("start_lon")
            if lat is None or lon is None:
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates", [])
                if geom.get("type") == "Polygon" and coords:
                    ring = coords[0]
                    lat = sum(c[1] for c in ring) / len(ring)
                    lon = sum(c[0] for c in ring) / len(ring)

            results.append(SearchResult(
                product_id=product_id,
                instrument=instrument_value,
                base_key=product_id,
                lat=lat,
                lon=lon,
                exists=True,
                has_core=True,
                has_browse=True,
                missing_files=[],
            ))

        if len(results) >= limit:
            break

    results.sort(key=lambda r: (
        0 if r.product_id.upper().startswith(query_upper) else 1,
        r.product_id
    ))

    return results[:limit]


def _liang_barsky_clip(
    x1: float, y1: float, x2: float, y2: float,
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> bool:
    """Test if line segment (x1,y1)-(x2,y2) intersects axis-aligned box.

    Same algorithm as frontend overlapFilter.ts `liangBarskyClip`.
    """
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0

    for p, q in [(-dx, x1 - xmin), (dx, xmax - x1),
                 (-dy, y1 - ymin), (dy, ymax - y1)]:
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return False
    return True


def _geometry_intersects_bbox(
    geom: dict,
    minlat: float, maxlat: float,
    westernlon: float, easternlon: float,
) -> bool:
    """Test if a GeoJSON geometry intersects a search bounding box.

    Uses the same approach as the frontend overlay filter:
    - Polygon: AABB intersection (bbox of polygon vs search bbox)
    - LineString: Liang-Barsky segment clipping against search bbox
    - Point: point-in-bbox test with small buffer

    Handles antimeridian crossing (westernlon > easternlon).
    """
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if not coords:
        return False

    # Build feature AABB
    if gtype == "Polygon" and coords:
        ring = coords[0]
        if not ring:
            return False
        feat_west = min(c[0] for c in ring)
        feat_east = max(c[0] for c in ring)
        feat_south = min(c[1] for c in ring)
        feat_north = max(c[1] for c in ring)

        # AABB vs search bbox intersection
        if feat_north < minlat or feat_south > maxlat:
            return False
        if westernlon <= easternlon:
            return not (feat_east < westernlon or feat_west > easternlon)
        else:
            # Search bbox crosses antimeridian
            return feat_east >= westernlon or feat_west <= easternlon

    elif gtype == "LineString" and coords:
        # Liang-Barsky: test each segment against search bbox
        for i in range(len(coords) - 1):
            x1, y1 = coords[i][0], coords[i][1]
            x2, y2 = coords[i + 1][0], coords[i + 1][1]
            # Skip antimeridian-crossing segments (orbital dateline transition)
            if abs(x2 - x1) > 180:
                continue
            if westernlon <= easternlon:
                if _liang_barsky_clip(x1, y1, x2, y2,
                                      westernlon, minlat, easternlon, maxlat):
                    return True
            else:
                # Antimeridian: test both sides
                if _liang_barsky_clip(x1, y1, x2, y2,
                                      westernlon, minlat, 180.0, maxlat):
                    return True
                if _liang_barsky_clip(x1, y1, x2, y2,
                                      -180.0, minlat, easternlon, maxlat):
                    return True
        return False

    elif gtype == "Point" and coords:
        lon, lat = coords[0], coords[1]
        if lat < minlat or lat > maxlat:
            return False
        if westernlon <= easternlon:
            return westernlon <= lon <= easternlon
        else:
            return lon >= westernlon or lon <= easternlon

    return False


def _geometry_centroid(geom: dict) -> tuple:
    """Extract centroid (lat, lon) from a GeoJSON geometry."""
    gtype = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if gtype == "Polygon" and coords:
        ring = coords[0]
        return (sum(c[1] for c in ring) / len(ring),
                sum(c[0] for c in ring) / len(ring))
    elif gtype == "LineString" and coords:
        mid = len(coords) // 2
        return (coords[mid][1], coords[mid][0])
    elif gtype == "Point" and coords:
        return (coords[1], coords[0])
    return (None, None)


def search_local_spatial(instrument_key: str, instrument_value: str,
                         minlat: float, maxlat: float,
                         westernlon: float, easternlon: float,
                         limit: int = 10) -> List[SearchResult]:
    """Search a local GeoJSON index for products intersecting a bounding box.

    Uses the same spatial logic as the frontend overlay filter:
    - Polygon: AABB intersection
    - LineString: Liang-Barsky segment-vs-bbox clipping
    - Point: point-in-bbox test
    """
    index_path = _LOCAL_INDICES.get(instrument_key)
    if not index_path or not index_path.exists():
        return []

    try:
        with open(index_path) as f:
            data = json.load(f)
    except Exception:
        return []

    results = []

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        product_id = (props.get("product_id") or props.get("ProductId") or
                      props.get("id") or props.get("PRODUCT_ID") or "")
        if not product_id:
            continue

        geom = feature.get("geometry", {})
        if not geom:
            continue

        # Use proper geometry intersection (same as overlay filter)
        if not _geometry_intersects_bbox(geom, minlat, maxlat,
                                         westernlon, easternlon):
            continue

        lat, lon = _geometry_centroid(geom)
        if lat is None:
            continue

        results.append(SearchResult(
            product_id=product_id,
            instrument=instrument_value,
            base_key=product_id,
            lat=lat,
            lon=lon,
            exists=True,
            has_core=True,
            has_browse=True,
            missing_files=[],
        ))

        if len(results) >= limit:
            break

    return results[:limit]


class SearchResponse(BaseModel):
    """Search response with multiple results."""
    query: str
    results: List[SearchResult]
    count: int


class DownloadRequest(BaseModel):
    """Request to start a download."""
    product_id: str
    instrument: str  # "crism" or "hirise"
    lat: Optional[float] = None
    lon: Optional[float] = None
    file_types: Optional[List[str]] = None  # Optional: only download specific types ("core", "browse", etc.)


class FileStatus(BaseModel):
    """Status of a single file in a download."""
    filename: str
    status: str
    bytes_downloaded: int
    bytes_total: Optional[int]
    progress_percent: float
    error: Optional[str] = None


class DownloadResponse(BaseModel):
    """Download task status."""
    task_id: str
    product_id: str
    base_key: str
    instrument: str
    status: str
    files: List[FileStatus]
    target_dir: str
    progress_percent: float
    total_bytes: int
    downloaded_bytes: int
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Search Endpoints
# =============================================================================

@router.get("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_products(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query (product ID or partial)"),
    instrument: Optional[str] = Query(None, description="Filter by instrument: crism or hirise"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    """
    Search ODE for products by product ID.

    This endpoint queries the ODE REST API for products matching the search
    query. Results include an existence check to indicate if the product is
    already downloaded locally.

    ## Query Behavior

    - **Typeahead search**: Matches partial product IDs
    - **Case-insensitive**: Searches are normalized to lowercase
    - **Sorting priority**:
      1. Exact prefix matches (starts with query)
      2. Substring matches (contains query)

    ## ODE REST Query

    The underlying ODE query uses:
    ```
    https://oderest.rsl.wustl.edu/live2/?target=mars&ihid=mro&iid={instrument}&productid=*{query}*&output=json
    ```

    This wildcards the query to find partial matches.

    ## Example

    ```
    GET /api/search?q=frt00009&limit=10
    ```

    Returns CRISM products with IDs containing "frt00009".
    """
    # Handle CRISM_TRR3 search - local index + ODE TRDR
    if instrument and instrument.lower() == "crism_trr3":
        local_results = search_local_index("CRISM_TRR3", "crism_trr3", q, limit)
        ode_results = await _search_ode_trr3_by_id(q, limit)
        # Merge: local results first (exist=True), then ODE results not in local
        local_keys = {r.base_key for r in local_results}
        merged = list(local_results)
        for r in ode_results:
            if r.base_key not in local_keys:
                merged.append(r)
        merged = merged[:limit]
        return SearchResponse(query=q, results=merged, count=len(merged))

    # Parse instrument filter
    inst_filter = None
    if instrument:
        try:
            inst_filter = Instrument(instrument.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid instrument: {instrument}. Use: {', '.join(e.value for e in Instrument)}.")

    # Auto-detect instrument from query if not specified
    if inst_filter is None:
        if is_crism_product_id(q):
            inst_filter = Instrument.CRISM
        elif is_hirise_product_id(q):
            inst_filter = Instrument.HIRISE
        elif is_sharad_highres_product_id(q):
            inst_filter = Instrument.SHARAD_HIGHRES
        elif is_sharad_product_id(q):
            inst_filter = Instrument.SHARAD
        elif is_hirise_dtm_product_id(q):
            inst_filter = Instrument.HIRISE_DTM

    # Handle HIRISE_DTM search - local index only
    if inst_filter == Instrument.HIRISE_DTM:
        dtm_results = search_local_index("HIRISE_DTM", "hirise_dtm", q, limit)
        return SearchResponse(
            query=q,
            results=dtm_results,
            count=len(dtm_results),
        )

    # Handle SHARAD High-Res search via ODE
    if inst_filter == Instrument.SHARAD_HIGHRES:
        try:
            products = await search_sharad_highres_products(q, limit)
        except Exception as e:
            raise HTTPException(500, f"SHARAD High-Res search failed: {e}")

        results = []
        for p in products:
            base_key = p.product_id.upper()
            existence = check_local_existence_detailed(p.product_id, p.instrument)

            results.append(SearchResult(
                product_id=p.product_id,
                instrument=p.instrument.value,
                base_key=base_key,
                lat=p.lat,
                lon=p.lon,
                exists=existence.exists,
                has_core=existence.has_core,
                has_browse=existence.has_browse,
                missing_files=existence.missing_files,
            ))

        return SearchResponse(
            query=q,
            results=results,
            count=len(results),
        )

    # Handle SHARAD search - first check local, then ODE
    if inst_filter == Instrument.SHARAD:
        # First check local index
        sharad_results = search_local_sharad(q, limit)

        # If no local results, search ODE for USRDRV2 products
        if not sharad_results:
            try:
                products = await search_sharad_products(q, limit)
                for p in products:
                    base_key = p.product_id.upper().replace("_THM", "")
                    existence = check_local_existence_detailed(p.product_id, p.instrument)

                    sharad_results.append(SearchResult(
                        product_id=p.product_id,
                        instrument=p.instrument.value,
                        base_key=base_key,
                        lat=p.lat,
                        lon=p.lon,
                        exists=existence.exists,
                        has_core=existence.has_core,
                        has_browse=existence.has_browse,
                        missing_files=existence.missing_files,
                    ))
            except Exception as e:
                print(f"SHARAD ODE search error: {e}")

        return SearchResponse(
            query=q,
            results=sharad_results,
            count=len(sharad_results),
        )

    # Search ODE for CRISM/HiRISE
    try:
        products = await search_ode_products(q, inst_filter, limit)
    except Exception as e:
        raise HTTPException(500, f"ODE search failed: {e}")

    # Convert to response format with existence check
    results = []
    for p in products:
        # Determine base_key
        if p.instrument == Instrument.CRISM:
            base_key = parse_crism_base_key(p.product_id)
        else:
            base_key = p.product_id.upper()

        # Check local existence (detailed)
        existence = check_local_existence_detailed(p.product_id, p.instrument)

        results.append(SearchResult(
            product_id=p.product_id,
            instrument=p.instrument.value,
            base_key=base_key,
            lat=p.lat,
            lon=p.lon,
            exists=existence.exists,
            has_core=existence.has_core,
            has_browse=existence.has_browse,
            missing_files=existence.missing_files,
        ))

    return SearchResponse(
        query=q,
        results=results,
        count=len(results),
    )


@router.get("/search/local")
@limiter.limit("30/minute")
async def search_local_products(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query (product ID or partial)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """
    Search ALL local products and Mars regions.

    Searches CRISM, HiRISE, SHARAD, CTX index.geojson files plus Mars region lookup table.
    Returns matching products/regions sorted by relevance (prefix match first).
    Regions appear with instrument="REGION" and can be used to fly to locations.
    """
    from .mars_regions import find_all_matching_regions

    products = _load_all_products()
    query_upper = q.upper()
    query_lower = q.lower()

    # Score and filter products
    scored = []
    for p in products:
        pid_upper = p["product_id"].upper()
        title_upper = (p.get("title") or "").upper()
        if query_upper in pid_upper or query_upper in title_upper:
            # Score: 0 = exact, 1 = prefix, 2 = substring
            if pid_upper == query_upper:
                score = 0
            elif pid_upper.startswith(query_upper):
                score = 1
            elif title_upper.startswith(query_upper):
                score = 1
            else:
                score = 2
            scored.append((score, p))

    # Search Mars regions
    matching_regions = find_all_matching_regions(query_lower)
    for region in matching_regions:
        region_name_upper = region.name.upper()
        # Score regions: prioritize exact/prefix matches
        if region_name_upper == query_upper:
            score = 0
        elif region_name_upper.startswith(query_upper):
            score = 1
        else:
            score = 2
        scored.append((score, {
            "product_id": region.name,  # Use name as ID for display
            "instrument": "REGION",
            "title": region.description,
            "lat": region.lat,
            "lon": region.lon,
            "radius_deg": region.radius_deg,
        }))

    scored.sort(key=lambda x: (x[0], x[1]["product_id"]))
    results = [s[1] for s in scored[:limit]]

    return {
        "query": q,
        "results": results,
        "count": len(results),
        "total_products": len(products),
    }


@router.get("/search/spatial", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_spatial(
    request: Request,
    minlat: float = Query(..., ge=-90, le=90, description="Southern latitude boundary"),
    maxlat: float = Query(..., ge=-90, le=90, description="Northern latitude boundary"),
    westernlon: float = Query(..., ge=-360, le=360, description="Western longitude boundary"),
    easternlon: float = Query(..., ge=-360, le=360, description="Eastern longitude boundary"),
    instrument: Optional[str] = Query(None, description="Filter by instrument"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
):
    """
    Search ODE for products within a bounding box.

    This endpoint finds products whose footprints intersect a rectangular area
    defined by latitude/longitude boundaries.

    ## Coordinate System

    - **Latitude**: -90 to 90 degrees (Mars planetocentric)
    - **Longitude**: Can be 0-360 (east positive) or -180 to 180
    - **minlat/maxlat**: Southern/Northern latitude boundaries
    - **westernlon/easternlon**: Western/Eastern longitude boundaries

    ## ODE REST Query

    The underlying ODE query uses:
    ```
    https://oderest.rsl.wustl.edu/live2/?target=mars&minlat={minlat}&maxlat={maxlat}&westernlon={westernlon}&easternlon={easternlon}&...
    ```

    ## Example

    ```
    GET /api/search/spatial?minlat=35&maxlat=70&westernlon=-130&easternlon=150
    ```

    Returns products in the Arcadia Planitia region.
    """
    if minlat < -90 or minlat > 90 or maxlat < -90 or maxlat > 90:
        raise HTTPException(400, "Latitude must be between -90 and 90")
    if westernlon < -360 or westernlon > 360 or easternlon < -360 or easternlon > 360:
        raise HTTPException(400, "Longitude must be between -360 and 360")

    # Validate lat range
    if minlat > maxlat:
        raise HTTPException(400, "minlat must be less than or equal to maxlat")

    # Handle CRISM_TRR3 spatial search - local index + ODE TRDR
    if instrument and instrument.lower() == "crism_trr3":
        local_results = search_local_spatial(
            "CRISM_TRR3", "crism_trr3", minlat, maxlat, westernlon, easternlon, limit
        )
        ode_results = await _search_ode_trr3_spatial(minlat, maxlat, westernlon, easternlon, limit)
        local_keys = {r.base_key for r in local_results}
        merged = list(local_results)
        for r in ode_results:
            if r.base_key not in local_keys:
                merged.append(r)
        merged = merged[:limit]
        return SearchResponse(
            query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
            results=merged, count=len(merged),
        )

    inst_filter = None
    if instrument:
        try:
            inst_filter = Instrument(instrument.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid instrument: {instrument}")

    # HIRISE_DTM: search local index instead of ODE
    if inst_filter == Instrument.HIRISE_DTM:
        dtm_results = search_local_spatial(
            "HIRISE_DTM", "hirise_dtm", minlat, maxlat, westernlon, easternlon, limit
        )
        return SearchResponse(
            query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
            results=dtm_results,
            count=len(dtm_results),
        )

    # Local-first: search local indices first for faster response
    _inst_local_map = {
        Instrument.CRISM: ("CRISM", "crism"),
        Instrument.HIRISE: ("HIRISE", "hirise"),
        Instrument.SHARAD: ("SHARAD", "sharad"),
        Instrument.SHARAD_HIGHRES: ("SHARAD_HIGHRES", "sharad_highres"),
    }

    local_results = []
    if inst_filter and inst_filter in _inst_local_map:
        key, val = _inst_local_map[inst_filter]
        local_results = search_local_spatial(
            key, val, minlat, maxlat, westernlon, easternlon, limit
        )
    elif inst_filter is None:
        # Search all local indices
        for key, val in _inst_local_map.values():
            local_results.extend(search_local_spatial(
                key, val, minlat, maxlat, westernlon, easternlon, limit
            ))
        local_results = local_results[:limit]

    # If local results are sufficient, skip slow ODE call
    if len(local_results) >= limit:
        return SearchResponse(
            query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
            results=local_results,
            count=len(local_results),
        )

    # Supplement with ODE results
    try:
        products = await search_ode_spatial(minlat, maxlat, westernlon, easternlon, inst_filter, limit)
    except Exception as e:
        # If ODE fails but we have local results, return those
        if local_results:
            return SearchResponse(
                query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
                results=local_results,
                count=len(local_results),
            )
        raise HTTPException(500, f"ODE spatial search failed: {e}")

    # Merge: local results first, then ODE results not already present
    local_pids = {r.product_id.upper() for r in local_results}
    results = list(local_results)
    for p in products:
        if p.product_id.upper() in local_pids:
            continue
        if p.instrument == Instrument.CRISM:
            base_key = parse_crism_base_key(p.product_id)
        else:
            base_key = p.product_id.upper()

        existence = check_local_existence_detailed(p.product_id, p.instrument)

        results.append(SearchResult(
            product_id=p.product_id,
            instrument=p.instrument.value,
            base_key=base_key,
            lat=p.lat,
            lon=p.lon,
            exists=existence.exists,
            has_core=existence.has_core,
            has_browse=existence.has_browse,
            missing_files=existence.missing_files,
        ))

    results = results[:limit]

    return SearchResponse(
        query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
        results=results,
        count=len(results),
    )


# =============================================================================
# Existence Check Endpoint
# =============================================================================

@router.get("/exists/{instrument}/{product_id}")
@limiter.limit("30/minute")
async def check_exists(request: Request, instrument: str, product_id: str):
    """
    Check if a product already exists locally (with detailed file info).

    ## Storage Layout

    - **CRISM**: `data/crism/{base_key}/`
    - **HiRISE**: `data/hirise/{product_id}/`

    For CRISM, the base_key is the first two underscore-separated tokens:
    - `frt0001fd76_07_if166j_mtr3` → `frt0001fd76_07`

    ## Response

    - `exists`: True only if ALL required files are present
    - `has_core`: True if core data files (.img, .lbl) exist
    - `has_browse`: True if at least one browse PNG exists
    - `missing_files`: List of missing file types
    - `existing_files`: List of existing file types

    ## Example

    ```
    GET /api/exists/crism/frt00009312_07_if165l_trr3
    ```
    """
    try:
        product_id = validate_product_id(product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    # Handle crism_trr3 separately (not in Instrument enum — uses mineral_cnn_data/)
    if instrument.lower() == "crism_trr3":
        from .mineral_cnn.constants import TRR_DATA_DIR
        obs_dir = os.path.join(TRR_DATA_DIR, product_id)
        if not os.path.isdir(obs_dir):
            # Try with _07 suffix
            obs_dir = os.path.join(TRR_DATA_DIR, f"{product_id}_07")
        has_data = os.path.isdir(obs_dir)
        has_trr = False
        if has_data:
            files = os.listdir(obs_dir)
            has_trr = any(f.upper().endswith("_TRR3.IMG") for f in files)
        return {
            "exists": has_data and has_trr,
            "product_id": product_id,
            "base_key": product_id,
            "instrument": "crism_trr3",
            "has_core": has_trr,
            "has_header": has_data and any(f.upper().endswith("_TRR3.LBL") for f in (os.listdir(obs_dir) if has_data else [])),
            "has_wavelength": False,
            "has_browse": False,
            "missing_files": [],
            "existing_files": os.listdir(obs_dir) if has_data else [],
        }

    try:
        inst = Instrument(instrument.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid instrument: {instrument}. Valid options: crism, hirise, sharad, sharad_highres, crism_trr3")

    if inst == Instrument.CRISM:
        base_key = parse_crism_base_key(product_id)
    elif inst == Instrument.SHARAD:
        base_key = product_id.upper().replace("_THM", "")
    else:
        base_key = product_id.upper()

    existence = check_local_existence_detailed(product_id, inst)

    return {
        "exists": existence.exists,
        "product_id": product_id,
        "base_key": base_key,
        "instrument": inst.value,
        "has_core": existence.has_core,
        "has_header": existence.has_header,
        "has_wavelength": existence.has_wavelength,
        "has_browse": existence.has_browse,
        "missing_files": existence.missing_files,
        "existing_files": existence.existing_files,
    }


# =============================================================================
# Download Endpoints
# =============================================================================

@router.post("/download", response_model=DownloadResponse)
@limiter.limit("20/minute")
async def start_download(http_request: Request, request: DownloadRequest):
    """
    Start downloading a product bundle (or missing files only).

    This initiates an async download task that fetches files for the
    specified product from ODE and stores them locally.

    ## CRISM Bundle Contents

    For CRISM products, downloads include:
    - **Core files**: `*.img`, `*.lbl`, `*.hdr` (if available)
    - **Wavelength table**: `*_wv*_mtr3.tab` or `*_trr3.tab`
    - **Browse images**: HYD, ICE, IC2, VNA products

    ## HiRISE Bundle Contents

    For HiRISE products, downloads include:
    - **RED JP2**: `*_RED.JP2` (main grayscale image)
    - **Label**: `*_RED.lbl`

    After download, JP2 is converted to GeoTIFF using GDAL.

    ## Download Options

    - **Full download**: If product doesn't exist, downloads all files
    - **Missing files only**: If some files exist, only downloads missing ones
    - **file_types**: Optional filter to download only specific types:
      - "core": .img, .lbl files
      - "header": .hdr file
      - "wavelength": .tab file
      - "browse": browse PNG files

    ## Storage Layout

    - **CRISM**: `data/crism/{base_key}/`
    - **HiRISE**: `data/hirise/{product_id}/`

    ## Example

    ```
    POST /api/download
    {
      "product_id": "frt00009312_07_if165l_trr3",
      "instrument": "crism",
      "lat": 18.5,
      "lon": -77.0,
      "file_types": ["browse"]  // Optional: only download browse files
    }
    ```
    """
    try:
        request.product_id = validate_product_id(request.product_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid product ID format")

    try:
        inst = Instrument(request.instrument.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid instrument: {request.instrument}. Valid options: {', '.join(e.value for e in Instrument)}.")

    # Check detailed existence - allow downloading missing files
    existence = check_local_existence_detailed(request.product_id, inst)

    # If fully downloaded and no specific file_types requested, reject
    if existence.exists and not request.file_types:
        raise HTTPException(409, "Product already fully downloaded")

    # Start download task (will skip existing files)
    task = await download_manager.start_download(
        request.product_id,
        inst,
        lat=request.lat,
        lon=request.lon,
        file_types=request.file_types,
    )

    return DownloadResponse(**task.to_dict())


@router.get("/download", response_model=List[DownloadResponse])
@limiter.limit("20/minute")
async def list_downloads(request: Request):
    """
    List all download tasks.

    Returns all active and completed download tasks with their current status.
    Useful for monitoring ongoing downloads or reviewing history.
    """
    tasks = download_manager.list_tasks()
    return [DownloadResponse(**t.to_dict()) for t in tasks]


@router.get("/download/{task_id}", response_model=DownloadResponse)
@limiter.limit("20/minute")
async def get_download_status(request: Request, task_id: str):
    """
    Get the status of a specific download task.

    ## Response Fields

    - `status`: One of `pending`, `queued`, `downloading`, `processing`, `completed`, `failed`
    - `files`: Array of individual file statuses with progress
    - `progress_percent`: Overall download progress (0-100)
    - `error`: Error message if status is `failed`

    ## Polling for Progress

    Frontend should poll this endpoint periodically (e.g., every 1-2 seconds)
    while `status` is `downloading` or `processing`.

    ## Example

    ```
    GET /api/download/a1b2c3d4
    ```
    """
    task = download_manager.get_task(task_id)

    if not task:
        raise HTTPException(404, f"Download task not found: {task_id}")

    return DownloadResponse(**task.to_dict())


@router.delete("/download/history")
@limiter.limit("20/minute")
async def clear_download_history(request: Request):
    """
    Clear all completed and failed download tasks from history.
    Active downloads are not affected.
    """
    removed = download_manager.clear_history()
    return {
        "status": "cleared",
        "removed_count": removed,
        "message": f"Cleared {removed} task(s) from history"
    }


@router.post("/download/{task_id}/retry", response_model=DownloadResponse)
@limiter.limit("20/minute")
async def retry_download(request: Request, task_id: str):
    """Retry a failed download task. Creates a new task with the same parameters."""
    task = await download_manager.retry_task(task_id)
    if not task:
        raise HTTPException(400, "Task not found or not in failed state")
    return DownloadResponse(**task.to_dict())


@router.delete("/download/{task_id}")
@limiter.limit("20/minute")
async def cancel_download(request: Request, task_id: str):
    """
    Cancel a specific download task.

    Terminates the aria2 process and marks the task as failed.

    ## Example

    ```
    DELETE /api/download/a1b2c3d4
    ```
    """
    task = download_manager.get_task(task_id)

    if not task:
        raise HTTPException(404, f"Download task not found: {task_id}")

    cancelled = await download_manager.cancel_task(task_id)

    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    else:
        return {"status": "not_active", "task_id": task_id, "message": "Task was not actively downloading"}


@router.delete("/download")
@limiter.limit("20/minute")
async def cancel_all_downloads(request: Request):
    """
    Cancel all active download tasks.

    Terminates all aria2 processes and marks tasks as failed.

    ## Example

    ```
    DELETE /api/download
    ```
    """
    cancelled_count = await download_manager.cancel_all_tasks()

    return {
        "status": "cancelled",
        "cancelled_count": cancelled_count,
        "message": f"Cancelled {cancelled_count} active download(s)"
    }
