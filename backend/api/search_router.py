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

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from .ode_client import (
    Instrument,
    ODEProduct,
    search_ode_products,
    search_ode_spatial,
    parse_crism_base_key,
    is_crism_product_id,
    is_hirise_product_id,
    is_sharad_product_id,
)
import json
from pathlib import Path
from .download_manager import (
    download_manager,
    check_local_existence,
    check_local_existence_detailed,
    DownloadStatus,
)


router = APIRouter(prefix="/api", tags=["Search & Download"])

# SHARAD index path
SHARAD_INDEX_PATH = Path(__file__).parent.parent / "sharad_data" / "index.geojson"


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
async def search_products(
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
    # Parse instrument filter
    inst_filter = None
    if instrument:
        try:
            inst_filter = Instrument(instrument.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid instrument: {instrument}. Use 'crism', 'hirise', or 'sharad'.")

    # Auto-detect instrument from query if not specified
    if inst_filter is None:
        if is_crism_product_id(q):
            inst_filter = Instrument.CRISM
        elif is_hirise_product_id(q):
            inst_filter = Instrument.HIRISE
        elif is_sharad_product_id(q):
            inst_filter = Instrument.SHARAD

    # Handle SHARAD search locally (already downloaded data)
    if inst_filter == Instrument.SHARAD:
        sharad_results = search_local_sharad(q, limit)
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


@router.get("/search/spatial", response_model=SearchResponse)
async def search_spatial(
    minlat: float = Query(..., ge=-90, le=90, description="Southern latitude boundary"),
    maxlat: float = Query(..., ge=-90, le=90, description="Northern latitude boundary"),
    westernlon: float = Query(..., ge=-180, le=360, description="Western longitude boundary"),
    easternlon: float = Query(..., ge=-180, le=360, description="Eastern longitude boundary"),
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
    # Validate lat range
    if minlat > maxlat:
        raise HTTPException(400, "minlat must be less than or equal to maxlat")

    inst_filter = None
    if instrument:
        try:
            inst_filter = Instrument(instrument.lower())
        except ValueError:
            raise HTTPException(400, f"Invalid instrument: {instrument}")

    try:
        products = await search_ode_spatial(minlat, maxlat, westernlon, easternlon, inst_filter, limit)
    except Exception as e:
        raise HTTPException(500, f"ODE spatial search failed: {e}")

    results = []
    for p in products:
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

    return SearchResponse(
        query=f"lat=[{minlat}, {maxlat}], lon=[{westernlon}, {easternlon}]",
        results=results,
        count=len(results),
    )


# =============================================================================
# Existence Check Endpoint
# =============================================================================

@router.get("/exists/{instrument}/{product_id}")
async def check_exists(instrument: str, product_id: str):
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
        inst = Instrument(instrument.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid instrument: {instrument}")

    if inst == Instrument.CRISM:
        base_key = parse_crism_base_key(product_id)
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
async def start_download(request: DownloadRequest):
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
        inst = Instrument(request.instrument.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid instrument: {request.instrument}")

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
async def list_downloads():
    """
    List all download tasks.

    Returns all active and completed download tasks with their current status.
    Useful for monitoring ongoing downloads or reviewing history.
    """
    tasks = download_manager.list_tasks()
    return [DownloadResponse(**t.to_dict()) for t in tasks]


@router.get("/download/{task_id}", response_model=DownloadResponse)
async def get_download_status(task_id: str):
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
