"""
SWIM (Subsurface Water Ice Mapping) API — serves regional ice data and map tiles.

Endpoints:
  GET /api/swim/regions        — Return SWIM data for all regions
  GET /api/swim/comparison     — Return formatted comparison data for charts
  GET /api/swim/tile/{layer}   — Serve pre-rendered SWIM PNG map tile
  GET /api/swim/layers         — List available SWIM tile layers
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/swim", tags=["SWIM Data"])

BASE_DIR = Path(__file__).parent.parent
SCIENCE_CONTEXT_PATH = BASE_DIR / "data" / "mars_science_context.json"
SWIM_TILES_DIR = BASE_DIR / "data" / "swim" / "tiles"


def _load_science_context() -> dict:
    """Load mars_science_context.json."""
    try:
        with open(SCIENCE_CONTEXT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load mars_science_context.json: %s", e)
        return {}


@router.get("/regions")
def get_swim_regions():
    """Return SWIM data for all regions that have it."""
    data = _load_science_context()
    regions = []

    for region_key, region_val in data.get("regions", {}).items():
        if not isinstance(region_val, dict):
            continue
        swim = region_val.get("swim_data")
        if not swim:
            continue

        regions.append({
            "region_id": region_key,
            "name": region_val.get("display_name", region_key.replace("_", " ").title()),
            "ice_confidence": region_val.get("ice_confidence", "unknown"),
            "key_findings": region_val.get("key_findings", []),
            "swim_data": swim,
        })

    # Sort by ice_consistency_score descending
    regions.sort(
        key=lambda r: r["swim_data"].get("ice_consistency_score", 0),
        reverse=True,
    )

    return JSONResponse(content={"regions": regions, "total": len(regions)})


@router.get("/comparison")
def get_swim_comparison():
    """Return ice depth comparison data formatted for visualization charts.

    Returns arrays suitable for recharts BarChart / grouped comparison.
    """
    data = _load_science_context()
    comparison = []

    for region_key, region_val in data.get("regions", {}).items():
        if not isinstance(region_val, dict):
            continue
        swim = region_val.get("swim_data")
        if not swim:
            continue

        comparison.append({
            "region": region_val.get("display_name", region_key.replace("_", " ").title()),
            "region_id": region_key,
            "ice_consistency": swim.get("ice_consistency_score", 0),
            "depth_to_ice_m": swim.get("depth_to_ice_m", 0),
            "thermal_inertia": swim.get("thermal_inertia_Jm2Ks05", 0),
            "neutron_h2o_pct": swim.get("neutron_h2o_wt_pct", 0),
            "radar_dielectric": swim.get("radar_dielectric", 0),
            "confidence": swim.get("confidence", "unknown"),
        })

    # Sort by ice consistency descending
    comparison.sort(key=lambda r: r["ice_consistency"], reverse=True)

    return JSONResponse(content={
        "comparison": comparison,
        "metrics": [
            {"key": "ice_consistency", "label": "SWIM Ice Consistency", "unit": "", "range": [0, 1]},
            {"key": "depth_to_ice_m", "label": "Depth to Ice", "unit": "m", "range": [0, 20]},
            {"key": "thermal_inertia", "label": "Thermal Inertia", "unit": "J/m²/K/s⁰·⁵", "range": [0, 400]},
            {"key": "neutron_h2o_pct", "label": "Neutron H₂O", "unit": "wt%", "range": [0, 100]},
            {"key": "radar_dielectric", "label": "Radar Dielectric (ε')", "unit": "", "range": [1, 10]},
        ],
    })


# Available SWIM tile layers (pre-rendered PNGs from SWIM4MIM GeoTIFFs)
SWIM_LAYERS = {
    "0-1m": {
        "file": "swim_0_1m.png",
        "label": "Ice Consistency (0–1 m)",
        "description": "Shallow ice: neutron, thermal, geomorphology, radar surface power",
        "source": "SWIM4MIM (Morgan & Putzig et al. 2025)",
        "resolution_m": 3000,
    },
    "1-5m": {
        "file": "swim_1_5m.png",
        "label": "Ice Consistency (1–5 m)",
        "description": "Mid-depth: geomorphology, radar surface power, radar dielectric",
        "source": "SWIM4MIM (Morgan & Putzig et al. 2025)",
        "resolution_m": 3000,
    },
    ">5m": {
        "file": "swim_5m.png",
        "label": "Ice Consistency (>5 m)",
        "description": "Deep ice: deep geomorphology, radar dielectric estimates",
        "source": "SWIM4MIM (Morgan & Putzig et al. 2025)",
        "resolution_m": 3000,
    },
}


@router.get("/layers")
def list_swim_layers():
    """List available SWIM tile layers."""
    layers = []
    for layer_id, meta in SWIM_LAYERS.items():
        tile_path = SWIM_TILES_DIR / meta["file"]
        layers.append({
            "id": layer_id,
            "label": meta["label"],
            "description": meta["description"],
            "source": meta["source"],
            "resolution_m": meta["resolution_m"],
            "available": tile_path.is_file(),
            "size_bytes": tile_path.stat().st_size if tile_path.is_file() else 0,
            "bounds": {"west": -180, "south": -60, "east": 180, "north": 60},
        })
    return JSONResponse(content={"layers": layers})


@router.get("/tile/{layer}")
def get_swim_tile(layer: str):
    """Serve a pre-rendered SWIM PNG map tile.

    Tile covers -180°..180° lon, -60°..60° lat at 3km/pixel (7200×2400).
    Projection: simple cylindrical (equirectangular), Mars 2000 sphere.
    Source: SWIM4MIM (swim.psi.edu), Morgan & Putzig et al. 2025.
    """
    meta = SWIM_LAYERS.get(layer)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown layer '{layer}'. Available: {list(SWIM_LAYERS.keys())}")

    tile_path = SWIM_TILES_DIR / meta["file"]
    if not tile_path.is_file():
        raise HTTPException(status_code=404, detail=f"Tile file not found for layer '{layer}'")

    return FileResponse(
        tile_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-SWIM-Layer": layer,
            "X-SWIM-Source": meta["source"],
            "X-SWIM-Bounds": "-180,-60,180,60",
        },
    )
