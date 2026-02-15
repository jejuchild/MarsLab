"""
Gemini-based natural language parser for MarsLab.

Uses Google Gemini API to parse user's natural language requests
into structured JSON plans for Mars data download.

IMPORTANT: Gemini is used ONLY for parsing intent.
It does NOT download files or access the filesystem.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Load .env file from backend directory
from dotenv import load_dotenv
_backend_dir = Path(__file__).parent.parent
load_dotenv(_backend_dir / ".env")

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models to try in order of preference (fast/cheap first)
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",  # Fastest, lowest quota usage
    "gemini-2.0-flash",       # Fast and capable
    "gemini-2.5-flash",       # Latest flash model
]

# System prompt for Gemini (strict parsing mode)
SYSTEM_PROMPT = """You are a scientific data acquisition parser for MarsLab.
Your ONLY task is to convert a user's natural language request
into a structured JSON plan for Mars data search and download.
Output JSON ONLY.
Do not include explanations or markdown.
Do not wrap the JSON in code blocks.

Allowed instruments:
- CRISM (spectrometer, spectral, mineral, composition, hyperspectral)
- HIRISE (high resolution camera, imagery, surface detail)
- SHARAD (radar, subsurface, ice, radargram, sounding)
- SHARAD_HIGHRES (high-resolution radar, full RDR)
- CTX (context camera, wide-angle)
- HIRISE_DTM (digital terrain model, elevation, topography, DTM)

Known Mars regions (use BBOX coordinates when mentioned):
- Arcadia Planitia: bbox [38, 60, -170, -140] (northern lowland with ice)
- Amazonis Planitia: bbox [15, 40, -170, -145]
- Utopia Planitia: bbox [30, 60, 100, 140] (Zhurong site)
- Elysium Planitia: bbox [-10, 15, 140, 170] (InSight site)
- Isidis Planitia: bbox [5, 22, 80, 100]
- Hellas Planitia: bbox [-55, -30, 55, 85] (deepest basin)
- Chryse Planitia: bbox [15, 30, -45, -25] (Viking 1 site)
- Valles Marineris: bbox [-20, -5, -80, -30] (canyon system)
- Jezero Crater: bbox [17.8, 19.0, 77.0, 78.2] (Perseverance site)
- Gale Crater: bbox [-7, -4, 136, 139] (Curiosity site)
- Olympus Mons: bbox [15, 23, -137, -128] (largest volcano)
- Syrtis Major: bbox [0, 16, 62, 78]
- Meridiani Planum: bbox [-5, 2, -8, 5] (Opportunity site)
- Arabia Terra: bbox [5, 35, -15, 25]
- Nili Fossae: bbox [18, 24, 72, 78] (diverse mineralogy)
- Mawrth Vallis: bbox [19, 25, -22, -16] (phyllosilicates)
- Tharsis: bbox [-15, 15, -115, -85] (volcanic plateau)
- North Pole: bbox [78, 90, -180, 180]
- South Pole: bbox [-90, -78, -180, 180]

Output this exact JSON schema:
{
  "intent": "download",
  "targets": ["CRISM", "HIRISE", "SHARAD"],
  "region": {
    "type": "named_region | bbox | point | global",
    "name": "string or null",
    "bbox": {
      "minLat": number,
      "maxLat": number,
      "minLon": number,
      "maxLon": number
    },
    "point": {
      "lat": number,
      "lon": number
    },
    "radiusKm": number
  },
  "filters": {
    "maxResults": number,
    "spatialPredicate": "any | intersects | crosses | within | nearest",
    "distribution": "any | even | dense | sparse",
    "terrainHint": "string or null",
    "crossInstrument": null or {
      "instrument": "HIRISE_DTM",
      "titleContains": "string or null"
    }
  }
}

Rules:
- If user mentions specific instruments, include only those in targets
- If no instrument mentioned, include all: ["CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES"]
- For named regions, set type to "named_region", fill name AND bbox from the table above
- For coordinates, set type to "point" and fill point
- For explicit bounding box, set type to "bbox" and fill bbox
- If no region, set type to "global" with point/bbox null
- Default radiusKm is 100
- Default maxResults is 20
- Longitude in -180 to 180 range
- spatialPredicate: "intersects" for overlap, "crosses" for crossing, "nearest" for proximity
- distribution: "even" for evenly distributed, "dense" for high-density areas, "sparse" for low-density
- terrainHint: free-text description of terrain constraints (e.g. "craters", "low-density terrain", "ice deposits")
- crossInstrument: USE THIS when the user wants products from instrument A that spatially overlap/intersect with products from instrument B.
  Put only instrument A in "targets". Put instrument B in crossInstrument.instrument.
  If the user specifies a title/description filter on instrument B, put it in crossInstrument.titleContains.
  This is for queries like "find X that intersects/overlaps with Y" or "X products over Y coverage".

Examples:
- "Download SHARAD tracks that cross craters" -> targets=["SHARAD"], spatialPredicate="crosses", terrainHint="craters"
- "HiRISE and CRISM data evenly distributed in Arcadia Planitia" -> targets=["HIRISE","CRISM"], distribution="even"
- "High-res SHARAD intersecting with HiRISE DTM" -> targets=["SHARAD_HIGHRES"], spatialPredicate="intersects", crossInstrument={"instrument":"HIRISE_DTM","titleContains":null}
- "SHARAD_HIGHRES that intersects HiRISE DTM with title related to terraced crater" -> targets=["SHARAD_HIGHRES"], spatialPredicate="intersects", crossInstrument={"instrument":"HIRISE_DTM","titleContains":"terraced crater"}
- "CTX images overlapping with CRISM data in Jezero" -> targets=["CTX"], spatialPredicate="intersects", crossInstrument={"instrument":"CRISM","titleContains":null} """


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeminiRegion:
    """Region specification from Gemini."""
    type: str  # named_region, bbox, point, global
    name: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None  # minLat, maxLat, minLon, maxLon
    point: Optional[Dict[str, float]] = None  # lat, lon
    radiusKm: float = 100.0


@dataclass
class GeminiCrossFilter:
    """Cross-instrument spatial filter.

    Specifies a reference instrument whose products are used as the spatial
    reference for the primary target instruments.  Optionally filters the
    reference products by title substring.
    """
    instrument: str  # Reference instrument (e.g. "HIRISE_DTM")
    titleContains: Optional[str] = None  # Text filter on reference product title


@dataclass
class GeminiFilters:
    """Filters from Gemini."""
    maxResults: int = 20
    spatialPredicate: str = "any"  # any, intersects, crosses, within, nearest
    distribution: str = "any"  # any, even, dense, sparse
    terrainHint: Optional[str] = None
    crossInstrument: Optional[GeminiCrossFilter] = None


@dataclass
class GeminiPlan:
    """Parsed plan from Gemini."""
    intent: str
    targets: List[str]
    region: GeminiRegion
    filters: GeminiFilters
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent,
            "targets": self.targets,
            "region": asdict(self.region),
            "filters": asdict(self.filters),
            "error": self.error,
        }


# =============================================================================
# Validation
# =============================================================================

VALID_INSTRUMENTS = {"CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES", "CTX", "HIRISE_DTM"}
MAX_RESULTS_CAP = 50  # Hard limit on max results


def validate_plan(plan: GeminiPlan, user_max_results: int = 20) -> tuple[bool, str]:
    """
    Validate the parsed plan.

    Args:
        plan: Parsed plan from Gemini
        user_max_results: User-specified max results limit

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check intent
    if plan.intent != "download":
        return False, f"Invalid intent: {plan.intent}. Only 'download' is supported."

    # Check targets
    if not plan.targets:
        return False, "No instruments specified in targets."

    invalid_targets = set(plan.targets) - VALID_INSTRUMENTS
    if invalid_targets:
        return False, f"Invalid instruments: {invalid_targets}. Valid: {VALID_INSTRUMENTS}"

    # Check region
    region = plan.region
    if region.type not in ("named_region", "bbox", "point", "global"):
        return False, f"Invalid region type: {region.type}"

    # Validate coordinates if present
    if region.point:
        lat = region.point.get("lat")
        lon = region.point.get("lon")
        if lat is not None and (lat < -90 or lat > 90):
            return False, f"Invalid latitude: {lat}. Must be between -90 and 90."
        if lon is not None and (lon < -360 or lon > 360):
            return False, f"Invalid longitude: {lon}. Must be between -360 and 360."

    if region.bbox:
        minLat = region.bbox.get("minLat")
        maxLat = region.bbox.get("maxLat")
        minLon = region.bbox.get("minLon")
        maxLon = region.bbox.get("maxLon")

        if minLat is not None and (minLat < -90 or minLat > 90):
            return False, f"Invalid minLat: {minLat}"
        if maxLat is not None and (maxLat < -90 or maxLat > 90):
            return False, f"Invalid maxLat: {maxLat}"
        if minLat is not None and maxLat is not None and minLat > maxLat:
            return False, f"minLat ({minLat}) cannot be greater than maxLat ({maxLat})"

    # Check filters
    max_results = plan.filters.maxResults
    if max_results < 1:
        return False, "maxResults must be at least 1"
    if max_results > MAX_RESULTS_CAP:
        plan.filters.maxResults = MAX_RESULTS_CAP  # Silently cap
    if max_results > user_max_results:
        plan.filters.maxResults = user_max_results  # Respect user limit

    return True, ""


def normalize_longitude(lon: float) -> float:
    """Normalize longitude to -180 to 180 range."""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def normalize_plan_coordinates(plan: GeminiPlan) -> None:
    """Normalize all coordinates in the plan."""
    if plan.region.point:
        lon = plan.region.point.get("lon")
        if lon is not None:
            plan.region.point["lon"] = normalize_longitude(lon)

    if plan.region.bbox:
        minLon = plan.region.bbox.get("minLon")
        maxLon = plan.region.bbox.get("maxLon")
        if minLon is not None:
            plan.region.bbox["minLon"] = normalize_longitude(minLon)
        if maxLon is not None:
            plan.region.bbox["maxLon"] = normalize_longitude(maxLon)


# =============================================================================
# Gemini API Integration
# =============================================================================

_gemini_client = None


def get_gemini_client():
    """Get or create Gemini client."""
    global _gemini_client

    if _gemini_client is not None:
        return _gemini_client

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set, AI parsing will be unavailable")
        return None

    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
        return _gemini_client
    except ImportError:
        logger.error("google-genai package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        return None


async def parse_with_gemini(query: str, max_results: int = 20) -> GeminiPlan:
    """
    Parse natural language query using Gemini.

    Args:
        query: User's natural language query
        max_results: User-specified max results

    Returns:
        GeminiPlan with parsed intent or error
    """
    client = get_gemini_client()

    if client is None:
        return GeminiPlan(
            intent="error",
            targets=[],
            region=GeminiRegion(type="global"),
            filters=GeminiFilters(maxResults=max_results),
            error="Gemini API not available. Please check GEMINI_API_KEY."
        )

    # Build the prompt
    user_prompt = f"""Parse this Mars data request:
"{query}"

User's max results limit: {max_results}

Output JSON only."""

    # Try models in order of preference
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            logger.info(f"Calling Gemini model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=[
                    {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]}
                ],
                config={
                    "temperature": 0.1,  # Low temperature for consistent parsing
                    "max_output_tokens": 1024,
                }
            )

            # Extract response text
            response_text = response.text.strip()
            logger.info(f"Gemini response: {response_text[:200]}...")

            # Parse JSON response
            plan = parse_gemini_response(response_text, max_results)
            plan.raw_response = response_text

            # Validate and normalize
            normalize_plan_coordinates(plan)
            is_valid, error_msg = validate_plan(plan, max_results)

            if not is_valid:
                plan.error = error_msg
                logger.warning(f"Plan validation failed: {error_msg}")

            return plan

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Gemini model {model_name} failed: {e}")
            continue

    # All models failed
    return GeminiPlan(
        intent="error",
        targets=[],
        region=GeminiRegion(type="global"),
        filters=GeminiFilters(maxResults=max_results),
        error=f"All Gemini models failed. Last error: {last_error}"
    )


def parse_gemini_response(response_text: str, default_max_results: int = 20) -> GeminiPlan:
    """
    Parse Gemini's JSON response into a GeminiPlan.

    Args:
        response_text: Raw response from Gemini
        default_max_results: Default max results if not specified

    Returns:
        GeminiPlan parsed from response
    """
    # Clean up response (remove markdown code blocks if present)
    text = response_text.strip()
    if text.startswith("```"):
        # Remove code block markers
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON: {e}\nResponse: {text}")
        return GeminiPlan(
            intent="error",
            targets=[],
            region=GeminiRegion(type="global"),
            filters=GeminiFilters(maxResults=default_max_results),
            error=f"Failed to parse Gemini response as JSON: {e}"
        )

    # Extract fields with defaults
    intent = data.get("intent", "download")
    targets = data.get("targets", ["CRISM", "HIRISE", "SHARAD", "SHARAD_HIGHRES"])

    # Parse region
    region_data = data.get("region", {})
    region = GeminiRegion(
        type=region_data.get("type", "global"),
        name=region_data.get("name"),
        bbox=region_data.get("bbox"),
        point=region_data.get("point"),
        radiusKm=region_data.get("radiusKm", 100.0),
    )

    # Parse filters
    filters_data = data.get("filters", {})
    cross_data = filters_data.get("crossInstrument")
    cross_filter = None
    if cross_data and isinstance(cross_data, dict) and cross_data.get("instrument"):
        cross_filter = GeminiCrossFilter(
            instrument=cross_data["instrument"],
            titleContains=cross_data.get("titleContains"),
        )
    filters = GeminiFilters(
        maxResults=filters_data.get("maxResults", default_max_results),
        spatialPredicate=filters_data.get("spatialPredicate", "any"),
        distribution=filters_data.get("distribution", "any"),
        terrainHint=filters_data.get("terrainHint"),
        crossInstrument=cross_filter,
    )

    return GeminiPlan(
        intent=intent,
        targets=targets,
        region=region,
        filters=filters,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def is_gemini_available() -> bool:
    """Check if Gemini API is available."""
    return bool(GEMINI_API_KEY) and get_gemini_client() is not None


def get_gemini_status() -> Dict[str, Any]:
    """Get Gemini API status."""
    return {
        "available": is_gemini_available(),
        "api_key_set": bool(GEMINI_API_KEY),
        "models": GEMINI_MODELS,
    }
