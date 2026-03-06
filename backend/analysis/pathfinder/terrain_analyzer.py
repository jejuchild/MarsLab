"""VLM-powered Mars terrain analysis for Pathfinder route planning.

Renders multi-channel terrain images from DEM-derived cost map data and
sends them to a Vision Language Model (Llama 3.2 Vision via Groq, with
Gemini Vision fallback) for structured terrain classification, hazard
identification, and traversability assessment.

Terrain classification taxonomy follows NASA JPL SPOC (Soil Property and
Object Classification) categories extended for orbital imagery:
    - bedrock:   Exposed bedrock, stable surface
    - sand:      Aeolian deposits, wheel-slip risk
    - regolith:  General regolith, nominal traversability
    - rocky:     Dense boulder / rock field
    - ice_rich:  Periglacial / ice-bearing terrain
    - mixed:     Heterogeneous surface

References:
    [1] Rothrock et al., "SPOC: Deep Learning-based Terrain Classification
        for Mars Rover Missions," AIAA SciTech, 2016
    [2] Swan et al., "AI4Mars: A Dataset for Terrain-Aware Autonomous
        Driving on Mars," ICRA 2021
    [3] Anthropic, "AI-Planned Drives on Mars," anthropic.com/mars, 2026
"""

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import requests as http_requests
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

from .cost_map import CostMapResult

logger = logging.getLogger(__name__)

# ── Terrain Classification Constants ────────────────────────────

TERRAIN_TYPES = [
    "bedrock", "sand", "regolith", "rocky", "ice_rich", "mixed",
]

TRAVERSABILITY_LEVELS = ["easy", "moderate", "difficult", "impassable"]

RISK_LEVELS = ["low", "moderate", "high", "extreme"]

# Cost multipliers per terrain type (applied on top of slope-based cost)
TERRAIN_COST_MULTIPLIERS: dict[str, float] = {
    "bedrock":  0.8,
    "sand":     1.3,
    "regolith": 1.0,
    "rocky":    1.5,
    "ice_rich": 1.2,
    "mixed":    1.1,
}


# ── Data Classes ────────────────────────────────────────────────

@dataclass
class TerrainZone:
    """A classified terrain zone within the analysis area."""
    zone_id: int
    terrain_type: str           # One of TERRAIN_TYPES
    confidence: float           # 0.0 - 1.0
    traversability: str         # One of TRAVERSABILITY_LEVELS
    hazards: list[str]          # e.g. ["steep_slope", "loose_material"]
    description: str            # Natural language description
    bbox_pct: list[float]       # [x1, y1, x2, y2] as % of image (0-100)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VLMTerrainResult:
    """Complete VLM terrain analysis result."""
    zones: list[TerrainZone]
    overall_assessment: str          # Full natural language analysis
    recommended_corridors: list[str] # e.g. ["south", "central"]
    risk_level: str                  # One of RISK_LEVELS
    analysis_model: str              # Model used (e.g. "llama-3.2-90b-vision")
    terrain_image_b64: str = ""      # Composite image (for frontend display)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "zones": [z.to_dict() for z in self.zones],
            "overall_assessment": self.overall_assessment,
            "recommended_corridors": self.recommended_corridors,
            "risk_level": self.risk_level,
            "analysis_model": self.analysis_model,
        }
        if self.terrain_image_b64:
            d["terrain_image_b64"] = self.terrain_image_b64
        return d


# ── Terrain Image Rendering ─────────────────────────────────────

def render_elevation_image(
    elevation: np.ndarray,
    size: int = 512,
) -> bytes:
    """Render elevation grid as a grayscale PNG with contour-like shading.

    Higher = brighter. Normalised to the data range with 2nd-98th
    percentile stretch for contrast.
    """
    finite = elevation[np.isfinite(elevation)]
    if len(finite) == 0:
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        return _encode_png_rgb(blank)

    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax - vmin < 0.5:
        vmax = vmin + 1.0

    normed = np.clip((elevation - vmin) / (vmax - vmin), 0.0, 1.0)
    normed = np.nan_to_num(normed, nan=0.0)

    # Terrain colormap: dark brown (low) → tan → white (high)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mars_elev",
        [(0.0, "#3b1f0b"), (0.3, "#8b5e3c"), (0.6, "#c4a882"), (1.0, "#f5f0e8")],
    )
    rgba = cmap(normed)[:, :, :3]  # Drop alpha
    rgb = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return _encode_png_rgb_from_img(img)


def render_slope_image(
    slope: np.ndarray,
    max_slope_deg: float = 30.0,
    size: int = 512,
) -> bytes:
    """Render slope grid as a color-coded PNG.

    Green (flat) → Yellow (moderate) → Red (steep) → Purple (extreme).
    """
    normed = np.clip(slope / max_slope_deg, 0.0, 1.0)
    normed = np.nan_to_num(normed, nan=0.0)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mars_slope",
        [
            (0.0, "#16a34a"),   # Green — flat
            (0.3, "#eab308"),   # Yellow — moderate
            (0.6, "#ef4444"),   # Red — steep
            (1.0, "#7c3aed"),   # Purple — extreme
        ],
    )
    rgba = cmap(normed)[:, :, :3]
    rgb = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return _encode_png_rgb_from_img(img)


def render_composite_terrain_image(
    cost_result: CostMapResult,
    size: int = 512,
) -> bytes:
    """Render a 3-channel composite terrain image for VLM analysis.

    Red channel   = slope intensity (steeper = brighter red)
    Green channel = inverse cost    (traversable = brighter green)
    Blue channel  = hazard mask     (hazard = bright blue)

    This gives the VLM a single image encoding slope, traversability,
    and hazards simultaneously.
    """
    rows, cols = cost_result.slope_grid.shape

    # Red: slope (normalised)
    slope_norm = np.clip(cost_result.slope_grid / 30.0, 0.0, 1.0)
    slope_norm = np.nan_to_num(slope_norm, nan=0.0)

    # Green: inverse cost (low cost = bright green = easy)
    cost = cost_result.cost_grid.copy()
    finite_cost = cost[np.isfinite(cost)]
    if len(finite_cost) > 0:
        cmin = float(np.percentile(finite_cost, 2))
        cmax = float(np.percentile(finite_cost, 98))
        if cmax - cmin < 0.001:
            cmax = cmin + 1.0
        cost_norm = np.clip((cost - cmin) / (cmax - cmin), 0.0, 1.0)
        cost_norm = np.nan_to_num(cost_norm, nan=1.0)
        green = 1.0 - cost_norm  # Invert: low cost = bright
    else:
        green = np.zeros((rows, cols), dtype=np.float64)

    # Blue: hazard
    blue = cost_result.hazard_mask.astype(np.float64)

    # Compose RGB
    rgb = np.stack([slope_norm, green, blue], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return _encode_png_rgb_from_img(img)


def _encode_png_rgb(rgb: np.ndarray) -> bytes:
    """Encode RGB numpy array as PNG bytes."""
    img = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _encode_png_rgb_from_img(img: Image.Image) -> bytes:
    """Encode PIL Image as PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── VLM Integration (Groq Llama Vision primary, Gemini fallback) ───

# Analysis prompt template — SPOC-inspired terrain classification
TERRAIN_ANALYSIS_PROMPT = """You are an expert Mars terrain analyst working on rover route planning.
You are analyzing orbital imagery of the Martian surface derived from DEM (Digital Elevation Model) data.

**Image encoding:**
- RED channel = slope intensity (brighter = steeper terrain)
- GREEN channel = traversability (brighter = easier to traverse)
- BLUE channel = detected hazards (bright blue = hazard zone)

**Terrain metadata:**
- Area: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°E to {lon_max:.2f}°E
- Grid: {rows} × {cols} pixels, ~{px_m:.0f} m/pixel
- Elevation range: {elev_min:.0f} m to {elev_max:.0f} m
- Max slope: {slope_max:.1f}°, Mean slope: {slope_mean:.1f}°
- Hazard coverage: {hazard_pct:.1f}%
- Rover: {rover_name} (max safe slope: {max_slope:.0f}°)

**Task:** Analyze this terrain and return a JSON object with:
1. Identify 3-6 distinct terrain zones visible in the image
2. Classify each zone's terrain type
3. Assess overall traversability and risk
4. Recommend safe travel corridors

**Terrain types (pick one per zone):**
- "bedrock": Exposed bedrock, stable and flat
- "sand": Aeolian sand deposits, wheel-slip risk
- "regolith": Standard Martian regolith, nominal
- "rocky": Boulder field or dense rock coverage
- "ice_rich": Periglacial features, possible subsurface ice
- "mixed": Heterogeneous, mixed materials

**Output EXACTLY this JSON (no markdown, no code blocks):**
{{
  "zones": [
    {{
      "zone_id": 1,
      "terrain_type": "regolith",
      "confidence": 0.85,
      "traversability": "easy",
      "hazards": [],
      "description": "Flat regolith plain with minimal obstacles",
      "bbox_pct": [0, 0, 50, 50]
    }}
  ],
  "overall_assessment": "Natural language summary of the entire terrain area...",
  "recommended_corridors": ["south", "central"],
  "risk_level": "moderate"
}}

Rules:
- confidence: 0.0 to 1.0
- traversability: "easy" | "moderate" | "difficult" | "impassable"
- risk_level: "low" | "moderate" | "high" | "extreme"
- hazards: choose from ["steep_slope", "loose_material", "boulder_field", "crater_rim", "shadow_zone", "sand_trap", "cliff_edge", "rough_texture"]
- bbox_pct: [x1, y1, x2, y2] as percentage of image dimensions (0-100)
- recommended_corridors: general directions ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "central"]
- Output raw JSON only. No markdown wrapping."""


# ── Groq (Llama Vision) ──────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Vision models in order of preference (best quality first)
GROQ_VISION_MODELS = [
    "llama-3.2-90b-vision-preview",   # Best quality, slower
    "llama-3.2-11b-vision-preview",   # Faster, decent quality
]


def _call_groq_vision(
    prompt: str,
    image_b64: str,
    model: str | None = None,
) -> tuple[str | None, str]:
    """Call Groq Llama Vision API with image input.

    Returns (response_text, model_used) or (None, "") on failure.
    Uses OpenAI-compatible multimodal message format.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.info("GROQ_API_KEY not set — Groq Vision unavailable")
        return None, ""

    models_to_try = [model] if model else GROQ_VISION_MODELS
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(f"VLM terrain analysis calling Groq model: {model_name}")
            resp = http_requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_b64}",
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2048,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text, model_name

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Groq Vision model {model_name} failed: {e}")
            continue

    logger.error(f"All Groq Vision models failed. Last error: {last_error}")
    return None, ""


# ── Gemini Vision (fallback) ─────────────────────────────────────

GEMINI_VISION_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
]


def _get_gemini_client():
    """Get Gemini client from existing integration."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from api.gemini_parser import get_gemini_client
        return get_gemini_client()
    except Exception as e:
        logger.warning(f"Could not get Gemini client: {e}")
        return None


def _call_gemini_vision(
    prompt: str,
    image_b64: str,
) -> tuple[str | None, str]:
    """Call Gemini Vision API as fallback.

    Returns (response_text, model_used) or (None, "") on failure.
    """
    client = _get_gemini_client()
    if client is None:
        return None, ""

    last_error = None
    for model_name in GEMINI_VISION_MODELS:
        try:
            logger.info(f"VLM terrain analysis calling Gemini model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[{
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                    ],
                }],
                config={
                    "temperature": 0.2,
                    "max_output_tokens": 2048,
                },
            )
            return response.text.strip(), model_name

        except Exception as e:
            last_error = str(e)
            logger.warning(f"Gemini Vision model {model_name} failed: {e}")
            continue

    logger.error(f"All Gemini Vision models failed. Last error: {last_error}")
    return None, ""


# ── Main Analysis Entrypoint ─────────────────────────────────────

async def analyze_terrain_vlm(
    cost_result: CostMapResult,
    rover_name: str = "Perseverance",
    max_slope: float = 30.0,
    image_size: int = 512,
) -> VLMTerrainResult | None:
    """Run VLM terrain analysis on cost map data.

    Provider priority:
        1. Groq  — Llama 3.2 Vision (90B → 11B fallback)
        2. Gemini — Google Gemini Vision (flash → lite fallback)

    Returns VLMTerrainResult or None if all providers unavailable.
    This is a graceful-degradation feature — route planning works
    without it, VLM just adds richer terrain intelligence.
    """
    # 1. Render composite terrain image
    composite_png = render_composite_terrain_image(cost_result, size=image_size)
    composite_b64 = base64.b64encode(composite_png).decode("ascii")

    # 2. Build prompt with terrain metadata
    meta = cost_result.meta
    elevation = cost_result.elevation_grid
    slope = cost_result.slope_grid
    hazard = cost_result.hazard_mask

    finite_elev = elevation[np.isfinite(elevation)]
    finite_slope = slope[np.isfinite(slope)]

    prompt = TERRAIN_ANALYSIS_PROMPT.format(
        lat_min=meta.get("lat_min", 0),
        lat_max=meta.get("lat_max", 0),
        lon_min=meta.get("lon_min", 0),
        lon_max=meta.get("lon_max", 0),
        rows=meta.get("rows", elevation.shape[0]),
        cols=meta.get("cols", elevation.shape[1]),
        px_m=meta.get("px_m_ns", 200),
        elev_min=float(np.min(finite_elev)) if len(finite_elev) > 0 else 0,
        elev_max=float(np.max(finite_elev)) if len(finite_elev) > 0 else 0,
        slope_max=float(np.max(finite_slope)) if len(finite_slope) > 0 else 0,
        slope_mean=float(np.mean(finite_slope)) if len(finite_slope) > 0 else 0,
        hazard_pct=float(np.sum(hazard)) / max(hazard.size, 1) * 100,
        rover_name=rover_name,
        max_slope=max_slope,
    )

    # 3. Try Groq (Llama Vision) first, then Gemini fallback
    raw_text, model_used = _call_groq_vision(prompt, composite_b64)

    if raw_text is None:
        logger.info("Groq Vision unavailable — falling back to Gemini Vision")
        raw_text, model_used = _call_gemini_vision(prompt, composite_b64)

    if raw_text is None:
        logger.warning("All VLM providers unavailable — skipping terrain analysis")
        return None

    # 4. Parse structured response
    result = _parse_vlm_response(raw_text, model_used, composite_b64)
    return result


# ── Response Parsing ─────────────────────────────────────────────

def _parse_vlm_response(
    raw_text: str,
    model_used: str,
    composite_b64: str,
) -> VLMTerrainResult | None:
    """Parse VLM JSON response into VLMTerrainResult.

    Handles common LLM output quirks: markdown code blocks, trailing
    commas, partial JSON.
    """
    # Strip markdown code block wrappers if present
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.error(f"Failed to parse VLM response as JSON: {text[:200]}")
                return _build_fallback_result(model_used, composite_b64)
        else:
            logger.error(f"No JSON found in VLM response: {text[:200]}")
            return _build_fallback_result(model_used, composite_b64)

    # Validate and build result
    try:
        zones = []
        for z in data.get("zones", []):
            terrain_type = z.get("terrain_type", "mixed")
            if terrain_type not in TERRAIN_TYPES:
                terrain_type = "mixed"

            traversability = z.get("traversability", "moderate")
            if traversability not in TRAVERSABILITY_LEVELS:
                traversability = "moderate"

            zones.append(TerrainZone(
                zone_id=z.get("zone_id", len(zones) + 1),
                terrain_type=terrain_type,
                confidence=max(0.0, min(1.0, float(z.get("confidence", 0.5)))),
                traversability=traversability,
                hazards=z.get("hazards", []),
                description=z.get("description", ""),
                bbox_pct=z.get("bbox_pct", [0, 0, 100, 100]),
            ))

        risk_level = data.get("risk_level", "moderate")
        if risk_level not in RISK_LEVELS:
            risk_level = "moderate"

        return VLMTerrainResult(
            zones=zones if zones else [_default_zone()],
            overall_assessment=data.get("overall_assessment", "Analysis completed."),
            recommended_corridors=data.get("recommended_corridors", []),
            risk_level=risk_level,
            analysis_model=model_used,
            terrain_image_b64=composite_b64,
        )

    except Exception as e:
        logger.error(f"Error building VLM result from parsed data: {e}")
        return _build_fallback_result(model_used, composite_b64)


def _default_zone() -> TerrainZone:
    """Default zone when VLM gives no zone data."""
    return TerrainZone(
        zone_id=1,
        terrain_type="regolith",
        confidence=0.3,
        traversability="moderate",
        hazards=[],
        description="Terrain analysis inconclusive — defaulting to general regolith.",
        bbox_pct=[0, 0, 100, 100],
    )


def _build_fallback_result(model_used: str, composite_b64: str) -> VLMTerrainResult:
    """Fallback result when VLM response cannot be parsed."""
    return VLMTerrainResult(
        zones=[_default_zone()],
        overall_assessment=(
            "VLM terrain analysis returned an unparseable response. "
            "Route planning continues with slope-based cost analysis only."
        ),
        recommended_corridors=[],
        risk_level="moderate",
        analysis_model=model_used,
        terrain_image_b64=composite_b64,
    )


# ── Cost Grid Augmentation ──────────────────────────────────────

def compute_cost_adjustment(
    cost_result: CostMapResult,
    vlm_result: VLMTerrainResult,
) -> np.ndarray:
    """Compute a cost adjustment grid from VLM terrain zones.

    Returns a float32 multiplier grid (same shape as cost_grid).
    Values > 1.0 increase cost (harder terrain), < 1.0 decrease cost.
    Zones with higher confidence have stronger effect.
    """
    rows, cols = cost_result.cost_grid.shape
    adjustment = np.ones((rows, cols), dtype=np.float32)

    for zone in vlm_result.zones:
        # Convert bbox_pct to pixel coordinates
        x1_pct, y1_pct, x2_pct, y2_pct = zone.bbox_pct
        r1 = int(y1_pct / 100.0 * rows)
        r2 = int(y2_pct / 100.0 * rows)
        c1 = int(x1_pct / 100.0 * cols)
        c2 = int(x2_pct / 100.0 * cols)

        # Clamp
        r1 = max(0, min(r1, rows - 1))
        r2 = max(r1 + 1, min(r2, rows))
        c1 = max(0, min(c1, cols - 1))
        c2 = max(c1 + 1, min(c2, cols))

        # Get terrain cost multiplier
        base_mult = TERRAIN_COST_MULTIPLIERS.get(zone.terrain_type, 1.0)

        # Blend with confidence: high confidence → full multiplier,
        # low confidence → closer to 1.0 (no change)
        blended = 1.0 + (base_mult - 1.0) * zone.confidence

        # Apply to zone region
        adjustment[r1:r2, c1:c2] = blended

    return adjustment


def augment_cost_grid(
    cost_result: CostMapResult,
    vlm_result: VLMTerrainResult,
) -> np.ndarray:
    """Apply VLM terrain analysis to augment the cost grid.

    Returns a new cost grid (does NOT modify cost_result in place).
    """
    adjustment = compute_cost_adjustment(cost_result, vlm_result)
    augmented = cost_result.cost_grid.copy()

    # Only adjust finite costs (don't change impassable cells)
    finite_mask = np.isfinite(augmented)
    augmented[finite_mask] *= adjustment[finite_mask]

    return augmented
