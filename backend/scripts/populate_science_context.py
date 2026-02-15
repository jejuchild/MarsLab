"""
Populate mars_science_context.json using Gemini API.

Sends regions in batches to Gemini, asks it to fill in science context,
then populates the instruments and general_knowledge sections.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in backend/.env")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

# Try models in order — each has its own rate limit bucket
MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

DATA_FILE = Path(__file__).parent.parent / "data" / "mars_science_context.json"

_active_model = MODELS[0]

def call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini with model fallback and retry logic."""
    global _active_model
    for model in MODELS:
        _active_model = model
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": 0.3,
                        "max_output_tokens": 8192,
                    },
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt == max_retries - 1:
                        logger.warning(f"  {model} exhausted, trying next model...")
                        break  # try next model
                    time.sleep(3 * (attempt + 1))
                else:
                    logger.warning(f"  {model} attempt {attempt+1} error: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise
    raise RuntimeError("All Gemini models exhausted")


def extract_json(text: str) -> dict:
    """Extract JSON from Gemini response (may be wrapped in markdown)."""
    # Strip markdown code fences
    if "```json" in text:
        text = text.split("```json", 1)[1]
        text = text.split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    return json.loads(text.strip())


def populate_regions(data: dict) -> dict:
    """Populate region science context in batches."""
    regions = data["regions"]
    region_ids = list(regions.keys())

    # Process in batches of 10
    batch_size = 10
    for i in range(0, len(region_ids), batch_size):
        batch_ids = region_ids[i:i+batch_size]
        batch = {rid: regions[rid] for rid in batch_ids}

        logger.info(f"\n--- Regions batch {i//batch_size + 1}/{(len(region_ids) + batch_size - 1)//batch_size} ---")
        logger.info(f"  Processing: {', '.join(r['display_name'] for r in batch.values())}")

        prompt = f"""You are a Mars planetary science expert. Fill in the science context for these Mars regions.

For each region, provide:
- "science_context": 2-3 sentence scientific description of the region's geology, history, and significance
- "key_findings": array of 2-4 key scientific findings with paper citations (e.g., "SHARAD detected subsurface ice at 20-40m depth (Bramson et al. 2015)")
- "relevant_minerals": array of minerals detected or expected (e.g., ["water ice", "phyllosilicates", "olivine"])
- "ice_confidence": one of "none", "low", "medium", "high" — how confident we are about subsurface/surface ice presence
- "landing_site_suitability": 1 sentence about suitability for future human or robotic landing
- "references": array of key paper citations in "Author+Year" format (e.g., ["Dundas+2018", "Bramson+2015"])

Return ONLY valid JSON with the same structure. Keep the display_name unchanged.

Regions to fill:
{json.dumps(batch, indent=2)}"""

        try:
            response_text = call_gemini(prompt)
            filled = extract_json(response_text)

            # Merge back
            for rid in batch_ids:
                if rid in filled:
                    regions[rid] = filled[rid]
                    logger.info(f"  Filled: {regions[rid]['display_name']}")
                else:
                    logger.warning(f"  Missing from response: {rid}")
        except Exception as e:
            logger.error(f"  ERROR on batch: {e}")

        # Rate limit — be generous to avoid quota hits
        time.sleep(5)

    return data


def populate_instruments(data: dict) -> dict:
    """Populate instrument details."""
    logger.info("\n--- Instruments ---")

    prompt = f"""You are a Mars planetary science expert specializing in remote sensing instruments.
Fill in the details for these Mars orbital instruments used by MRO (Mars Reconnaissance Orbiter).

For CRISM, provide:
- "key_bands": object mapping wavelength to what it detects, e.g. {{"1.5um": "H2O ice absorption", "1.9um": "bound water / hydrated minerals"}}. Include at least 8-10 key diagnostic bands.
- "interpretation_notes": 2-3 sentences on how to interpret CRISM data
- "ice_relevant_signatures": 1-2 sentences on which spectral features indicate ice
- "common_minerals": array of commonly detected minerals
- "limitations": 1 sentence on limitations

For HIRISE, provide:
- "resolution": resolution string (e.g., "25-50 cm/pixel")
- "interpretation_notes": 2-3 sentences
- "ice_relevant_features": what ice-related surface features HiRISE can detect
- "limitations": 1 sentence

For SHARAD, provide:
- "frequency": operating frequency
- "penetration_depth": typical penetration depth
- "interpretation_notes": 2-3 sentences on interpreting radargrams
- "ice_detection_method": how SHARAD detects subsurface ice
- "clutter_notes": 1-2 sentences on surface clutter vs real subsurface returns
- "limitations": 1 sentence

For CTX, provide:
- "resolution": resolution string
- "interpretation_notes": 2-3 sentences
- "limitations": 1 sentence

For HIRISE_DTM, provide:
- "resolution": resolution of DTMs
- "interpretation_notes": 2-3 sentences
- "slope_analysis_notes": how DTMs are used for slope analysis in landing site selection
- "limitations": 1 sentence

Return ONLY valid JSON matching this structure (keep full_name and spacecraft fields unchanged):
{json.dumps(data["instruments"], indent=2)}"""

    try:
        response_text = call_gemini(prompt)
        filled = extract_json(response_text)
        data["instruments"] = filled
        logger.info("  Instruments filled successfully")
    except Exception as e:
        logger.error(f"  ERROR: {e}")

    return data


def populate_general_knowledge(data: dict) -> dict:
    """Populate general Mars knowledge."""
    logger.info("\n--- General Knowledge ---")

    prompt = f"""You are a Mars planetary science expert. Fill in these general Mars knowledge fields.

For "shallow_ice":
- "description": 2-3 sentences on Mars subsurface ice
- "latitude_dependence": 1-2 sentences on how ice stability varies with latitude
- "depth_range": typical depth range of shallow ice
- "detection_methods": array of methods used to detect subsurface ice (e.g., ["SHARAD radar sounding", "neutron spectroscopy", "fresh impact crater exposure"])
- "key_references": array of key papers in "Author+Year" format

For "landing_constraints":
- "latitude": latitude constraints for human missions (solar power, thermal)
- "elevation": elevation constraints for EDL (entry, descent, landing)
- "slope": maximum safe slopes for landing
- "rock_abundance": rock abundance constraints
- "dust_coverage": dust-related constraints

For "mineral_signatures":
- "phyllosilicates": 1-2 sentences on what phyllosilicates tell us about Mars history
- "sulfates": 1-2 sentences
- "carbonates": 1-2 sentences
- "olivine": 1-2 sentences
- "hematite": 1-2 sentences
- "perchlorate": 1-2 sentences (include ISRU relevance)

For "water_history":
- "noachian": 1-2 sentences on water during Noachian period (~4.1-3.7 Ga)
- "hesperian": 1-2 sentences on Hesperian period (~3.7-3.0 Ga)
- "amazonian": 1-2 sentences on Amazonian period (~3.0 Ga-present)

For "isru_relevance":
- "water_ice": 2-3 sentences on using ice for ISRU (in-situ resource utilization)
- "co2": 1-2 sentences on CO2 for ISRU
- "regolith": 1-2 sentences on regolith for ISRU

Return ONLY valid JSON matching this structure:
{json.dumps(data["general_knowledge"], indent=2)}"""

    try:
        response_text = call_gemini(prompt)
        filled = extract_json(response_text)
        data["general_knowledge"] = filled
        logger.info("  General knowledge filled successfully")
    except Exception as e:
        logger.error(f"  ERROR: {e}")

    return data


def main():
    # Load the empty scaffold
    with open(DATA_FILE) as f:
        data = json.load(f)

    logger.info("=== Populating mars_science_context.json with Gemini ===")

    # Step 1: Regions (batched)
    data = populate_regions(data)

    # Step 2: Instruments
    data = populate_instruments(data)

    # Step 3: General knowledge
    data = populate_general_knowledge(data)

    # Update meta
    data["meta"]["last_updated"] = "2025-02-10"
    data["meta"]["populated_by"] = f"Gemini ({_active_model})"

    # Write back
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== Done! Written to {DATA_FILE} ===")

    # Summary
    filled_regions = sum(1 for r in data["regions"].values() if r.get("science_context"))
    logger.info(f"  Regions filled: {filled_regions}/{len(data['regions'])}")


if __name__ == "__main__":
    main()
