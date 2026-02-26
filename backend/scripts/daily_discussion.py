#!/usr/bin/env python3
"""
MarsLab Daily AI Discussion Generator (v2 — Data-Grounded)
==========================================================

Generates daily AI-driven team discussions grounded in REAL MarsLab data:
  - HiRISE ice observations from hirise_ice.db (12,337 records)
  - Curated regional science context from mars_science_context.json
  - CRISM mineral confidence scores from score_stats.json
  - SHARAD depth estimates from sharad_reports/
  - Dielectric constant inversions from epsilon_results/
  - User field notes from field_notes.json
  - Recent git changes

Usage:
  python daily_discussion.py                    # Generate today's discussion
  python daily_discussion.py --date 2026-02-25  # Generate for a specific date
  python daily_discussion.py --dry-run          # Print prompt without calling API
  python daily_discussion.py --list-topics      # Show available topic focus areas

Cron setup (daily at 7am):
  0 7 * * * cd /disk1/cspark/MarsLab && python backend/scripts/daily_discussion.py

Output: backend/daily_discussions/YYYY-MM-DD.md
"""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUT_DIR = BACKEND_DIR / "daily_discussions"

# Data paths
DATA_DIR = BACKEND_DIR / "data"
HIRISE_ICE_DB = DATA_DIR / "hirise_ice.db"
SCIENCE_CONTEXT_FILE = DATA_DIR / "mars_science_context.json"
FIELD_NOTES_FILE = DATA_DIR / "field_notes.json"
CRISM_SCORE_STATS = BACKEND_DIR / "crism_score" / "score_stats.json"
SHARAD_REPORTS_DIR = BACKEND_DIR / "sharad_reports"
EPSILON_RESULTS_DIR = BACKEND_DIR / "hirise_dtm_data" / "epsilon_results"
DTM_METADATA_FILE = BACKEND_DIR / "hirise_dtm_data" / "crawl_metadata.json"

# Load .env
from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("daily_discussion")


# ---------------------------------------------------------------------------
# Topic Rotation
# ---------------------------------------------------------------------------

TOPIC_AREAS = [
    {
        "focus": "SHARAD Subsurface Analysis",
        "description": "Radar sounding interpretation, dielectric constant estimation, ice detection methodology, depth-to-reflector calculations, clutter discrimination",
        "science_keywords": ["SHARAD", "dielectric constant", "subsurface ice", "radargram", "reflectors", "permittivity", "two-way travel time"],
        "marslab_features": ["SharadHiresInspector", "Subsurface3DViewer", "RegolithPanel", "StratigraphyPanel", "epsilon inversion"],
    },
    {
        "focus": "CRISM Mineral Mapping",
        "description": "Spectral analysis, CNN mineral classification, band ratio interpretation, dust contamination effects on spectra, continuum removal methodology",
        "science_keywords": ["CRISM", "mineral classification", "spectral analysis", "band depth", "phyllosilicates", "hydration", "1.5/2.0 µm absorptions"],
        "marslab_features": ["Inspector", "BandRatioCalculator", "SpectralComparison", "CNN Mineral Classification", "MineralSequencePanel"],
    },
    {
        "focus": "Traverse Hazard Assessment",
        "description": "Slope analysis from HiRISE DTMs, terrain roughness quantification, thermal constraints on operations, ice proximity hazards, trafficability scoring",
        "science_keywords": ["slope", "traverse planning", "rover safety", "terrain roughness", "thermal inertia", "bearing capacity", "wheel sinkage"],
        "marslab_features": ["SlopeAnalysis3DTab", "HiRiseDTM3DViewer", "MeasurementTools", "DTMHoverReadout", "terrain overlays"],
    },
    {
        "focus": "Ice-Science Integration",
        "description": "Multi-criteria ice evidence fusion (SHARAD reflectors + CRISM hydration + terrain morphology + thermal inertia), SWIM-like consistency mapping, ice table depth estimation",
        "science_keywords": ["excess ice", "lobate debris aprons", "ice stability", "SWIM", "hydrogen abundance", "ice table depth", "sublimation lag"],
        "marslab_features": ["StratigraphyPanel", "AgenticPanel", "ice evidence fusion", "MapView overlays", "RegolithPanel"],
    },
    {
        "focus": "Data Pipeline Quality",
        "description": "Atmospheric correction methodology, dust detection algorithms, calibration validation, confidence scoring thresholds, CNN model performance metrics",
        "science_keywords": ["DISORT", "atmospheric correction", "dust opacity", "calibration", "confidence threshold", "aerosol scattering", "photometric correction"],
        "marslab_features": ["CNN confidence scores", "dust detection", "spectral pipeline", "data validation", "score maps"],
    },
    {
        "focus": "Multi-Instrument Correlation",
        "description": "Cross-referencing CTX, HiRISE, CRISM, SHARAD for comprehensive site characterization — spatial co-registration, temporal baseline analysis, resolution matching",
        "science_keywords": ["multi-instrument", "CTX context", "HiRISE detail", "data fusion", "spatial correlation", "co-registration"],
        "marslab_features": ["LayerPanel", "FieldNoteModal", "ReportPanel", "ComparisonMode", "product URL resolver"],
    },
    {
        "focus": "Arcadia Planitia Regional Science",
        "description": "Regional geological context — volcanic resurfacing history, ice emplacement timing, periglacial landform evolution, landing site engineering constraints",
        "science_keywords": ["Arcadia Planitia", "mid-latitude ice", "periglacial features", "polygonal terrain", "landing site", "thermal contraction cracks"],
        "marslab_features": ["MapView", "RegionDashboard", "DataDownloadPage", "terrain overlays", "CraterDetectPanel"],
    },
]

CHARACTERS = [
    {
        "name": "Dr. Elena Vasquez",
        "role": "Mars Geologist / Principal Investigator",
        "personality": "Passionate about mineralogy, always connects observations to geological processes. "
                       "Has published on Amazonian-age ice deposits. Pushes for science return. "
                       "Thinks in terms of paragenetic sequences and depositional environments. "
                       "Cites Feldman et al. 2004 on hydrogen abundance and Levy et al. 2009 on polygonal terrain.",
    },
    {
        "name": "James Park",
        "role": "Project Lead / Systems Engineer",
        "personality": "Practical, schedule-conscious, bridges science and engineering. "
                       "Asks 'how long will this take?' and 'what's the risk?'. "
                       "Converts science requirements into traverse waypoints. "
                       "Quantifies slope constraints: max 15° sustained, 25° peak for 10m.",
    },
    {
        "name": "Dr. Anika Rao",
        "role": "Rover Software Engineer / Autonomy Lead",
        "personality": "Safety-first mindset. Knows the rover's physical limits. "
                       "Thinks about sol-by-sol energy budgets, wheel slip on icy regolith, "
                       "thermal cycling effects on sampling mechanisms. "
                       "References Arvidson et al. 2017 on MER wheel sinkage in sulfate terrain.",
    },
    {
        "name": "Marcus Chen",
        "role": "Ground Operations Lead / MarsLab Power User",
        "personality": "Operates MarsLab daily. First to notice UI issues or suggest workflow improvements. "
                       "Has queried every API endpoint. Tracks which CRISM observations have CNN scores. "
                       "Knows exact counts: '1,504 CRISM observations scored, 12,337 HiRISE ice-tagged images'. "
                       "Suggests concrete MarsLab features with component names and API routes.",
    },
    {
        "name": "Dr. Fatima Al-Rashid",
        "role": "Spectral Data Scientist / CNN Model Lead",
        "personality": "Quantitative, careful with uncertainty. Built the CRISM mineral CNN. "
                       "Insists on reporting confidence intervals, not point estimates. "
                       "Knows that ice_score > 0.7 has 92% validation accuracy but struggles below 0.3. "
                       "Cites Viviano-Beck et al. 2014 on CRISM band parameters and Bishop et al. 2008 on phyllosilicate spectra. "
                       "Always asks: 'what's the false positive rate at this threshold?'",
    },
    {
        "name": "Dr. Yuri Petrov",
        "role": "Planetary Geophysicist / Radar Science Lead",
        "personality": "Deep SHARAD expertise. Thinks in permittivity and loss tangent. "
                       "Can estimate εr from two-way travel time and geometric depth in his head. "
                       "Cites Plaut et al. 2007 on SHARAD design, Seu et al. 2007 on MARSIS/SHARAD comparison, "
                       "and Campbell et al. 2008 on dielectric properties of Martian materials. "
                       "Distinguishes nadir clutter from real subsurface returns using along-track coherence. "
                       "Knows εr ≈ 3.0-3.15 implies >95% pure water ice, εr ≈ 4-6 implies ice-cemented regolith.",
    },
]


# ---------------------------------------------------------------------------
# Expanded Scientific Reference Library
# ---------------------------------------------------------------------------

SCIENCE_REFERENCES = """
## Arcadia Planitia Ice Science — Key Quantitative References

### Subsurface Ice Detection (SHARAD)
- Bramson et al. 2015, GRL: Detected massive subsurface ice deposit at Arcadia Planitia. Excess ice extends to ~170m depth in some locations. Dielectric constant ε'=3.0 (pure water ice). Best-fit models require >80% ice purity.
- Bramson et al. 2017, JGR: Mapped buried ice extent in Arcadia and Utopia. Thickest deposits (>100m) at 43-48°N. Volume estimate: 2.4×10⁴ km³ of ice in Arcadia alone.
- Plaut et al. 2007, Science: SHARAD instrument description. 15-25 MHz chirp, 15m free-space range resolution, 0.3-1 km along-track resolution. Penetration depth >1km in ice.
- Seu et al. 2007, JGR: SHARAD system design — 10W peak power, 85µs chirp, 10 MHz bandwidth centered at 20 MHz. Vertical resolution in ice: ~8.4m (for ε'=3.15).

### Ice Exposures and Morphology (HiRISE)
- Dundas et al. 2018, Science: Discovered 8 mid-latitude ice scarps at 55-58°N. Ice purity >90% (blue spectral signature). Scarp retreat rate ~1-2 mm/yr from sublimation. None in Arcadia directly, but implications for shallow ice accessibility.
- Dundas et al. 2014, JGR: Fresh impact craters at 39-55°N expose clean water ice at 1-10m depth. Craters at 43°N in Arcadia show ice within 1-2m of surface.
- Levy et al. 2009, JGR: Polygonal terrain at 40-60°N indicates thermal contraction of ice-rich permafrost. Polygon diameter 5-20m correlates with ice table depth (1-5m).
- Sizemore et al. 2015, Icarus: Ice-cemented crust thickness in Arcadia: 0.5-3m based on polygon morphology and thermal models.

### Ice Table Depth and Thermal Models
- Feldman et al. 2004, JGR: Mars Odyssey Neutron Spectrometer — hydrogen abundance at 40-50°N equivalent to >40% water-equivalent hydrogen by mass in upper 1m. Arcadia Planitia is within the high-H zone.
- Mellon & Jakosky 1995, JGR: Thermal model predicts ice table at 5-20cm depth above 50°N, deepening to 1-5m at 40°N depending on thermal inertia and albedo.
- Schorghofer 2007, Nature: Ice table oscillation model — current ice is receding equatorward from 30°N since last high-obliquity epoch (~5 Ma). Arcadia at 45°N is well within the stability zone.
- Putzig et al. 2005, Icarus: TES thermal inertia in Arcadia: 150-300 J m⁻² K⁻¹ s⁻½ (moderate, consistent with fine-grained ice-cemented regolith overlying pure ice).

### CRISM Spectral Indicators
- Viviano-Beck et al. 2014, JGR: CRISM type spectra catalog. Ice signature: 1.5µm and 2.0µm H₂O absorptions, 1.25µm Fresnel reflection peak. Band depth >0.02 for confident detection.
- Langevin et al. 2007, JGR: OMEGA detected surface water ice seasonal cycle at high latitudes. 1.5µm band depth decreases with dust contamination — dusty ice has band depth <0.01.
- Bishop et al. 2008, Clay Minerals: Phyllosilicate spectral library. Al-smectite (montmorillonite) at 2.21µm, Fe/Mg-smectite at 2.30µm. Distinguishing ice + dust from hydrated minerals requires careful band ratio analysis.

### SWIM and Ice Consistency
- Morgan et al. 2021, Nature Astronomy: SWIM (Subsurface Water Ice Mapping) — integrated 5 datasets (thermal, epithermal neutron, radar dielectric, radar geomorphology, geomorphology) at 200m/pixel. Arcadia scores: ice_consistency 0.78-0.92 (very high), depth_confidence: high.
- Pathare et al. 2018, Icarus: Excess ice fraction in Arcadia from SHARAD: 80-100% pure ice in top 50m, transitioning to 40-60% ice-regolith mixture below.

### Landing Site Engineering
- Putzig et al. 2023, Space Science Reviews: SWIM v2 — refined ice maps for human Mars missions. Arcadia Planitia remains top-3 ISRU site. Key concern: dust cover thickness (0.1-2m) before reaching competent ice.
- Dundas et al. 2021, JGR: Mid-latitude ice accessibility for ISRU — Arcadia and Utopia are primary candidates. Recommended drilling depth: 1-5m to reach massive ice.
"""


# ---------------------------------------------------------------------------
# Data-Grounding Context Gatherers
# ---------------------------------------------------------------------------

def get_recent_git_changes(days: int = 7) -> str:
    """Get a summary of recent git changes to MarsLab."""
    try:
        since = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        result = subprocess.run(
            ["git", "log", f"--since={since}", "--oneline", "--no-merges", "-20"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "(no recent changes detected)"


def get_science_context(region: str = "arcadia_planitia") -> str:
    """Load curated science context for the target region."""
    try:
        data = json.loads(SCIENCE_CONTEXT_FILE.read_text(encoding="utf-8"))
        region_data = data.get("regions", {}).get(region, {})
        if not region_data:
            return "(no science context available)"

        lines = [
            f"Region: {region_data.get('display_name', region)}",
            f"Science Context: {region_data.get('science_context', '')}",
            f"Ice Confidence: {region_data.get('ice_confidence', 'unknown')}",
            f"Landing Site Suitability: {region_data.get('landing_site_suitability', '')}",
            f"Relevant Minerals: {', '.join(region_data.get('relevant_minerals', []))}",
            "Key Findings:",
        ]
        for finding in region_data.get("key_findings", []):
            lines.append(f"  - {finding}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load science context: %s", e)
        return "(science context unavailable)"


def get_arcadia_hirise_products(date: datetime.date, n: int = 5) -> str:
    """Query hirise_ice.db for real HiRISE products in Arcadia Planitia."""
    if not HIRISE_ICE_DB.exists():
        return "(hirise_ice.db not found)"

    try:
        # Use date as seed for reproducible but varied selection
        seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
        conn = sqlite3.connect(str(HIRISE_ICE_DB))
        conn.row_factory = sqlite3.Row

        # Arcadia Planitia: lat 38-52°N, lon roughly -180 to -140°E
        cursor = conn.execute("""
            SELECT image_id, title, lat, lon, resolution_cm,
                   solar_longitude, acquisition_date, science_theme, caption
            FROM hirise_ice
            WHERE lat BETWEEN 38 AND 52
              AND lon BETWEEN -180 AND -140
            ORDER BY image_id
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "(no HiRISE ice products found in Arcadia Planitia)"

        # Deterministic shuffle using date seed
        rng = random.Random(seed)
        selected = rng.sample(rows, min(n, len(rows)))

        lines = [f"### Today's HiRISE Products ({len(selected)} of {len(rows)} Arcadia ice observations)"]
        for r in selected:
            caption_preview = (r["caption"] or "")[:200].replace("\n", " ").strip()
            res_str = '%.1f' % r['resolution_cm'] if r['resolution_cm'] else '?'
            ls_str = f', Ls={round(r["solar_longitude"])}°' if r['solar_longitude'] else ''
            acq_str = f', acquired {r["acquisition_date"]}' if r['acquisition_date'] else ''
            cap_str = f'\n  Caption: {caption_preview}' if caption_preview else ''
            lines.append(
                f"- **{r['image_id']}**: \"{r['title']}\" — "
                f"{r['lat']:.2f}°N, {r['lon']:.1f}°E, "
                f"{res_str}cm/px{ls_str}{acq_str}{cap_str}"
            )
        lines.append(f"\nTotal HiRISE ice observations in Arcadia: {len(rows)}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to query hirise_ice.db: %s", e)
        return f"(hirise_ice.db query error: {e})"


def get_crism_score_context(date: datetime.date, n: int = 4) -> str:
    """Load CRISM CNN mineral detection scores for discussion context."""
    if not CRISM_SCORE_STATS.exists():
        return "(CRISM score stats not found)"

    try:
        stats = json.loads(CRISM_SCORE_STATS.read_text(encoding="utf-8"))
        total = len(stats)

        # Find high-scoring ice observations
        ice_ranked = sorted(
            [(k, v) for k, v in stats.items() if "ice" in v],
            key=lambda x: x[1]["ice"].get("max_score", 0),
            reverse=True,
        )

        # Deterministic selection
        seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Pick some top scorers and some random ones
        top_ice = ice_ranked[:10]
        selected = rng.sample(top_ice, min(n, len(top_ice)))

        lines = [f"### CRISM Mineral Confidence Scores ({total} observations scored)"]
        for obs_id, data in selected:
            ice = data.get("ice", {})
            hyd = data.get("hyd", {})
            lines.append(
                f"- **{obs_id}**: ice_score max={ice.get('max_score', 0):.2f}, "
                f"mean={ice.get('mean_score', 0):.3f}, "
                f"pixels_above_0.5={ice.get('threshold_counts', {}).get('0.5', 0):,} / "
                f"{ice.get('valid_pixels', 0):,} valid | "
                f"hyd_score max={hyd.get('max_score', 0):.2f}, mean={hyd.get('mean_score', 0):.3f}"
            )

        # Global stats
        all_ice_max = [v["ice"]["max_score"] for v in stats.values() if "ice" in v and v["ice"].get("max_score")]
        if all_ice_max:
            lines.append(f"\nGlobal ice score distribution: "
                         f"max={max(all_ice_max):.2f}, "
                         f"median={sorted(all_ice_max)[len(all_ice_max)//2]:.2f}, "
                         f"observations with max_ice>0.5: {sum(1 for s in all_ice_max if s > 0.5)}/{total}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load CRISM scores: %s", e)
        return f"(CRISM score error: {e})"


def get_sharad_depth_context() -> str:
    """Load SHARAD depth estimation results from reports."""
    if not SHARAD_REPORTS_DIR.exists():
        return "(no SHARAD reports found)"

    lines = ["### SHARAD Subsurface Depth Estimates"]
    try:
        for report_dir in sorted(SHARAD_REPORTS_DIR.iterdir()):
            if not report_dir.is_dir() or report_dir.name.startswith("DEBUG"):
                continue
            csv_path = report_dir / "depth_estimation_table.csv"
            if not csv_path.exists():
                continue

            rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
            lines.append(f"\n**{report_dir.name}** — depth table:")
            for row in rows:
                mat = row.get("Material", "?")
                mean_d = row.get("Mean_Depth_m", "?")
                min_d = row.get("Min_Depth_m", "?")
                max_d = row.get("Max_Depth_m", "?")
                eps = row.get("Dielectric_Constant", "?")
                lines.append(f"  - {mat} (ε'={eps}): depth {mean_d}m (range {min_d}–{max_d}m)")

        if len(lines) == 1:
            return "(no SHARAD depth tables found)"
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load SHARAD depths: %s", e)
        return f"(SHARAD depth error: {e})"


def get_epsilon_context() -> str:
    """Load dielectric constant inversion results."""
    if not EPSILON_RESULTS_DIR.exists():
        return "(no epsilon results found)"

    lines = ["### Dielectric Constant (εr) Inversions from Terraced Craters"]
    try:
        for fpath in sorted(EPSILON_RESULTS_DIR.glob("epsilon_terrace_*.json")):
            data = json.loads(fpath.read_text(encoding="utf-8"))
            product = data.get("sharad_product", fpath.stem)
            estimates = data.get("epsilon_estimates", [])
            summary = data.get("summary", "")
            dtms = data.get("dtms", [])

            lines.append(f"\n**{product}** — {len(estimates)} εr estimates from {len(dtms)} nearby craters:")
            for dtm in dtms[:3]:
                lines.append(f"  Crater: \"{dtm.get('title', '?')}\" at {dtm.get('center_lat', 0):.2f}°N, "
                             f"{dtm.get('center_lon', 0):.2f}°E (distance: {dtm.get('distance_km', 0):.0f}km)")
            for est in estimates:
                lines.append(
                    f"  εr = {est.get('epsilon_r', 0):.2f} "
                    f"(range {est.get('epsilon_low', 0):.2f}–{est.get('epsilon_high', 0):.2f}), "
                    f"depth = {est.get('depth_true_m', 0):.1f} ± {est.get('depth_unc_m', 0):.1f}m, "
                    f"TWT = {est.get('twt_us', 0):.4f} µs — {est.get('interpretation', '?')}"
                )
            if summary:
                lines.append(f"  Summary: {summary}")

        if len(lines) == 1:
            return "(no epsilon inversions available)"
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load epsilon results: %s", e)
        return f"(epsilon error: {e})"


def get_dtm_context() -> str:
    """Load HiRISE DTM metadata for Arcadia-relevant products."""
    if not DTM_METADATA_FILE.exists():
        return "(no DTM metadata found)"

    try:
        data = json.loads(DTM_METADATA_FILE.read_text(encoding="utf-8"))
        arcadia_dtms = [d for d in data
                        if d.get("center_lat") and 35 <= d["center_lat"] <= 55]

        if not arcadia_dtms:
            return "(no Arcadia DTMs found)"

        lines = [f"### HiRISE DTMs in Arcadia Region ({len(arcadia_dtms)} of {len(data)} total)"]
        for d in arcadia_dtms[:6]:
            lines.append(
                f"- **{d.get('obs_id', '?')}**: \"{d.get('title', '?')}\" — "
                f"{d.get('center_lat', 0):.2f}°N, {d.get('center_lon', 0):.1f}°E, "
                f"res={d.get('resolution_m', '?')}m/px"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load DTM metadata: %s", e)
        return f"(DTM metadata error: {e})"


def get_field_notes_context() -> str:
    """Load user field notes for grounding in team observations."""
    if not FIELD_NOTES_FILE.exists():
        return "(no field notes found)"

    try:
        notes = json.loads(FIELD_NOTES_FILE.read_text(encoding="utf-8"))
        if not notes:
            return "(no field notes recorded)"

        lines = [f"### Team Field Notes ({len(notes)} annotations)"]
        for n in notes:
            lines.append(
                f"- [{n.get('instrument', '?')}] **{n.get('product_id', '?')}** "
                f"({n.get('lat', 0):.2f}°N, {n.get('lon', 0):.1f}°E): "
                f"\"{n.get('memo', '')}\" — tags: {', '.join(n.get('tags', []))} "
                f"({n.get('created_at', '?')[:10]})"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.warning("Failed to load field notes: %s", e)
        return f"(field notes error: {e})"


def get_previous_topics(n: int = 5) -> list[str]:
    """Get topics from the N most recent discussions to avoid repetition."""
    topics = []
    if OUTPUT_DIR.exists():
        files = sorted(OUTPUT_DIR.glob("*.md"), reverse=True)[:n]
        for f in files:
            for line in f.read_text(encoding="utf-8").splitlines()[:5]:
                if line.startswith("## Focus:"):
                    topics.append(line.replace("## Focus:", "").strip())
                    break
    return topics


def select_topic(date: datetime.date) -> dict:
    """Deterministically select today's topic, avoiding recent repeats."""
    previous = get_previous_topics()
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
    candidates = [t for t in TOPIC_AREAS if t["focus"] not in previous]
    if not candidates:
        candidates = TOPIC_AREAS
    return candidates[seed % len(candidates)]


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(date: datetime.date, topic: dict, context: dict) -> str:
    """Build the LLM prompt with real data grounding."""

    characters_desc = "\n".join(
        f"- **{c['name']}** ({c['role']}): {c['personality']}"
        for c in CHARACTERS
    )

    return f"""You are an expert science writer producing a high-quality simulated team meeting discussion for the MarsLab project — a Mars scientific data visualization and analysis platform used for planning a rover traverse in Arcadia Planitia.

Your output will be read by planetary scientists. It must be scientifically accurate, cite real data, and demonstrate genuine analytical reasoning — not generic summaries.

## Date: {date.isoformat()}

## Today's Focus: {topic['focus']}
{topic['description']}

## Science Keywords: {', '.join(topic['science_keywords'])}
## MarsLab Features to Reference: {', '.join(topic['marslab_features'])}

## Characters
{characters_desc}

## REAL DATA CONTEXT — Use These Exact Values

{context['science_context']}

{context['hirise_products']}

{context['crism_scores']}

{context['sharad_depths']}

{context['epsilon_results']}

{context['dtm_context']}

{context['field_notes']}

## Recent MarsLab Development Context
{context['git_changes']}

{SCIENCE_REFERENCES}

## CRITICAL INSTRUCTIONS

Write a scientifically rigorous team discussion of **2000-3000 words** with these 3 sections:

### Section 1: Today's Data Analysis (~1000 words)
The team works through a specific analysis workflow in MarsLab. Requirements:
- **Use the EXACT product IDs, coordinates, and measurements from the REAL DATA above** — do NOT invent fake data
- Show characters clicking specific MarsLab UI elements by name (Inspector, StratigraphyPanel, SlopeAnalysis3DTab, etc.)
- Include **quantitative observations**: band depths (e.g., "1.5µm absorption depth of 0.034"), slopes (e.g., "mean slope 8.3° with max 22.1°"), dielectric constants, ice scores
- Characters should **perform calculations on-screen**: "If TWT is 1.16µs and we assume ε'=3.15, that gives us d = c·t/2√ε' = 98m..."
- Show **data loading and interpretation**, not just discussing data abstractly

### Section 2: Scientific Interpretation & Debate (~800 words)
The team **genuinely disagrees** on interpretations. Requirements:
- Dr. Petrov and Dr. Al-Rashid should argue about what the SHARAD/CRISM data means — with specific numbers
- At least TWO scientists must propose **competing hypotheses** with different predictions
- Reference **at least 3 different published papers** by author name with specific findings (use the reference library above)
- Include **quantitative uncertainty discussion**: "The εr range of 4.7-52.0 is too wide for a definitive ice interpretation"
- One character should identify a **potential error or confounding factor** (e.g., surface clutter, dust contamination, seasonal CO₂ frost)

### Section 3: Actionable Outcomes (~700 words)
End with TWO concrete lists:

**MarsLab Improvements** (3-4 specific, implementable features):
- Each must reference a specific component, API endpoint, or data source
- Include HOW it would work technically (not just "add a feature to...")
- Prioritize by scientific value

**Research Insights** (3-4 findings relevant to the Arcadia traverse):
- Each must be tied to specific data from Section 1 (cite the product ID)
- Include uncertainty estimates: "We estimate ice table at 2.1 ± 0.8m based on..."
- At least one must be a **novel hypothesis** the team formulated during the discussion

## Style Requirements
- Natural conversation: interruptions, follow-ups, jokes, disagreements
- Scientists ARGUE with data, not assertions — "But Fatima, your CNN ice_score of 0.72 for frt00003156 conflicts with..."
- Characters mention specific MarsLab UI actions: "I just opened the SharadHiresInspector for R_0277201..."
- Use the characters' expertise: Petrov thinks in εr, Al-Rashid in spectral band depths, Vasquez in geological processes, Rao in rover constraints
- Reference the field notes from the team: "Marcus, you tagged that DTM with 'ice?' — what made you suspicious?"
- Include at least one moment of **genuine scientific insight** where connecting two datasets reveals something new

Write the discussion now. Start with: ## Focus: {topic['focus']}"""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_discussion(prompt: str) -> Optional[str]:
    """Call Groq API to generate the discussion."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set in environment. Cannot generate discussion.")
        return None

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.72,
                "max_tokens": 8192,
                "top_p": 0.92,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error("Groq API request failed: %s", e)
        return None
    except (KeyError, IndexError) as e:
        logger.error("Unexpected API response format: %s", e)
        return None


def save_discussion(date: datetime.date, topic: dict, content: str, context: dict) -> Path:
    """Save generated discussion to markdown file with rich metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{date.isoformat()}.md"

    header = f"""# MarsLab Daily Discussion — {date.strftime('%B %d, %Y')}

**Generated**: {datetime.datetime.now().isoformat()}
**Topic**: {topic['focus']}
**Science Keywords**: {', '.join(topic['science_keywords'])}
**MarsLab Features**: {', '.join(topic['marslab_features'])}
**Data Sources**: hirise_ice.db, mars_science_context.json, crism_score_stats, sharad_reports, field_notes
**Generator**: v2 (data-grounded)

---

"""
    filepath.write_text(header + content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MarsLab Daily AI Discussion Generator (v2 — Data-Grounded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Generate for specific date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompt without calling API.",
    )
    parser.add_argument(
        "--list-topics", action="store_true",
        help="List available topic focus areas.",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Override topic selection (use exact focus name from --list-topics).",
    )
    args = parser.parse_args()

    if args.list_topics:
        print("Available topic focus areas:\n")
        for i, t in enumerate(TOPIC_AREAS, 1):
            print(f"  {i}. {t['focus']}")
            print(f"     {t['description']}")
            print(f"     Keywords: {', '.join(t['science_keywords'])}")
            print(f"     Features: {', '.join(t['marslab_features'])}")
            print()
        return

    date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()

    # Select topic
    if args.topic:
        matching = [t for t in TOPIC_AREAS if t["focus"].lower() == args.topic.lower()]
        if not matching:
            logger.error("Topic '%s' not found. Use --list-topics to see options.", args.topic)
            sys.exit(1)
        topic = matching[0]
    else:
        topic = select_topic(date)

    logger.info("Date: %s | Topic: %s", date.isoformat(), topic["focus"])

    # Gather ALL context (data-grounded)
    logger.info("Gathering real data context...")
    context = {
        "git_changes": get_recent_git_changes(),
        "science_context": get_science_context("arcadia_planitia"),
        "hirise_products": get_arcadia_hirise_products(date, n=5),
        "crism_scores": get_crism_score_context(date, n=4),
        "sharad_depths": get_sharad_depth_context(),
        "epsilon_results": get_epsilon_context(),
        "dtm_context": get_dtm_context(),
        "field_notes": get_field_notes_context(),
    }

    for key, val in context.items():
        logger.info("  %s: %d chars", key, len(val))

    # Build prompt
    prompt = build_prompt(date, topic, context)
    logger.info("Prompt length: %d chars (~%d tokens est.)", len(prompt), len(prompt) // 4)

    if args.dry_run:
        print("=" * 72)
        print("DRY RUN — Prompt that would be sent to Groq API:")
        print("=" * 72)
        print(prompt)
        print("=" * 72)
        print(f"\nPrompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens est.)")
        print(f"Topic: {topic['focus']}")
        return

    # Generate
    logger.info("Generating discussion via Groq API (%s)...", GROQ_MODEL)
    content = generate_discussion(prompt)

    if content is None:
        logger.error("Generation failed. No output produced.")
        sys.exit(1)

    # Save
    filepath = save_discussion(date, topic, content, context)
    logger.info("Discussion saved to %s", filepath)
    logger.info("Word count: ~%d", len(content.split()))

    # Print summary
    print(f"\n✓ Daily discussion generated: {filepath}")
    print(f"  Topic: {topic['focus']}")
    print(f"  Words: ~{len(content.split())}")
    print(f"  Data sources: {sum(1 for v in context.values() if 'not found' not in v and 'error' not in v)}/8 active")


if __name__ == "__main__":
    main()
