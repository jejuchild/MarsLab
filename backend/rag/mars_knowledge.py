"""
Built-in Mars science knowledge for RAG seeding.

Contains curated knowledge about Mars instruments, missions,
geology, and science concepts. Automatically ingested on first use.
"""

import logging
import os
from typing import List, Dict

from .ingestion import ingest_text

logger = logging.getLogger(__name__)

_SEEDED = False
_COLLECTION = "mars_science"

# ---------------------------------------------------------------------------
# Curated Mars Science Knowledge Base
# ---------------------------------------------------------------------------

MARS_KNOWLEDGE: List[Dict[str, str]] = [
    # ── Missions ──────────────────────────────────────────────────────────
    {
        "title": "Mars 2020 Perseverance Rover Mission Overview",
        "source": "marslab://knowledge/perseverance",
        "text": """
Mars 2020 Perseverance rover landed in Jezero Crater on February 18, 2021.
Jezero is a 45-km diameter impact crater located at 18.4°N, 77.5°E on the
northwest rim of Isidis Planitia. The crater once hosted an ancient lake
fed by a river delta, making it a prime target for astrobiology.

Key instruments:
- Mastcam-Z: Stereoscopic zoom camera for panoramic and mineralogical surveys
- SuperCam: Combines LIBS, Raman, visible/infrared spectroscopy, and a microphone
- PIXL (Planetary Instrument for X-ray Lithochemistry): X-ray fluorescence for elemental composition at fine scale
- SHERLOC (Scanning Habitable Environments with Raman & Luminescence for Organics & Chemicals): UV Raman and fluorescence for organics and minerals
- RIMFAX: Ground-penetrating radar for subsurface imaging to 10m depth
- MEDA: Weather station measuring temperature, wind, pressure, humidity, dust
- MOXIE: Oxygen production demo from CO2 atmosphere

The rover has collected and cached over 20 sample tubes for future
Mars Sample Return mission. It has traversed over 25 km exploring
the crater floor, delta front, and Jezero crater rim.
""",
    },
    {
        "title": "Mars Science Laboratory Curiosity Rover",
        "source": "marslab://knowledge/curiosity",
        "text": """
Curiosity rover landed in Gale Crater on August 6, 2012. Gale Crater is
154 km in diameter at 4.6°S, 137.4°E, featuring Mount Sharp (Aeolis Mons),
a 5.5-km tall central mound of layered sedimentary rock.

Key instruments:
- ChemCam: LIBS + remote micro-imager for elemental composition at distance
- CheMin: X-ray diffraction for crystalline mineral identification
- SAM (Sample Analysis at Mars): Mass spectrometer, gas chromatograph, and
  tunable laser spectrometer for organic and volatile analysis
- APXS: Alpha Particle X-ray Spectrometer for elemental abundance
- MAHLI: Mars Hand Lens Imager for close-up geology
- RAD: Radiation Assessment Detector
- DAN: Dynamic Albedo of Neutrons for subsurface hydrogen detection

Key discoveries: ancient habitable environment with fresh water, organic
molecules in mudstones, methane seasonal variations, boron detection,
clay mineral transitions in Mount Sharp's stratigraphy indicating
changing aqueous conditions over geological time.
""",
    },
    {
        "title": "InSight Mars Lander Seismology Mission",
        "source": "marslab://knowledge/insight",
        "text": """
InSight (Interior Exploration using Seismic Investigations, Geodesy and
Heat Transport) operated at Elysium Planitia (4.5°N, 135.6°E) from
November 2018 to December 2022.

Key instrument: SEIS (Seismic Experiment for Interior Structure) —
the first seismometer on Mars, detected over 1,300 marsquakes.

Major findings:
- Mars has a liquid iron core ~1,830 km in radius (larger than expected)
- Crust thickness: 24-72 km (thinner under northern lowlands)
- Multiple seismic discontinuities in the upper mantle
- Marsquakes primarily tectonic (from Cerberus Fossae graben system)
- 2025 discovery: AI analysis revealed impact-generated seismic waves
  travel deeper than models predicted, with a newly identified
  seismic pathway through the upper mantle

In 2025, AI was used to correlate InSight seismic data with MRO imagery,
identifying a fresh impact crater in Cerberus Fossae as the source of
previously unattributed seismic events.
""",
    },
    # ── Instruments ───────────────────────────────────────────────────────
    {
        "title": "CRISM (Compact Reconnaissance Imaging Spectrometer for Mars)",
        "source": "marslab://knowledge/crism",
        "text": """
CRISM is a visible-infrared imaging spectrometer on Mars Reconnaissance
Orbiter (MRO), operating since 2006.

Specifications:
- Spectral range: 362-3920 nm (VNIR + IR)
- Spatial resolution: ~18 m/pixel (FRT mode), ~200 m/pixel (MSP/HSP)
- 544 spectral channels in full-resolution targeted (FRT) mode

CRISM detects minerals through diagnostic absorption features:
- Phyllosilicates (clays): ~1.4, ~1.9, ~2.2-2.3 μm
- Sulfates: ~1.4, ~1.9, ~2.4 μm
- Carbonates: ~2.3, ~2.5, ~3.4 μm
- Olivine: broad ~1 μm absorption
- Pyroxene: ~1 and ~2 μm absorptions
- Iron oxides/hydroxides: ~0.5-0.9 μm

Key mineral parameters (summary products):
- OLINDEX3: Olivine index (Mg-Fe silicate)
- LCPINDEX2/HCPINDEX2: Low/High calcium pyroxene
- D2300: 2.3 μm dropoff (Fe/Mg phyllosilicates)
- BD1900R2: 1.9 μm band depth (hydration)
- BD2500_2: 2.5 μm band depth (carbonates)
- SINDEX2: Sulfate index

CRISM data is archived in NASA PDS and available as TRR3 (targeted
reduced data record version 3) products.
""",
    },
    {
        "title": "HiRISE (High Resolution Imaging Science Experiment)",
        "source": "marslab://knowledge/hirise",
        "text": """
HiRISE on MRO provides the highest resolution orbital imagery of Mars.

Specifications:
- Resolution: 25-50 cm/pixel (highest of any planetary orbiter)
- Swath width: ~6 km at ~300 km altitude
- 14-bit dynamic range, 3 color channels (BG, RED, NIR)
- Stereo pairs enable Digital Terrain Models (DTMs) at ~1 m/post

Key applications:
- Landing site selection and characterization
- Geomorphology: gullies, RSL, dunes, polygonal terrain, ice features
- Change detection: new impact craters, seasonal frost dynamics
- Layer counting in sedimentary deposits for stratigraphy
- Support for rover traverse planning

HiRISE DTMs are produced using ISIS and Ames Stereo Pipeline (ASP)
from stereo pairs. They provide critical elevation data for slope
analysis, volume calculations, and 3D geological interpretation.

The archive contains over 70,000 observations covering ~3% of Mars
at full resolution.
""",
    },
    {
        "title": "SHARAD (SHAllow RADar Sounder)",
        "source": "marslab://knowledge/sharad",
        "text": """
SHARAD is a ground-penetrating radar on MRO operating at 20 MHz
center frequency with 10 MHz bandwidth.

Capabilities:
- Vertical resolution: ~15 m in free space, ~10 m in ice
- Penetration depth: up to 1 km in ice, ~100-200 m in rock
- Along-track resolution: ~300-700 m (after synthetic aperture processing)
- Cross-track footprint: ~3-6 km

SHARAD detects subsurface interfaces where dielectric properties change:
- Ice/regolith boundaries
- Layered deposits within polar caps
- Buried impact structures
- Possible subsurface water ice deposits

Key discoveries:
- Massive subsurface water ice deposits in Arcadia Planitia and
  Utopia Planitia — enough to cover entire planet in ~1.5 m of water
- Internal layering in polar caps recording climate history
- Buried impact craters under northern lowland sediments
- Evidence for volcanic intrusions beneath Elysium Planitia

SWIM (Subsurface Water Ice Mapping) project uses SHARAD data combined
with thermal, neutron, and geomorphic evidence to map ice accessibility
for future human missions.
""",
    },
    # ── Geology & Mineralogy ──────────────────────────────────────────────
    {
        "title": "Mars Mineralogy and Spectral Analysis Guide",
        "source": "marslab://knowledge/mineralogy",
        "text": """
Mars surface mineralogy reveals a history of volcanic, aqueous,
and aeolian processes. Key mineral groups:

MAFIC MINERALS (volcanic primary):
- Olivine (Mg,Fe)2SiO4: Common in ancient Noachian terrains. CRISM
  detects via broad ~1 μm absorption. Indicates unweathered basalt.
- Pyroxene (Ca,Mg,Fe)Si2O6: Dominant on Mars. Low-calcium (LCP) and
  high-calcium (HCP) varieties map distinct volcanic compositions.

PHYLLOSILICATES (aqueous alteration):
- Fe/Mg smectites (nontronite, saponite): Most common clays on Mars.
  2.3 μm absorption. Indicate near-neutral pH water interaction.
- Al-phyllosilicates (kaolinite, montmorillonite): Less common.
  2.2 μm absorption. May indicate more acidic or leaching conditions.

SULFATES (evaporite/acidic):
- Monohydrated (kieserite MgSO4·H2O): Detected in Valles Marineris
  layered deposits and Meridiani Planum.
- Polyhydrated sulfates: Broader 1.9 μm absorption.
- Jarosite KFe3(SO4)2(OH)6: Indicates acidic conditions (pH <4).

CARBONATES:
- Mg-carbonates detected in limited locations (Nili Fossae, Jezero).
  Indicate alkaline water chemistry. Important for carbon cycle studies.

IRON OXIDES:
- Hematite (Fe2O3): Both crystalline (Meridiani) and nanophase (dust).
- Goethite (FeOOH): Possible in some altered terrains.
""",
    },
    {
        "title": "Mars Geological Timeline and Epochs",
        "source": "marslab://knowledge/geology_timeline",
        "text": """
Mars geological history is divided into three major periods:

PRE-NOACHIAN (>4.1 Ga):
- Formation of Mars, late heavy bombardment
- Possible global magma ocean
- Formation of crustal dichotomy (northern lowlands vs southern highlands)
- Giant impact basins: Hellas, Argyre, Isidis

NOACHIAN (4.1-3.7 Ga):
- Heavy cratering, widespread volcanic activity
- Extensive aqueous alteration → phyllosilicate formation
- Valley networks suggest warm/wet episodes or precipitation
- Jezero delta formation during this period
- Magnetic field likely still active early Noachian

HESPERIAN (3.7-3.0 Ga):
- Catastrophic outflow channels (e.g., Kasei Valles)
- Sulfate deposit formation → shift to acidic surface conditions
- Tharsis volcanism peak
- Transition from clay-forming to sulfate-forming environment
- "Great drying" — loss of standing surface water

AMAZONIAN (3.0 Ga - present):
- Cold, dry, hyperarid conditions
- Ongoing minor volcanism (youngest Olympus Mons flows ~2 Ma)
- Periglacial features, glacial deposits at mid-latitudes
- Seasonal CO2 frost cycle, active aeolian processes
- Recurring Slope Lineae (RSL) — debated water involvement
""",
    },
    # ── Climate & Atmosphere ──────────────────────────────────────────────
    {
        "title": "Mars Atmosphere and Climate System",
        "source": "marslab://knowledge/climate",
        "text": """
Mars has a thin CO2-dominated atmosphere (~95.3% CO2, ~2.7% N2, ~1.6% Ar).

Mean surface pressure: ~636 Pa (0.6% of Earth's)
Temperature range: ~130-300 K depending on latitude, season, and time
CO2 frost point: ~148 K at surface pressure

Seasonal CO2 cycle: ~25% of atmospheric mass condenses at polar caps
during winter, creating a global pressure oscillation.

Dust is the primary atmospheric driver on Mars:
- Background τ ~0.2-0.5 (optical depth)
- Regional storms: τ up to ~4
- Planet-encircling dust events (PEDE): τ > 6 (2001, 2007, 2018)
  The 2018 PEDE ended the Opportunity rover mission.

Water vapor column: ~10-70 precipitable microns, seasonally variable.
Aphelion cloud belt (Ls 50-150): water ice clouds at ~10-30 km altitude.

Key atmospheric science questions:
- Mechanism for global dust storm initiation
- Methane detection controversy (Curiosity vs TGO discrepancy)
- Historical atmospheric loss rates (MAVEN measurements)
- Hydrogen escape and D/H ratio evolution
""",
    },
    {
        "title": "Mars Subsurface Water Ice Distribution",
        "source": "marslab://knowledge/water_ice",
        "text": """
Subsurface water ice is a critical resource for future Mars exploration
and a key scientific target for understanding Mars climate history.

Detection methods:
- SHARAD/MARSIS radar: Direct detection of dielectric interfaces
- Neutron spectroscopy: Hydrogen detection (proxy for water) in top ~1 m
- Thermal inertia: High TI suggests ice-cemented ground
- Geomorphology: Scalloped terrain, polygonal ground, lobate debris aprons
- Fresh impact craters: Expose bright ice at mid-latitudes

SWIM (Subsurface Water Ice Mapping) project integrates:
- Neutron data (N) — hydrogen abundance
- Thermal data (T) — thermal inertia consistency
- Radar data (RD/RS) — SHARAD dielectric detections
- Geomorphic data (G) — surface features indicating ice

Key ice deposits:
- Arcadia Planitia (40-55°N): Extensive shallow ice, SWIM scores >0.7
  Possible excess ice (pore-filling + excess) within 1-10 m depth
- Utopia Planitia: SHARAD-detected massive ice deposit, possibly
  remnant of ancient ocean/ice sheet
- Deuteronilus Mensae: Lobate debris aprons confirmed as debris-covered
  glaciers by SHARAD
- Polar layered deposits: >3 km thick, ~4 × 10⁶ km³ of water ice

Human exploration implications:
- Ice at <1 m depth accessible for ISRU
- Arcadia Planitia and Deuteronilus Mensae are prime ISRU targets
- Minimum ice requirement: ~1 m³ per person per year for propellant/life support
""",
    },
    # ── Astrobiology ──────────────────────────────────────────────────────
    {
        "title": "Mars Biosignature Detection and Astrobiology",
        "source": "marslab://knowledge/astrobiology",
        "text": """
Mars biosignature detection is a primary goal of Mars 2020 and
potential future Mars Sample Return missions.

Types of biosignatures sought:
1. Morphological: Stromatolite-like structures, microfossils
2. Chemical: Complex organic molecules, specific isotope ratios,
   molecular weight distributions (e.g., fatty acids with even-carbon preference)
3. Mineralogical: Biogenic mineral precipitation patterns
4. Contextual: Preservation environments (lacustrine sediments, hydrothermal)

Perseverance detection capabilities:
- SHERLOC: Deep UV Raman (248.6 nm) detects aromatic organics and
  ring-containing compounds. Maps organic distribution at ~100 μm scale.
- PIXL: X-ray fluorescence at ~120 μm scale for elemental chemistry.
  Can detect bio-relevant elements (P, S, Mn) and textural patterns.
- SuperCam: Remote LIBS + Raman at distance for initial screening.

Key challenges:
- Radiation damage destroys surface organics (top ~5 cm exposed to GCR)
- Perchlorate oxidants in regolith can destroy organics during heating
- Abiotic processes can mimic some biosignature patterns
- Extraordinary claims require extraordinary evidence

Promising Jezero targets:
- Delta sediments: Fine-grained lacustrine facies with highest preservation potential
- Carbonate-bearing units: Jezero margin carbonates (Séítah formation)
- Igneous floor units: Olivine-rich cumulate rocks with potential hydrothermal alteration
""",
    },
    # ── AI on Mars ────────────────────────────────────────────────────────
    {
        "title": "AI and Machine Learning Applications in Mars Exploration",
        "source": "marslab://knowledge/ai_mars",
        "text": """
AI/ML is increasingly integrated into Mars exploration operations and science.

AUTONOMOUS NAVIGATION:
- AutoNav: Perseverance's autonomous driving system, processes stereo
  images to build 3D terrain maps and plan paths avoiding hazards.
  Can drive 120+ m/sol without human intervention.
- 2025 milestone: First AI-planned drive where a vision-capable AI
  analyzed terrain imagery and autonomously planned the complete route
  (Dec 10, 2025, along Jezero crater rim).

SCIENCE AUTONOMY:
- AEGIS (Autonomous Exploration for Gathering Increased Science):
  Originally on Curiosity, ported to Perseverance. Identifies interesting
  rock targets and autonomously commands ChemCam/SuperCam observations.
- PIXL AI: Automated mineral identification from X-ray fluorescence
  spectra, enabling rapid rock characterization (JPL, 2024).

CRATER DETECTION:
- AI-assisted fresh crater detection in HiRISE/CTX temporal pairs
  (NASA, 2024). Discovered 40+ new impact sites that human analysts missed.
- InSight-MRO AI correlation (2025): AI linked marsquake seismic data
  with orbital imagery to pinpoint a fresh impact crater.

MARS-BENCH (NeurIPS 2025):
- First systematic benchmark for evaluating foundation models on Mars
  science tasks. 20 datasets covering classification, segmentation,
  and object detection of geological features.
- Key finding: Mars-domain pre-training may outperform general vision models.

NASA FAIMM (2025-2026):
- Foundational AI for the Moon and Mars program, applying Foundation
  Models to planetary science and exploration tasks.
""",
    },
]


def seed_knowledge(collection: str = _COLLECTION, force: bool = False) -> Dict:
    """
    Seed the vector store with built-in Mars science knowledge.

    Parameters
    ----------
    collection : str
        Target collection name.
    force : bool
        Re-ingest even if already seeded.

    Returns
    -------
    Dict with seeding statistics.
    """
    global _SEEDED

    if _SEEDED and not force:
        logger.info("Knowledge already seeded, skipping")
        return {"status": "already_seeded", "documents": 0, "chunks": 0}

    total_chunks = 0
    results = []

    for doc in MARS_KNOWLEDGE:
        result = ingest_text(
            text=doc["text"],
            source=doc["source"],
            title=doc["title"],
            collection=collection,
            chunk_size=512,
            chunk_overlap=64,
            metadata={"type": "built_in_knowledge"},
        )
        results.append(result)
        total_chunks += result.get("chunks", 0)

    _SEEDED = True

    summary = {
        "status": "ok",
        "documents": len(MARS_KNOWLEDGE),
        "total_chunks": total_chunks,
        "collection": collection,
    }
    logger.info(f"Seeded {len(MARS_KNOWLEDGE)} Mars knowledge documents → {total_chunks} chunks")
    return summary


def auto_ingest_local_knowledge(collection: str = _COLLECTION) -> Dict:
    """
    Auto-ingest documents from MarsLab's knowledge/ and mars_research/ dirs.
    """
    from .ingestion import ingest_directory

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs_to_scan = [
        os.path.join(base, "..", "knowledge"),
        os.path.join(base, "mars_research"),
        os.path.join(base, "agent_reports"),
    ]

    total_files = 0
    total_chunks = 0

    for d in dirs_to_scan:
        d = os.path.normpath(d)
        if not os.path.isdir(d):
            continue

        result = ingest_directory(
            dir_path=d,
            collection=collection,
            extensions=[".txt", ".md", ".json"],
        )
        total_files += result.get("files_processed", 0)
        total_chunks += result.get("total_chunks", 0)

    return {
        "status": "ok",
        "directories_scanned": len(dirs_to_scan),
        "files_processed": total_files,
        "total_chunks": total_chunks,
    }
