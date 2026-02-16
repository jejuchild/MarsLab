#!/usr/bin/env bash
#
# generate_custom_dtms.sh — End-to-end HiRISE stereo DTM generation pipeline
#
# Installs ASP 3.6.0 + ISIS 9.0.0 via conda, finds stereo pairs in a given
# bounding box (default: Arcadia Planitia), downloads EDR images, and processes
# the top N pairs through ASP to generate custom DTMs.
#
# Usage:
#   ./generate_custom_dtms.sh                          # Arcadia Planitia, top 3
#   ./generate_custom_dtms.sh --top 5                  # top 5 pairs
#   ./generate_custom_dtms.sh --bbox -170 38 -160 42   # custom bbox
#   ./generate_custom_dtms.sh --skip-install            # skip conda/ASP setup
#   ./generate_custom_dtms.sh --phase 5                 # resume from phase 5
#
set -eo pipefail

# ── Configuration ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$BACKEND_DIR")"
WORK_DIR="${BACKEND_DIR}/asp_workspace"
CONDA_DIR="${HOME}/miniconda3"
ASP_ENV="asp"
ISISDATA_DIR="${HOME}/isisdata"
ASP_STANDALONE="${HOME}/StereoPipeline-3.5.0-2025-04-28-x86_64-Linux"

# Arcadia Planitia bbox (IAU Gazetteer, -180/180 longitude)
DEFAULT_LON_MIN=165.86
DEFAULT_LAT_MIN=33.87
DEFAULT_LON_MAX=-149.57
DEFAULT_LAT_MAX=64.17

TOP_N=3
SKIP_INSTALL=0
START_PHASE=1
NUM_CORES=$(( $(nproc) - 4 ))  # Leave 4 cores for system
[ "$NUM_CORES" -lt 4 ] && NUM_CORES=4

CHECKPOINT_FILE="${WORK_DIR}/.checkpoint"

# ── Parse arguments ──────────────────────────────────────
LON_MIN=$DEFAULT_LON_MIN
LAT_MIN=$DEFAULT_LAT_MIN
LON_MAX=$DEFAULT_LON_MAX
LAT_MAX=$DEFAULT_LAT_MAX

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top)       TOP_N="$2";        shift 2 ;;
        --bbox)      LON_MIN="$2"; LAT_MIN="$3"; LON_MAX="$4"; LAT_MAX="$5"; shift 5 ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --phase)     START_PHASE="$2";  shift 2 ;;
        --cores)     NUM_CORES="$2";    shift 2 ;;
        --work-dir)  WORK_DIR="$2";     shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--top N] [--bbox LON_MIN LAT_MIN LON_MAX LAT_MAX] [--skip-install] [--phase N]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[$(date +%H:%M:%S)] $*\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
err()  { echo -e "\033[1;31m  ✗ $*\033[0m" >&2; }
warn() { echo -e "\033[1;33m  ⚠ $*\033[0m"; }

save_checkpoint() { echo "$1" > "$CHECKPOINT_FILE"; }
get_checkpoint()  { [[ -f "$CHECKPOINT_FILE" ]] && cat "$CHECKPOINT_FILE" || echo "0"; }

disk_free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }

# ── Phase 1: Install Miniconda ───────────────────────────
phase_1_conda() {
    log "PHASE 1: Install Miniconda"

    if [[ -f "${CONDA_DIR}/bin/conda" ]]; then
        ok "Miniconda already installed at ${CONDA_DIR}"
        return 0
    fi

    local free_gb
    free_gb=$(disk_free_gb "$HOME")
    if [[ "$free_gb" -lt 30 ]]; then
        err "Need at least 30 GB free disk space. Only ${free_gb} GB available."
        exit 1
    fi

    log "Downloading Miniconda installer..."
    local installer="/tmp/miniconda_installer.sh"
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O "$installer"

    log "Installing Miniconda to ${CONDA_DIR}..."
    bash "$installer" -b -p "$CONDA_DIR"
    rm -f "$installer"

    # Initialize conda for this session
    eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
    ok "Miniconda installed"
}


# ── Phase 2: Install ASP + ISIS ──────────────────────────
phase_2_asp() {
    log "PHASE 2: Install ASP 3.6.0 + ISIS 9.0.0"

    eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"

    # Check if environment already exists and has ASP
    if conda env list | grep -q "^${ASP_ENV} "; then
        conda activate "$ASP_ENV"
        if command -v parallel_stereo &>/dev/null; then
            ok "ASP environment '${ASP_ENV}' already exists with ASP installed"
            return 0
        fi
        warn "Environment exists but ASP not found, reinstalling..."
        conda deactivate
        conda env remove -n "$ASP_ENV" -y
    fi

    log "Creating conda environment '${ASP_ENV}' with ASP 3.6.0..."
    conda config --set channel_priority flexible
    conda create -n "$ASP_ENV" -y \
        -c nasa-ames-stereo-pipeline \
        -c usgs-astrogeology \
        -c conda-forge \
        -c defaults \
        stereo-pipeline=3.6.0

    conda activate "$ASP_ENV"

    # Verify installation
    if ! command -v parallel_stereo &>/dev/null; then
        err "ASP installation failed: parallel_stereo not found"
        exit 1
    fi

    local asp_version
    asp_version=$(parallel_stereo --version 2>&1 | head -1 || echo "unknown")
    ok "ASP installed: ${asp_version}"

    # Install GDAL Python bindings for asp_index_updater.py
    pip install GDAL 2>/dev/null || warn "GDAL pip install failed (may already be included)"
    ok "ASP + ISIS environment ready"
}


# ── Phase 3: Download ISISDATA ───────────────────────────
phase_3_isisdata() {
    log "PHASE 3: Download ISISDATA (base + MRO kernels)"

    eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
    conda activate "$ASP_ENV"

    export ISISDATA="$ISISDATA_DIR"
    export ISISROOT="$CONDA_PREFIX"
    mkdir -p "$ISISDATA"

    # Check if data already downloaded
    if [[ -d "${ISISDATA}/base" && -d "${ISISDATA}/mro" ]]; then
        local base_size mro_size
        base_size=$(du -sm "${ISISDATA}/base" 2>/dev/null | cut -f1 || echo "0")
        mro_size=$(du -sm "${ISISDATA}/mro" 2>/dev/null | cut -f1 || echo "0")
        if [[ "$base_size" -gt 10 && "$mro_size" -gt 10 ]]; then
            ok "ISISDATA already present (base: ${base_size}MB, mro: ${mro_size}MB)"
            return 0
        fi
    fi

    # Download only calibration files (no SPICE kernels — we use the web service)
    # This saves 10-20 GB of disk space vs full kernel download
    log "Downloading base calibration data (no kernels)..."
    downloadIsisData base "$ISISDATA" --no-kernels
    ok "Base calibration data downloaded"

    log "Downloading MRO calibration data (no kernels)..."
    downloadIsisData mro "$ISISDATA" --no-kernels
    ok "MRO calibration data downloaded"

    local total_size
    total_size=$(du -sh "$ISISDATA" 2>/dev/null | cut -f1 || echo "?")
    ok "ISISDATA ready: ${total_size} at ${ISISDATA} (SPICE kernels via web service)"
}


# ── Phase 4: Find stereo pairs ──────────────────────────
phase_4_find_pairs() {
    log "PHASE 4: Find HiRISE stereo pairs"
    log "  Bbox: lon [${LON_MIN}, ${LON_MAX}], lat [${LAT_MIN}, ${LAT_MAX}]"
    log "  Top N: ${TOP_N}"

    mkdir -p "$WORK_DIR"

    local pairs_json="${WORK_DIR}/stereo_pairs.json"

    # Split large bbox into ~10°x20° chunks to avoid ODE query timeout
    # Then merge results and pick the top N
    cd "$PROJECT_DIR"
    python3 << 'PYEOF'
import asyncio, json, os, sys
sys.path.insert(0, ".")
from backend.scripts.find_stereo_pairs import (
    query_hirise_in_bbox, find_stereo_pairs,
    load_existing_dtm_pairs, annotate_existing_dtms,
    save_results_json,
)
import aiohttp

lon_min = float(os.environ["LON_MIN"])
lat_min = float(os.environ["LAT_MIN"])
lon_max = float(os.environ["LON_MAX"])
lat_max = float(os.environ["LAT_MAX"])
top_n = int(os.environ["TOP_N"])
work_dir = os.environ["WORK_DIR"]

# Handle antimeridian wrapping: convert to 0-360 for chunking
lon_min_360 = lon_min % 360
lon_max_360 = lon_max % 360
if lon_min_360 > lon_max_360:
    # Wraps antimeridian — treat as a contiguous 0-360 range
    total_lon_span = (360 - lon_min_360) + lon_max_360
else:
    total_lon_span = lon_max_360 - lon_min_360

# Chunk params
LAT_CHUNK = 10.0
LON_CHUNK = 20.0

async def main():
    all_images = {}  # obs_id -> HiRISEImage (dedup)

    async with aiohttp.ClientSession() as session:
        lat = lat_min
        while lat < lat_max:
            chunk_lat_max = min(lat + LAT_CHUNK, lat_max)

            # Walk longitude in 0-360 space
            lon_360 = lon_min_360
            remaining = total_lon_span
            while remaining > 0:
                chunk_lon_span = min(LON_CHUNK, remaining)
                chunk_lon_max_360 = lon_360 + chunk_lon_span

                # Convert back to -180/180 for the bbox
                c_lon_min = lon_360 - 360 if lon_360 > 180 else lon_360
                c_lon_max = chunk_lon_max_360 - 360 if chunk_lon_max_360 > 180 else chunk_lon_max_360

                bbox = {
                    "lat_min": lat, "lat_max": chunk_lat_max,
                    "lon_min": c_lon_min, "lon_max": c_lon_max,
                }
                print(f"  Chunk: lat [{lat:.1f}, {chunk_lat_max:.1f}], "
                      f"lon [{c_lon_min:.1f}, {c_lon_max:.1f}]", flush=True)

                images = await query_hirise_in_bbox(bbox, session, max_results=500)
                for img in images:
                    all_images[img.observation_id] = img

                lon_360 += chunk_lon_span
                remaining -= chunk_lon_span
                await asyncio.sleep(0.5)

            lat += LAT_CHUNK

    image_list = list(all_images.values())
    print(f"\n  Total unique HiRISE images: {len(image_list)}", flush=True)

    if len(image_list) < 2:
        print("  Not enough images for stereo pairs", flush=True)
        # Write empty result
        save_results_json([], f"{work_dir}/stereo_pairs.json",
                          {"mode": "bbox_chunked", "total_images": len(image_list)})
        return

    pairs = find_stereo_pairs(image_list, min_overlap=0.3)
    dtm_pairs = load_existing_dtm_pairs()
    annotate_existing_dtms(pairs, dtm_pairs)

    new_count = sum(1 for p in pairs if p.existing_dtm is None)
    print(f"  Found {len(pairs)} stereo pairs ({new_count} new)", flush=True)

    save_results_json(pairs, f"{work_dir}/stereo_pairs.json",
                      {"mode": "bbox_chunked", "total_images": len(image_list)})

asyncio.run(main())
PYEOF

    if [[ ! -f "$pairs_json" ]]; then
        err "Stereo pair search failed — no output JSON"
        exit 1
    fi

    local pair_count
    pair_count=$(python3 -c "
import json, sys
with open('${pairs_json}') as f:
    data = json.load(f)
new = [p for p in data['pairs'] if p['existing_dtm'] is None]
print(len(new[:${TOP_N}]))
")
    ok "Found ${pair_count} new stereo pair candidates"

    if [[ "$pair_count" -eq 0 ]]; then
        warn "No new stereo pairs found. All candidates already have DTMs."
        exit 0
    fi
}


# ── Phase 5: Download EDR images ─────────────────────────
phase_5_download() {
    log "PHASE 5: Download HiRISE EDR images"

    eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
    conda activate "$ASP_ENV" 2>/dev/null || true

    local pairs_json="${WORK_DIR}/stereo_pairs.json"
    if [[ ! -f "$pairs_json" ]]; then
        err "No stereo_pairs.json found. Run phase 4 first."
        exit 1
    fi

    # Extract top N new pairs and download EDRs
    python3 << 'PYEOF'
import json, subprocess, sys, os
from pathlib import Path

work_dir = os.environ.get("WORK_DIR", "backend/asp_workspace")
top_n = int(os.environ.get("TOP_N", "3"))

with open(f"{work_dir}/stereo_pairs.json") as f:
    data = json.load(f)

new_pairs = [p for p in data["pairs"] if p["existing_dtm"] is None][:top_n]

if not new_pairs:
    print("  No new pairs to download.")
    sys.exit(0)

# Build download manifest
urls = []
for pair in new_pairs:
    for img_key in ("image_a", "image_b"):
        obs_id = pair[img_key]
        # Parse orbit folder: ESP_060706_2195 → ORB_060700_060799
        parts = obs_id.split("_")
        if len(parts) >= 2:
            orbit = int(parts[1])
            orb_start = (orbit // 100) * 100
            orb_end = orb_start + 99
            orb_folder = f"ORB_{orb_start:06d}_{orb_end:06d}"
        else:
            continue

        # URL needs prefix subdir: PDS/EDR/ESP/ORB_.../ESP_.../
        prefix = parts[0]  # ESP, PSP, or TRA
        base_url = f"https://hirise-pds.lpl.arizona.edu/PDS/EDR/{prefix}/{orb_folder}/{obs_id}/"
        obs_dir = Path(work_dir) / "edr" / obs_id
        obs_dir.mkdir(parents=True, exist_ok=True)

        urls.append((base_url, str(obs_dir), obs_id))

# Download each observation's RED EDR files
for base_url, dest_dir, obs_id in urls:
    dest_path = Path(dest_dir)
    # Check if already downloaded
    existing = list(dest_path.glob("*RED*.IMG"))
    if len(existing) >= 8:
        print(f"  {obs_id}: Already have {len(existing)} RED EDR files, skipping")
        continue

    print(f"  Downloading {obs_id} RED EDRs...")
    # Use wget to get RED channel EDR files
    cmd = [
        "wget", "-r", "-np", "-nd", "-q",
        "--accept-regex", r".*RED.*\.IMG$",
        "-P", dest_dir,
        base_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    downloaded = list(dest_path.glob("*RED*.IMG"))
    print(f"  {obs_id}: Downloaded {len(downloaded)} RED EDR files")

# Save selected pairs for processing
selected = []
for pair in new_pairs:
    selected.append({
        "obs_a": pair["image_a"],
        "obs_b": pair["image_b"],
        "overlap_bbox": pair["overlap_bbox"],
        "emission_diff": pair["emission_diff"],
        "composite_score": pair["composite_score"],
    })
with open(f"{work_dir}/selected_pairs.json", "w") as f:
    json.dump(selected, f, indent=2)
print(f"\n  Saved {len(selected)} pairs to selected_pairs.json")
PYEOF

    ok "EDR downloads complete"
}


# ── Phase 6: ASP Processing ─────────────────────────────
phase_6_asp_process() {
    log "PHASE 6: ASP Stereo Processing (${NUM_CORES} cores)"

    # Set up hybrid PATH: standalone ASP (working C++ binaries) + conda ISIS tools
    # Conda ASP 3.6.0 C++ binaries segfault, so we use standalone ASP 3.5.0 for
    # stereo_parse, parallel_stereo, point2dem, cam2map4stereo.py, hiedr2mosaic.py
    # ISIS tools (hi2isis, hical, etc.) come from the conda env
    export PATH="${ASP_STANDALONE}/bin:${CONDA_DIR}/envs/${ASP_ENV}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    export ISISDATA="$ISISDATA_DIR"
    export ISISROOT="${CONDA_DIR}/envs/${ASP_ENV}"
    # ISIS/conda shared libs
    export LD_LIBRARY_PATH="${CONDA_DIR}/envs/${ASP_ENV}/lib:${LD_LIBRARY_PATH:-}"

    # Verify the hybrid setup works
    log "  Verifying ASP + ISIS setup..."
    local asp_ver
    asp_ver=$(stereo_parse --version 2>&1 | head -1) || { err "stereo_parse not working"; exit 1; }
    log "  ASP: ${asp_ver}"
    which hi2isis >/dev/null 2>&1 || { err "hi2isis not found in PATH"; exit 1; }
    ok "ASP (standalone) + ISIS (conda) ready"

    local selected="${WORK_DIR}/selected_pairs.json"
    if [[ ! -f "$selected" ]]; then
        err "No selected_pairs.json found. Run phase 5 first."
        exit 1
    fi

    local pair_count
    pair_count=$(python3 -c "import json; print(len(json.load(open('${selected}'))))")

    for i in $(seq 0 $((pair_count - 1))); do
        log "Processing pair $((i + 1)) of ${pair_count}..."

        # Extract pair info
        local obs_a obs_b
        obs_a=$(python3 -c "import json; print(json.load(open('${selected}'))[$i]['obs_a'])")
        obs_b=$(python3 -c "import json; print(json.load(open('${selected}'))[$i]['obs_b'])")

        log "  Pair: ${obs_a} + ${obs_b}"

        local edr_a="${WORK_DIR}/edr/${obs_a}"
        local edr_b="${WORK_DIR}/edr/${obs_b}"
        local proc_dir="${WORK_DIR}/processing/${obs_a}_${obs_b}"
        local out_dir="${WORK_DIR}/output/${obs_a}_${obs_b}"
        mkdir -p "$proc_dir" "$out_dir"

        # Check if already processed
        if [[ -f "${out_dir}/run-DEM.tif" ]]; then
            ok "Already processed: ${out_dir}/run-DEM.tif"
            continue
        fi

        # Verify EDR files exist
        local red_a red_b
        red_a=$(ls "${edr_a}/"*RED*.IMG 2>/dev/null | wc -l)
        red_b=$(ls "${edr_b}/"*RED*.IMG 2>/dev/null | wc -l)
        if [[ "$red_a" -lt 1 || "$red_b" -lt 1 ]]; then
            err "Missing EDR files: ${obs_a} has ${red_a}, ${obs_b} has ${red_b} RED files"
            continue
        fi

        cd "$proc_dir"

        # Disable errexit for the processing loop — we handle errors per-pair
        set +e

        # Step 1: Stitch RED CCDs into mosaics using hiedr2mosaic.py
        # Usage: hiedr2mosaic.py [--web] [-t threads] EDR.IMG-files
        # Output: {common_prefix}.mos_hijitreged.norm.cub (where prefix = e.g. ESP_036165_2240_RED)
        # --web uses SPICE web service instead of local kernels
        log "  Step 1/4: Stitching CCD mosaics (with SPICE web)..."

        log "    Processing ${obs_a}..."
        hiedr2mosaic.py --web -t "${NUM_CORES}" "${edr_a}/"*RED*.IMG 2>&1 | tail -5
        local cub_a="${obs_a}_RED.mos_hijitreged.norm.cub"

        log "    Processing ${obs_b}..."
        hiedr2mosaic.py --web -t "${NUM_CORES}" "${edr_b}/"*RED*.IMG 2>&1 | tail -5
        local cub_b="${obs_b}_RED.mos_hijitreged.norm.cub"

        if [[ ! -f "$cub_a" || ! -f "$cub_b" ]]; then
            err "hiedr2mosaic failed for pair ${obs_a}+${obs_b}"
            # List what was actually created for debugging
            ls -la *.norm.cub 2>/dev/null || echo "  No .norm.cub files found"
            set -e
            continue
        fi
        ok "Mosaics created: ${cub_a}, ${cub_b}"

        # Step 2: Map-project for better stereo matching
        # cam2map4stereo.py appends '.map' suffix → e.g. ESP_036165_2240_RED.mos_hijitreged.norm.map.cub
        log "  Step 2/4: Map-projecting images..."
        cam2map4stereo.py "$cub_a" "$cub_b" 2>&1 | tail -5

        # cam2map4stereo.py may use a shorter output name (e.g. {obs}_RED.map.cub)
        # rather than preserving the full .mos_hijitreged.norm suffix, so glob for it
        local map_a map_b
        map_a=$(ls "${obs_a}"*RED*.map.cub 2>/dev/null | head -1)
        map_b=$(ls "${obs_b}"*RED*.map.cub 2>/dev/null | head -1)

        if [[ -z "$map_a" || -z "$map_b" || ! -f "$map_a" || ! -f "$map_b" ]]; then
            err "cam2map4stereo failed"
            ls -la *.map.cub 2>/dev/null || echo "  No .map.cub files found"
            set -e
            continue
        fi
        ok "Map-projection complete: $(basename "$map_a"), $(basename "$map_b")"

        # Step 3: Stereo correlation
        log "  Step 3/4: Running stereo correlation (${NUM_CORES} cores)..."
        mkdir -p "$out_dir"
        # Limit processes to avoid OOM: asp_mgm uses ~7GB per tile
        local stereo_procs=$(( $(grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}') / 8 ))
        [ "$stereo_procs" -gt "$NUM_CORES" ] && stereo_procs="$NUM_CORES"
        [ "$stereo_procs" -lt 2 ] && stereo_procs=2
        log "  Using ${stereo_procs} parallel processes (memory-safe)"

        parallel_stereo \
            "$map_a" "$map_b" \
            "${out_dir}/run" \
            --stereo-algorithm asp_mgm \
            --subpixel-mode 3 \
            --alignment-method none \
            --processes "$stereo_procs" \
            2>&1 | tail -10

        if [[ ! -f "${out_dir}/run-PC.tif" ]]; then
            err "parallel_stereo failed for pair ${obs_a}+${obs_b}"
            set -e
            continue
        fi
        ok "Stereo correlation complete"

        # Step 4: Generate DEM from point cloud
        log "  Step 4/4: Generating DEM..."
        point2dem "${out_dir}/run-PC.tif" \
            -o "${out_dir}/run" \
            --dem-spacing 1.0 \
            --reference-spheroid mars \
            --dem-hole-fill-len 50 \
            --nodata-value -32767 \
            2>&1 | tail -5

        if [[ -f "${out_dir}/run-DEM.tif" ]]; then
            ok "DTM generated: ${out_dir}/run-DEM.tif"

            # Generate orthoimage (optional, non-critical)
            mapproject "${out_dir}/run-DEM.tif" "$map_a" "${out_dir}/${obs_a}_ortho.tif" \
                2>&1 | tail -2 || warn "Ortho generation failed (non-critical)"
        else
            err "point2dem failed for pair ${obs_a}+${obs_b}"
            set -e
            continue
        fi

        # Re-enable errexit
        set -e

        # Cleanup intermediate files to save disk space
        log "  Cleaning up intermediate files for pair $((i + 1))..."
        rm -f "${proc_dir}/"*.cub
        rm -f "${out_dir}/run-PC.tif"  # Point cloud is huge (~15 GB)
        rm -f "${out_dir}/run-F.tif" "${out_dir}/run-L.tif" "${out_dir}/run-D.tif" "${out_dir}/run-RD.tif"

        local dem_size
        dem_size=$(du -sh "${out_dir}/run-DEM.tif" 2>/dev/null | cut -f1 || echo "?")
        ok "Pair $((i + 1)) done: ${dem_size}"
    done

    ok "All pairs processed"
}


# ── Phase 7: Index Results ───────────────────────────────
phase_7_index() {
    log "PHASE 7: Index custom DTMs"

    eval "$("${CONDA_DIR}/bin/conda" shell.bash hook)"
    conda activate "$ASP_ENV" 2>/dev/null || true
    export ISISDATA="$ISISDATA_DIR"

    local selected="${WORK_DIR}/selected_pairs.json"
    local pair_count
    pair_count=$(python3 -c "import json; print(len(json.load(open('${selected}'))))")

    local indexed=0

    for i in $(seq 0 $((pair_count - 1))); do
        local obs_a obs_b
        obs_a=$(python3 -c "import json; print(json.load(open('${selected}'))[$i]['obs_a'])")
        obs_b=$(python3 -c "import json; print(json.load(open('${selected}'))[$i]['obs_b'])")

        local dem="${WORK_DIR}/output/${obs_a}_${obs_b}/run-DEM.tif"
        local ortho="${WORK_DIR}/output/${obs_a}_${obs_b}/${obs_a}_ortho.tif"

        if [[ ! -f "$dem" ]]; then
            warn "No DEM for ${obs_a}+${obs_b}, skipping index"
            continue
        fi

        log "  Indexing ${obs_a} + ${obs_b}..."

        local ortho_arg=""
        if [[ -f "$ortho" ]]; then
            ortho_arg="--ortho ${ortho}"
        fi

        cd "$PROJECT_DIR"
        python -m backend.scripts.asp_index_updater \
            --dtm "$dem" \
            --obs-a "$obs_a" \
            --obs-b "$obs_b" \
            $ortho_arg

        indexed=$((indexed + 1))
    done

    ok "Indexed ${indexed} custom DTMs"
}


# ── Main ─────────────────────────────────────────────────
main() {
    log "HiRISE Custom DTM Generation Pipeline"
    log "  Work directory: ${WORK_DIR}"
    log "  Bbox: lon [${LON_MIN}, ${LON_MAX}], lat [${LAT_MIN}, ${LAT_MAX}]"
    log "  Top pairs: ${TOP_N}"
    log "  Cores: ${NUM_CORES}"

    mkdir -p "$WORK_DIR"

    local free_gb
    free_gb=$(disk_free_gb "$WORK_DIR")
    log "  Disk free: ${free_gb} GB"

    if [[ "$free_gb" -lt 30 ]]; then
        err "Insufficient disk space: ${free_gb} GB free, need at least 30 GB"
        exit 1
    fi

    # Run phases
    if [[ $SKIP_INSTALL -eq 0 && $START_PHASE -le 1 ]]; then
        phase_1_conda
        save_checkpoint 1
    fi

    if [[ $SKIP_INSTALL -eq 0 && $START_PHASE -le 2 ]]; then
        phase_2_asp
        save_checkpoint 2
    fi

    if [[ $SKIP_INSTALL -eq 0 && $START_PHASE -le 3 ]]; then
        phase_3_isisdata
        save_checkpoint 3
    fi

    if [[ $START_PHASE -le 4 ]]; then
        phase_4_find_pairs
        save_checkpoint 4
    fi

    if [[ $START_PHASE -le 5 ]]; then
        phase_5_download
        save_checkpoint 5
    fi

    if [[ $START_PHASE -le 6 ]]; then
        phase_6_asp_process
        save_checkpoint 6
    fi

    if [[ $START_PHASE -le 7 ]]; then
        phase_7_index
        save_checkpoint 7
    fi

    log "Pipeline complete!"
    ok "Custom DTMs generated and indexed."
    ok "Restart the MarsLab backend to see new DTMs on the map."
}

# Export env vars for child processes
export WORK_DIR TOP_N LON_MIN LAT_MIN LON_MAX LAT_MAX
main
