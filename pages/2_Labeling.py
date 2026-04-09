"""Mastcam-Z Labeling - Co-registered imagery via PDS XYZ + SPICE.

Uses SPICE-transformed lon/lat coordinates with RAS texture for labeling.
DTM overlay is available separately for alignment verification.
"""

import json
import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Labeling - MarsLab", layout="wide")
st.title("Mastcam-Z Labeling")

OUTPUT_DIR = Path("/disk1/cspark/mastcam/coregister_data/output")
PDS_CACHE = Path("/disk1/cspark/mastcam/coregister_data/pds_cache/mastcamz")
HIRISE_DIR = Path("/disk1/cspark/mastcam/coregister_data/pds_cache/hirise")
RESULTS_PATH = OUTPUT_DIR / "batch_results.json"


@st.cache_data
def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        # Filter valid Jezero coordinates
        valid = {}
        for sol, info in data.items():
            lr = info.get("lon_range") or info.get("combined_lon_range", [0, 0])
            if 77.3 < lr[0] < 77.7:
                valid[sol] = info
        return valid
    return {}


@st.cache_data
def load_lonlat(sol: int):
    npz_path = OUTPUT_DIR / f"sol{sol:05d}_lonlat.npz"
    if npz_path.exists():
        data = np.load(str(npz_path))
        return data["lon"], data["lat"]
    return None, None


@st.cache_data
def load_texture(sol: int):
    """Load RAS texture as RGB image."""
    sol_dir = PDS_CACHE / f"sol{sol:05d}"
    if not sol_dir.exists():
        return None

    ras_files = sorted(sol_dir.glob("*RAS*.IMG"))
    if not ras_files:
        return None

    ras_path = ras_files[0]
    label_path = ras_path.with_suffix(".xml")

    try:
        from coregister.mastcam_xyz import parse_pds4_label
        meta = {}
        if label_path.exists():
            meta = parse_pds4_label(label_path)

        lines = meta.get("lines", 0)
        samples = meta.get("samples", 0)
        bands = meta.get("bands", 3)
        start_byte = meta.get("start_byte", 0)

        if lines == 0 or samples == 0:
            return None

        raw = np.fromfile(str(ras_path), dtype=">f4", offset=start_byte)
        expected = bands * lines * samples
        if raw.size < expected:
            return None

        raw = raw[:expected].reshape(bands, lines, samples)
        if bands >= 3:
            rgb = np.moveaxis(raw[:3], 0, -1).astype(np.float64)
        else:
            rgb = np.stack([raw[0]] * 3, axis=-1).astype(np.float64)

        from coregister.drape import normalize_to_uint8
        return normalize_to_uint8(rgb)
    except Exception as e:
        st.warning(f"Texture load error: {e}")
        return None


@st.cache_data
def load_dtm_overlay(sol: int):
    """Load DTM-draped PNG for verification overlay."""
    png_path = OUTPUT_DIR / f"sol{sol:05d}_dtm_draped.png"
    if png_path.exists():
        return Image.open(png_path)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────

results = load_results()

if not results:
    st.error("No coregistration results found. Run the pipeline first:")
    st.code("python run_batch.py\npython run_drape_dtm.py")
    st.stop()

st.sidebar.header("Sol Selection")
sol_list = sorted([int(s) for s in results.keys()])
selected_sol = st.sidebar.selectbox(
    "Sol",
    sol_list,
    format_func=lambda s: f"Sol {s}",
)

info = results[str(selected_sol)]
st.sidebar.markdown("---")
st.sidebar.markdown("**Product Info**")
pid = info.get('product_id', f"{info.get('n_frames', '?')} frames")
st.sidebar.text(f"ID: {str(pid)[:40]}...")
st.sidebar.text(f"Valid pts: {info.get('valid_points', info.get('total_valid_points', 0)):,}")
lon_range = info.get('lon_range') or info.get('combined_lon_range', [0, 0])
lat_range = info.get('lat_range') or info.get('combined_lat_range', [0, 0])
st.sidebar.text(f"Lon: [{lon_range[0]:.4f}, {lon_range[1]:.4f}]")
st.sidebar.text(f"Lat: [{lat_range[0]:.4f}, {lat_range[1]:.4f}]")

# ── Labels storage ────────────────────────────────────────────────────

LABELS_DIR = Path("/disk1/cspark/mastcam/labels")
LABELS_DIR.mkdir(exist_ok=True)


def load_labels(sol: int) -> dict:
    label_file = LABELS_DIR / f"sol{sol:05d}_labels.json"
    if label_file.exists():
        with open(label_file) as f:
            return json.load(f)
    return {"sol": sol, "annotations": []}


def save_labels(sol: int, labels: dict):
    label_file = LABELS_DIR / f"sol{sol:05d}_labels.json"
    with open(label_file, "w") as f:
        json.dump(labels, f, indent=2)


# ── Main Content ──────────────────────────────────────────────────────

tab_image, tab_coords, tab_label, tab_dtm = st.tabs([
    "Image", "Coordinates", "Labeling", "DTM Verification"
])

# ── Tab 1: Image ──────────────────────────────────────────────────────
with tab_image:
    st.subheader(f"Sol {selected_sol} - Mastcam-Z RAS Image")

    texture = load_texture(selected_sol)
    if texture is not None:
        st.image(texture, caption=f"RAS texture ({texture.shape[1]}x{texture.shape[0]})",
                 use_container_width=True)
    else:
        st.warning("RAS texture not available for this sol")

# ── Tab 2: Coordinates ───────────────────────────────────────────────
with tab_coords:
    st.subheader(f"Sol {selected_sol} - SPICE Coordinates")

    lon, lat = load_lonlat(selected_sol)

    if lon is not None:
        valid_mask = ~np.isnan(lon)
        valid_count = np.sum(valid_mask)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valid Points", f"{valid_count:,}")
        with col2:
            st.metric("Lon Range", f"{np.nanmin(lon):.6f} - {np.nanmax(lon):.6f}")
        with col3:
            st.metric("Lat Range", f"{np.nanmin(lat):.6f} - {np.nanmax(lat):.6f}")

        # Coordinate coverage visualization
        st.markdown("**Pixel Coordinate Map**")
        col_l, col_r = st.columns(2)

        with col_l:
            # Longitude heatmap
            lon_display = lon.copy()
            lon_display[~valid_mask] = np.nanmin(lon)
            lon_norm = (lon_display - np.nanmin(lon))
            denom = np.nanmax(lon) - np.nanmin(lon)
            if denom > 0:
                lon_norm = lon_norm / denom
            st.image(lon_norm, caption="Longitude (normalized)", clamp=True)

        with col_r:
            # Latitude heatmap
            lat_display = lat.copy()
            lat_display[~valid_mask] = np.nanmin(lat)
            lat_norm = (lat_display - np.nanmin(lat))
            denom = np.nanmax(lat) - np.nanmin(lat)
            if denom > 0:
                lat_norm = lat_norm / denom
            st.image(lat_norm, caption="Latitude (normalized)", clamp=True)

        # Valid point mask
        st.image(valid_mask.astype(np.float32), caption="Valid points mask", clamp=True)
    else:
        st.warning("No coordinate data for this sol")

# ── Tab 3: Labeling ──────────────────────────────────────────────────
with tab_label:
    st.subheader(f"Sol {selected_sol} - Annotation")

    labels = load_labels(selected_sol)

    # Show image for reference
    texture = load_texture(selected_sol)
    if texture is not None:
        st.image(texture, caption="Reference image (click coordinates below to annotate)",
                 use_container_width=True)

    # Label categories
    CATEGORIES = [
        "Bedrock", "Regolith", "Sand", "Pebble/Cobble", "Boulder",
        "Layered deposit", "Vein/Fracture fill", "Drill hole",
        "Rover track", "Artifact", "Other"
    ]

    st.markdown("---")
    st.markdown("**Add Annotation**")

    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", CATEGORIES)
        pixel_x = st.number_input("Pixel X", min_value=0, max_value=2000, value=0)
        pixel_y = st.number_input("Pixel Y", min_value=0, max_value=2000, value=0)

    with col2:
        notes = st.text_area("Notes", height=100)
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.1)

    # Compute lon/lat for the selected pixel
    lon, lat = load_lonlat(selected_sol)
    pixel_lon = pixel_lat = None
    if lon is not None and 0 <= pixel_y < lon.shape[0] and 0 <= pixel_x < lon.shape[1]:
        pixel_lon = float(lon[pixel_y, pixel_x])
        pixel_lat = float(lat[pixel_y, pixel_x])
        if not np.isnan(pixel_lon):
            st.info(f"Coordinates: lon={pixel_lon:.6f}, lat={pixel_lat:.6f}")
        else:
            st.warning("Selected pixel has no valid coordinates")

    if st.button("Add Annotation"):
        annotation = {
            "category": category,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
            "lon": pixel_lon if pixel_lon and not np.isnan(pixel_lon) else None,
            "lat": pixel_lat if pixel_lat and not np.isnan(pixel_lat) else None,
            "notes": notes,
            "confidence": confidence,
        }
        labels["annotations"].append(annotation)
        save_labels(selected_sol, labels)
        st.success(f"Annotation added ({category} at [{pixel_x}, {pixel_y}])")
        st.rerun()

    # Display existing annotations
    st.markdown("---")
    st.markdown(f"**Annotations ({len(labels['annotations'])})**")

    if labels["annotations"]:
        for i, ann in enumerate(labels["annotations"]):
            col_a, col_b, col_c = st.columns([3, 4, 1])
            with col_a:
                st.markdown(f"**{ann['category']}** @ ({ann['pixel_x']}, {ann['pixel_y']})")
            with col_b:
                if ann.get("lon") and ann.get("lat"):
                    st.caption(f"lon={ann['lon']:.6f}, lat={ann['lat']:.6f} | conf={ann['confidence']}")
                else:
                    st.caption(f"No coords | conf={ann['confidence']}")
            with col_c:
                if st.button("X", key=f"del_{i}"):
                    labels["annotations"].pop(i)
                    save_labels(selected_sol, labels)
                    st.rerun()
            if ann.get("notes"):
                st.caption(f"  {ann['notes']}")

        # Export
        if st.button("Export Labels (JSON)"):
            st.download_button(
                "Download",
                data=json.dumps(labels, indent=2),
                file_name=f"sol{selected_sol:05d}_labels.json",
                mime="application/json",
            )
    else:
        st.caption("No annotations yet")

# ── Tab 4: DTM Verification ──────────────────────────────────────────
with tab_dtm:
    st.subheader(f"Sol {selected_sol} - DTM Alignment Verification")
    st.caption("Compare SPICE-derived coordinates against HiRISE DTM draping for alignment quality check")

    dtm_img = load_dtm_overlay(selected_sol)
    if dtm_img is not None:
        st.image(dtm_img, caption="DTM-draped overlay (RGB on HiRISE DTM grid)",
                 use_container_width=True)

        # Show coordinate comparison
        lon, lat = load_lonlat(selected_sol)
        if lon is not None:
            valid = ~np.isnan(lon)
            st.markdown("**Coordinate Summary**")
            st.json({
                "spice_lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
                "spice_lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
                "valid_pixels": int(np.sum(valid)),
                "total_pixels": int(lon.size),
                "coverage_pct": f"{np.sum(valid) / lon.size * 100:.1f}%",
                "dtm_bounds": {
                    "lon": [77.4615, 77.5842],
                    "lat": [18.3368, 18.5886],
                    "resolution_m": 1.01,
                },
            })
    else:
        st.info("DTM overlay not available. Run `python run_drape_dtm.py` to generate.")
