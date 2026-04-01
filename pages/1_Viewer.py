"""Mastcam-Z Panorama Viewer - Browse 360 panoramas from Jezero Crater."""

import json
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Viewer - MarsLab", layout="wide")
st.title("Mastcam-Z Panorama Viewer")

DOWNLOADS = Path("/disk1/cspark/mastcam/downloads")
PREVIEW_DIR = DOWNLOADS / "previews"
FULL_DIR = DOWNLOADS / "full"
META_PATH = DOWNLOADS / "panorama_metadata.json"


@st.cache_data
def load_metadata():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return []


@st.cache_data
def get_previews():
    if not PREVIEW_DIR.exists():
        return []
    return sorted(PREVIEW_DIR.glob("*_preview.jpg"))


@st.cache_data
def get_full_panoramas():
    if not FULL_DIR.exists():
        return []
    return sorted(FULL_DIR.glob("*_equirectangular.jpg"))


metadata = load_metadata()
previews = get_previews()
panoramas = get_full_panoramas()

st.sidebar.header("Panorama Browser")
st.sidebar.markdown(f"**{len(previews)}** previews, **{len(panoramas)}** full panoramas")

if metadata:
    st.subheader("Panorama Locations")

    locations = [m for m in metadata if m.get("pano_id", "").startswith("-") is False]
    pano_locations = [m for m in metadata if not m.get("pano_id", "").startswith("-")]

    if pano_locations:
        cols = st.columns(3)
        for i, m in enumerate(pano_locations[:30]):
            with cols[i % 3]:
                st.markdown(f"**{m['name']}** (#{m['pano_id']})")
                st.caption(f"lon: {m['lon']:.3f}, lat: {m['lat']:.3f}")

if previews:
    st.subheader("Preview Gallery")

    selected = st.selectbox(
        "Select panorama",
        previews,
        format_func=lambda p: p.stem.replace("_preview", "").replace("_", " "),
    )

    if selected:
        st.image(str(selected), use_container_width=True)

        # Try to find matching full panorama
        base_name = selected.stem.replace("_preview", "")
        full_match = FULL_DIR / f"{base_name}_equirectangular.jpg"
        if full_match.exists():
            with st.expander("Full Resolution Equirectangular"):
                st.image(str(full_match), use_container_width=True)
                st.caption(f"Size: {full_match.stat().st_size / 1024 / 1024:.1f} MB")

elif panoramas:
    st.subheader("Full Panoramas")
    selected = st.selectbox(
        "Select panorama",
        panoramas,
        format_func=lambda p: p.stem.replace("_equirectangular", "").replace("_", " "),
    )
    if selected:
        st.image(str(selected), use_container_width=True)

else:
    st.info("No panorama data found. Run `python crawl_mastcam.py` to download panoramas.")
