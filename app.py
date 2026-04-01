"""MarsLab - Mastcam-Z Viewer & Labeling Platform"""

import streamlit as st

st.set_page_config(
    page_title="MarsLab",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MarsLab")
st.subheader("Mars 2020 Perseverance Mastcam-Z Analysis Platform")

st.markdown("""
### Pages

- **Viewer** -- Browse Mastcam-Z 360 panoramas from Jezero Crater
- **Labeling** -- Co-registered Mastcam-Z / HiRISE data with PDS XYZ + SPICE

### Pipeline

This platform includes a coregistration pipeline that:
1. Downloads Mastcam-Z XYZ stereo products from PDS Imaging Node
2. Transforms rover-frame coordinates to Mars body-fixed via SPICE
3. Drapes Mastcam-Z textures onto HiRISE DTM at 1 m/px resolution
4. Outputs georeferenced GeoTIFF products in Mars 2000 CRS

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Run the coregistration pipeline
python -m coregister --sol 22 --no-dtm
python run_batch.py        # batch all sols
python run_drape_dtm.py    # drape onto HiRISE DTM
```
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Sols Processed", "47")
with col2:
    st.metric("DTM Resolution", "1.01 m/px")
with col3:
    st.metric("HiRISE Coverage", "Jezero Crater")
