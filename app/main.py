import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CLASS_DISPLAY_NAMES
from app.state import init_session_state, reset_active_image
from app.tabs import (
    tab1_dataset,
    tab2_imgproc,
    tab3_geometry,
    tab4_filters,
    tab5_segmentation,
    tab6_features,
    tab7_classical_ml,
    tab8_deep_learning,
    tab9_modern_cv
)

st.set_page_config(
    page_title="Prototype",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        border-radius: 6px;
        height: 42px;
        font-weight: 500;
        font-size: 0.88rem;
        text-align: left;
        padding-left: 12px;
        margin-bottom: 3px;
        border: 1px solid #E2E8F0;
        background-color: #F8FAFC;
        color: #1E293B;
        line-height: 1.2;
    }
    div[data-testid="stSidebar"] div.stButton > button:hover {
        border-color: #2E7D32;
        color: #2E7D32;
        background-color: #F1F8E9;
    }
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background-color: #2E7D32;
        color: #FFFFFF;
        border-color: #2E7D32;
        font-weight: 600;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

init_session_state()

NAV_TABS = [
    ("dataset", "Dataset Explorer"),
    ("imgproc", "Image Processing"),
    ("geometry", "Geometric Transformations"),
    ("filters", "Noise & Filtering"),
    ("segmentation", "Segmentation & Morphology"),
    ("features", "Edge Detection & Feature Extraction"),
    ("classical_ml", "Classical Machine Learning"),
    ("deep_learning", "Deep Learning"),
    ("modern_cv", "Modern CV"),
]

if "current_nav" not in st.session_state:
    st.session_state["current_nav"] = "dataset"

with st.sidebar:
    st.markdown("### Prototype")

    for tab_id, tab_label in NAV_TABS:
        is_active = (st.session_state["current_nav"] == tab_id)
        btn_type = "primary" if is_active else "secondary"
        if st.button(tab_label, key=f"nav_{tab_id}", type=btn_type):
            st.session_state["current_nav"] = tab_id
            st.rerun()

    st.divider()

    img = st.session_state.get("input_image")
    meta = st.session_state.get("input_meta", {})

    if img is not None:
        st.markdown("**Ảnh hiện tại:**")
        st.image(img, width=200)
        lbl_display = CLASS_DISPLAY_NAMES.get(meta.get("label"), meta.get("label", "N/A"))
        st.caption(f"{lbl_display} ({img.shape[1]}x{img.shape[0]} px)")

        if st.button("Đặt lại ảnh gốc", key="btn_reset_sidebar", use_container_width=True):
            reset_active_image()
            st.success("Đã khôi phục ảnh ban đầu.")
            st.rerun()

cur_tab = st.session_state["current_nav"]

if cur_tab == "dataset":
    tab1_dataset.render()
elif cur_tab == "imgproc":
    tab2_imgproc.render()
elif cur_tab == "geometry":
    tab3_geometry.render()
elif cur_tab == "filters":
    tab4_filters.render()
elif cur_tab == "segmentation":
    tab5_segmentation.render()
elif cur_tab == "features":
    tab6_features.render()
elif cur_tab == "classical_ml":
    tab7_classical_ml.render()
elif cur_tab == "deep_learning":
    tab8_deep_learning.render()
elif cur_tab == "modern_cv":
    tab9_modern_cv.render()
