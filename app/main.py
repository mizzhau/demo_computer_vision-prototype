import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import CLASS_DISPLAY_NAMES
from app.state import init_session_state, reset_active_image, require_active_image
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
    #MainMenu {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
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
    meta = st.session_state.get("input_meta")

    if img is not None:
        st.markdown("**Ảnh đang sử dụng:**")
        st.image(img, width=200)
        lbl_display = "Ảnh người dùng tải lên"
        if meta and meta.get("label") and meta.get("label") != "user_upload":
            lbl_display = CLASS_DISPLAY_NAMES.get(meta.get("label"), meta.get("label", "N/A"))
        st.caption(f"{lbl_display} ({img.shape[1]}x{img.shape[0]} px)")

        if st.button("Đặt lại ảnh gốc", key="btn_reset_sidebar", use_container_width=True, help="Xóa ảnh hiện tại và chọn lại ảnh khác"):
            reset_active_image()
            st.toast("Đã xóa ảnh hiện tại. Vui lòng chọn hoặc tải ảnh mới.", icon="🔄")
            st.rerun()
    else:
        st.markdown("**Ảnh hiện tại:** *(Chưa có)*")
        st.caption("Chưa có ảnh nào được chọn.")
        if st.button("Chọn / Tải ảnh lên", key="btn_sidebar_choose", type="primary", use_container_width=True):
            st.session_state["current_nav"] = "dataset"
            st.rerun()

cur_tab = st.session_state["current_nav"]

PROCESSING_TABS = {
    "imgproc": tab2_imgproc,
    "geometry": tab3_geometry,
    "filters": tab4_filters,
    "segmentation": tab5_segmentation,
    "features": tab6_features,
    "deep_learning": tab8_deep_learning,
}

if cur_tab == "dataset":
    tab1_dataset.render()
elif cur_tab in PROCESSING_TABS:
    if require_active_image():
        PROCESSING_TABS[cur_tab].render()
elif cur_tab == "classical_ml":
    tab7_classical_ml.render()
elif cur_tab == "modern_cv":
    tab9_modern_cv.render()
