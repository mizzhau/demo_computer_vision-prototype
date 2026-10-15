import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from config import BASE_DIR, MANIFEST_PATH, SAMPLE_DIR, TARGET_CLASSES


def init_session_state():
    if "manifest_df" not in st.session_state:
        if MANIFEST_PATH.exists():
            st.session_state["manifest_df"] = pd.read_csv(MANIFEST_PATH)
        else:
            st.session_state["manifest_df"] = None

    if "input_image" not in st.session_state:
        st.session_state["input_image"] = None
        st.session_state["input_meta"] = None

    if "original_image" not in st.session_state:
        st.session_state["original_image"] = None

    if "current_nav" not in st.session_state:
        st.session_state["current_nav"] = "dataset"


def set_active_image(img_rgb: np.ndarray, meta: dict):
    st.session_state["input_image"] = img_rgb.copy()
    st.session_state["original_image"] = img_rgb.copy()
    st.session_state["input_meta"] = meta


def reset_active_image():
    """Xóa ảnh đang sử dụng hiện tại và chuyển về tab dataset để chọn/tải ảnh khác."""
    st.session_state["input_image"] = None
    st.session_state["original_image"] = None
    st.session_state["input_meta"] = None
    st.session_state["current_nav"] = "dataset"


@st.dialog("Lỗi, Chưa có ảnh")
def show_no_image_dialog():
    st.error("❌ **Lỗi: Chưa có ảnh được chọn!**")
    st.markdown(
        "Bạn chưa chọn ảnh từ thư viện dataset hoặc chưa tải ảnh cá nhân lên.\n\n"
        "Vui lòng vào tab **Dataset Explorer** để chọn một ảnh từ thư viện hoặc tải ảnh của bạn lên để áp dụng các kỹ thuật xử lý."
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👉 Đến Dataset Explorer", type="primary", use_container_width=True, key="dlg_btn_goto_dataset"):
            st.session_state["current_nav"] = "dataset"
            st.rerun()
    with col2:
        if st.button("Đóng", use_container_width=True, key="dlg_btn_close"):
            st.session_state["current_nav"] = "dataset"
            st.rerun()


def require_active_image() -> bool:
    """Kiểm tra xem đã có ảnh hay chưa. Nếu chưa, hiển thị pop-up và thông báo lỗi."""
    img = st.session_state.get("input_image")
    if img is None:
        st.toast("Lỗi, Chưa có ảnh!", icon="❌")
        show_no_image_dialog()
        st.error("### ❌ Lỗi, Chưa có ảnh")
        st.warning("Bạn chưa chọn ảnh mẫu từ thư viện hoặc chưa tải ảnh cá nhân lên.")
        if st.button("👉 Chọn ảnh từ thư viện / Tải ảnh lên", type="primary", key="btn_page_goto_dataset"):
            st.session_state["current_nav"] = "dataset"
            st.rerun()
        return False
    return True
