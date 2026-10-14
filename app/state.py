import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

from config import BASE_DIR, MANIFEST_PATH, SAMPLE_DIR, TARGET_CLASSES


def init_session_state():
    if "manifest_df" not in st.session_state:
        if MANIFEST_PATH.exists():
            st.session_state["manifest_df"] = pd.read_csv(MANIFEST_PATH)
        else:
            st.session_state["manifest_df"] = None

    if "input_image" not in st.session_state:
        default_img_path = None
        seg_path_val = None
        if st.session_state["manifest_df"] is not None and len(st.session_state["manifest_df"]) > 0:
            row = st.session_state["manifest_df"].iloc[0]
            candidate = BASE_DIR / row["path"]
            if candidate.exists():
                default_img_path = candidate
                raw_seg = row.get("seg_path")
                if isinstance(raw_seg, str) and raw_seg.strip() and not pd.isna(raw_seg):
                    seg_path_val = raw_seg

        if default_img_path and default_img_path.exists():
            with Image.open(default_img_path) as pil_img:
                st.session_state["input_image"] = np.array(pil_img.convert("RGB"))
            st.session_state["input_meta"] = {
                "source": "dataset",
                "label": row["label"],
                "file_name": default_img_path.name,
                "path": str(default_img_path),
                "seg_path": seg_path_val,
                "is_upload": False
            }
        else:
            st.session_state["input_image"] = np.zeros((256, 256, 3), dtype=np.uint8)
            st.session_state["input_meta"] = {
                "source": "empty",
                "label": "None",
                "file_name": "empty",
                "seg_path": None,
                "is_upload": False
            }

    if "original_image" not in st.session_state:
        st.session_state["original_image"] = st.session_state["input_image"].copy()


def set_active_image(img_rgb: np.ndarray, meta: dict):
    st.session_state["input_image"] = img_rgb.copy()
    st.session_state["original_image"] = img_rgb.copy()
    st.session_state["input_meta"] = meta


def reset_active_image():
    if "original_image" in st.session_state:
        st.session_state["input_image"] = st.session_state["original_image"].copy()
