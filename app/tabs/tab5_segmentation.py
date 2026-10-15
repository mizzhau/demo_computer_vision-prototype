import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from config import BASE_DIR
from core.imgproc.segment import apply_threshold, apply_morphology, extract_foreground_leaf
from core.evaluation.metrics import compute_iou_dice
from app.state import require_active_image


def render():
    st.subheader("Segmentation & Morphology")

    img = st.session_state.get("input_image")
    meta = st.session_state.get("input_meta", {})
    if img is None:
        require_active_image()
        return

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Kênh đầu vào phân ngưỡng:**")
        channel_select = st.selectbox(
            "Kênh đầu vào:",
            ["Grayscale", "Kênh Saturation (HSV)", "Kênh Value (HSV)"],
            label_visibility="collapsed"
        )
        st.markdown("**Thuật toán phân ngưỡng:**")
        method = st.selectbox(
            "Thuật toán:",
            ["Otsu", "Global Threshold", "Adaptive Mean", "Adaptive Gaussian"],
            label_visibility="collapsed"
        )

        if channel_select == "Grayscale":
            gray_input = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif "Saturation" in channel_select:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            gray_input = hsv[:, :, 1]
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            gray_input = hsv[:, :, 2]

        thresh_val = 127
        if "Global" in method:
            st.markdown("**Giá trị ngưỡng (0-255):**")
            thresh_val = st.slider("Giá trị ngưỡng:", 0, 255, 127, label_visibility="collapsed")

        invert_mask = st.checkbox("Đảo ngược mặt nạ nhị phân", value=False)
        method_code = "otsu"
        if "Global" in method:
            method_code = "global"
        elif "Mean" in method:
            method_code = "adaptive_mean"
        elif "Gaussian" in method:
            method_code = "adaptive_gaussian"

        raw_mask, computed_th = apply_threshold(
            gray_input, method=method_code, threshold_value=thresh_val, invert=invert_mask
        )
        if method_code == "otsu":
            st.caption(f"Ngưỡng Otsu tự động: {computed_th:.1f}")

        st.markdown("**Phép toán hình thái:**")
        morph_op = st.selectbox(
            "Phép toán hình thái:",
            ["Opening", "Closing", "Erosion", "Dilation"],
            label_visibility="collapsed"
        )
        op_code = morph_op.lower()
        st.markdown("**Kích thước Kernel:**")
        k_size = st.slider("Kích thước Kernel:", 3, 15, 5, step=2, label_visibility="collapsed")
        st.markdown("**Số lần lặp (Iterations):**")
        iters = st.slider("Số lần lặp:", 1, 5, 1, label_visibility="collapsed")

        clean_mask = apply_morphology(raw_mask, op_name=op_code, kernel_size=k_size, iterations=iters)

    final_mask, foreground, overlay = extract_foreground_leaf(img, clean_mask, keep_largest_only=True)

    with col_main:
        r1_c1, r1_c2 = st.columns(2)
        r1_c1.image(raw_mask, caption="Mặt nạ nhị phân thô", clamp=True, width=240)
        r1_c2.image(final_mask, caption="Mặt nạ sau Morphology", clamp=True, width=240)

        r2_c1, r2_c2 = st.columns(2)
        r2_c1.image(overlay, caption="Mặt nạ phủ lên ảnh gốc", width=240)
        r2_c2.image(foreground, caption="Tiền cảnh bóc tách", width=240)

    seg_rel_path = meta.get("seg_path")
    if (
        isinstance(seg_rel_path, str) and
        seg_rel_path.strip() and
        not pd.isna(seg_rel_path) and
        not meta.get("is_upload")
    ):
        gt_full_path = BASE_DIR / seg_rel_path
        if gt_full_path.exists():
            st.divider()
            st.markdown("**Đánh giá với Ground Truth:**")
            with Image.open(gt_full_path) as pil_gt:
                gt_rgb = np.array(pil_gt.convert("RGB"))
            gt_mask = (cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY) > 10).astype(np.uint8) * 255

            iou, dice = compute_iou_dice(final_mask, gt_mask)

            c_gt1, c_gt2, c_gt3 = st.columns([1, 1, 1])
            c_gt1.image(gt_mask, caption="Ground Truth", clamp=True, width=160)
            c_gt2.metric("IoU (Jaccard)", f"{iou * 100:.2f}%")
            c_gt3.metric("Dice Score", f"{dice * 100:.2f}%")
