import streamlit as st
import numpy as np
import pandas as pd

from core.imgproc.filters import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_mean_filter,
    apply_gaussian_blur,
    apply_median_filter,
    apply_bilateral_filter,
    compute_mse,
    compute_psnr
)
from app.components.compare import image_compare_view
from app.state import require_active_image


def render():
    st.subheader("Noise & Filtering")

    clean_img = st.session_state.get("input_image")
    if clean_img is None:
        require_active_image()
        return

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        st.markdown("**Loại nhiễu:**")
        noise_type = st.radio(
            "Loại nhiễu:",
            ["Gaussian Noise", "Salt & Pepper"],
            label_visibility="collapsed"
        )

        if "Gaussian" in noise_type:
            st.markdown("**Độ lệch chuẩn nhiễu (σ):**")
            sigma = st.slider("Độ lệch chuẩn nhiễu (σ):", 0.0, 60.0, 25.0, step=5.0, label_visibility="collapsed")
            noisy_img = add_gaussian_noise(clean_img, sigma=sigma, seed=42)
        else:
            st.markdown("**Tỷ lệ điểm nhiễu (%):**")
            amount = st.slider("Tỷ lệ điểm nhiễu (%):", 0.0, 0.20, 0.05, step=0.01, label_visibility="collapsed")
            noisy_img = add_salt_and_pepper_noise(clean_img, amount=amount, seed=42)

        st.markdown("**Bộ lọc xem chi tiết:**")
        selected_filter = st.selectbox(
            "Bộ lọc xem chi tiết:",
            ["Gaussian Blur", "Median Filter", "Mean Filter", "Bilateral Filter"],
            label_visibility="collapsed"
        )
        st.markdown("**Kích thước Kernel (ksize):**")
        kernel_size = st.slider("Kích thước Kernel (ksize):", 3, 15, 5, step=2, label_visibility="collapsed")

    filtered_mean = apply_mean_filter(noisy_img, kernel_size)
    filtered_gauss = apply_gaussian_blur(noisy_img, kernel_size, sigma=1.5)
    filtered_median = apply_median_filter(noisy_img, kernel_size)
    filtered_bilateral = apply_bilateral_filter(noisy_img, d=kernel_size)

    filter_dict = {
        "Gaussian Blur": filtered_gauss,
        "Median Filter": filtered_median,
        "Mean Filter": filtered_mean,
        "Bilateral Filter": filtered_bilateral
    }
    active_filtered = filter_dict[selected_filter]

    with col_main:
        image_compare_view(
            noisy_img, active_filtered,
            "Ảnh có nhiễu",
            f"Sau khi lọc ({selected_filter})"
        )

    st.markdown("**So sánh định lượng chất lượng khử nhiễu (MSE & PSNR):**")
    eval_rows = []
    for f_name, f_img in filter_dict.items():
        mse_val = compute_mse(clean_img, f_img)
        psnr_val = compute_psnr(clean_img, f_img)
        eval_rows.append({
            "Bộ lọc": f_name,
            "Kernel": f"{kernel_size}x{kernel_size}",
            "MSE": round(mse_val, 2),
            "PSNR (dB)": round(psnr_val, 2)
        })

    eval_df = pd.DataFrame(eval_rows).sort_values(by="PSNR (dB)", ascending=False)
    st.dataframe(eval_df, use_container_width=True)

    with st.expander("Soi chi tiết gân lá (Crop Zoom 64x64)"):
        h, w = clean_img.shape[:2]
        cx, cy = w // 2, h // 2
        crop_clean = clean_img[cy-32:cy+32, cx-32:cx+32]
        crop_noisy = noisy_img[cy-32:cy+32, cx-32:cx+32]
        crop_filtered = active_filtered[cy-32:cy+32, cx-32:cx+32]

        zc1, zc2, zc3 = st.columns(3)
        zc1.image(crop_clean, caption="Ảnh gốc sạch", width=140)
        zc2.image(crop_noisy, caption="Vùng nhiễu", width=140)
        zc3.image(crop_filtered, caption=f"Sau lọc ({selected_filter})", width=140)
