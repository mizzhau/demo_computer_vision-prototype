import streamlit as st
import cv2
import numpy as np
import pandas as pd

from core.imgproc.edges import apply_sobel, apply_laplacian, apply_canny
from core.imgproc.segment import apply_threshold, apply_morphology, extract_foreground_leaf
from core.features.shape import extract_largest_contour, compute_shape_features, draw_geometric_overlays
from core.features.color_hist import compute_color_features
from core.features.hog_feat import compute_hog_features
from app.state import require_active_image


def render():
    st.subheader("Edge Detection & Feature Extraction")

    img = st.session_state.get("input_image")
    if img is None:
        require_active_image()
        return

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    tab_edges, tab_shape, tab_hog, tab_export = st.tabs([
        "Cạnh (Edges)", "Đặc trưng hình học", "Đặc trưng HOG", "Xuất vector"
    ])

    with tab_edges:
        col_e_ctrl, col_e_view = st.columns([1, 2])
        with col_e_ctrl:
            st.markdown("**Toán tử phát hiện cạnh:**")
            edge_algo = st.selectbox(
                "Toán tử cạnh:",
                ["Canny Edge Detector", "Sobel Filter", "Laplacian of Gaussian"],
                label_visibility="collapsed"
            )
            if edge_algo == "Canny Edge Detector":
                st.markdown("**Ngưỡng dưới (Threshold 1):**")
                th1 = st.slider("Threshold 1:", 10, 200, 50, label_visibility="collapsed")
                st.markdown("**Ngưỡng trên (Threshold 2):**")
                th2 = st.slider("Threshold 2:", 50, 300, 150, label_visibility="collapsed")
                edge_img = apply_canny(gray, th1, th2)
            elif edge_algo == "Sobel Filter":
                st.markdown("**Kernel Sobel:**")
                sobel_k = st.slider("Kernel Sobel:", 3, 7, 3, step=2, label_visibility="collapsed")
                abs_x, abs_y, edge_img = apply_sobel(gray, ksize=sobel_k)
            else:
                st.markdown("**Kernel Laplacian:**")
                lap_k = st.slider("Kernel Laplacian:", 3, 7, 3, step=2, label_visibility="collapsed")
                edge_img = apply_laplacian(gray, ksize=lap_k)

        with col_e_view:
            st.image(edge_img, caption=f"Bản đồ cạnh ({edge_algo})", clamp=True, width=320)

    img_hash = hash(img.tobytes())

    def get_shape_color_cached():
        cache_key = f"shape_color_{img_hash}"
        if cache_key not in st.session_state:
            mask_otsu, _ = apply_threshold(gray, method="otsu")
            mask_clean = apply_morphology(mask_otsu, op_name="closing", kernel_size=5)
            clean_mask, _, _ = extract_foreground_leaf(img, mask_clean)
            contour = extract_largest_contour(clean_mask)
            s_feats = compute_shape_features(contour)
            c_feats = compute_color_features(img, mask=clean_mask)
            st.session_state[cache_key] = (s_feats, c_feats, contour)
        return st.session_state[cache_key]

    with tab_shape:
        col_s_img, col_s_table = st.columns([1, 1])
        shape_feats, color_feats, contour = get_shape_color_cached()

        with col_s_img:
            overlay_geom = draw_geometric_overlays(img, contour)
            st.image(overlay_geom, caption="Contour & Bounding Box", width=280)

        with col_s_table:
            st.markdown("**Bảng thông số hình học trích xuất:**")
            st.write(f"- **Diện tích (Area):** `{shape_feats['area']:.1f} px²`")
            st.write(f"- **Chu vi (Perimeter):** `{shape_feats['perimeter']:.1f} px`")
            st.write(f"- **Tỷ lệ khung hình (Aspect Ratio):** `{shape_feats['aspect_ratio']:.3f}`")
            st.write(f"- **Độ đặc (Extent):** `{shape_feats['extent']:.3f}`")
            st.write(f"- **Độ lồi (Solidity):** `{shape_feats['solidity']:.3f}`")
            st.write(f"- **Độ tròn (Circularity):** `{shape_feats['circularity']:.3f}`")
            st.write(f"- **Hu Moments:** `{', '.join([f'{h:.2f}' for h in shape_feats['hu_moments']])}`")

    with tab_hog:
        col_h1, col_h2 = st.columns([1, 1])
        cache_key_hog = f"hog_{img_hash}"
        if cache_key_hog not in st.session_state:
            hog_vec, hog_img = compute_hog_features(img)
            st.session_state[cache_key_hog] = (hog_vec, hog_img)
        else:
            hog_vec, hog_img = st.session_state[cache_key_hog]
        with col_h1:
            st.image(hog_img, caption=f"HOG (Chiều dài: {len(hog_vec)})", width=260)
        with col_h2:
            st.markdown("**Mô tả đặc trưng HOG:**")
            st.caption(
                "HOG chia ảnh thành các cell nhỏ, tính toán hướng gradient cường độ pixel "
                "để mô tả cấu trúc viền và gân lá cây mà không phụ thuộc vào độ sáng tuyệt đối."
            )

    with tab_export:
        shape_feats, color_feats, _ = get_shape_color_cached()
        geom_vals = [
            shape_feats["area"], shape_feats["perimeter"], shape_feats["aspect_ratio"],
            shape_feats["extent"], shape_feats["solidity"], shape_feats["circularity"]
        ] + shape_feats["hu_moments"]
        color_vals = color_feats["hist_hsv_32d"] + [
            color_feats["channel_stats"]["r_mean"], color_feats["channel_stats"]["r_std"],
            color_feats["channel_stats"]["g_mean"], color_feats["channel_stats"]["g_std"],
            color_feats["channel_stats"]["b_mean"], color_feats["channel_stats"]["b_std"]
        ]
        full_vec = geom_vals + color_vals
        cols_names = [f"geom_{i}" for i in range(len(geom_vals))] + [f"color_{i}" for i in range(len(color_vals))]
        vec_df = pd.DataFrame([full_vec], columns=cols_names)

        st.markdown(f"**Vector đặc trưng ({len(full_vec)} chiều):**")
        st.dataframe(vec_df, use_container_width=True)

        csv_data = vec_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Tải xuống CSV",
            data=csv_data,
            file_name="leaf_feature_vector.csv",
            mime="text/csv"
        )
