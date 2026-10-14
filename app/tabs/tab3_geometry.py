import streamlit as st
import numpy as np

from core.imgproc.geometry import (
    resize_image,
    crop_image,
    flip_image,
    affine_transform,
    perspective_transform,
    generate_augmentations
)
from app.components.compare import image_compare_view
from app.components.matrix import display_matrix


def render():
    st.subheader("Geometric Transformations")

    img = st.session_state.get("input_image")
    if img is None:
        st.warning("Vui lòng chọn hoặc tải ảnh tại phần dataset.")
        return

    h, w = img.shape[:2]

    tab_basic, tab_affine, tab_persp, tab_aug = st.tabs([
        "Co giãn, Cắt & Lật", "Biến đổi Afin (Affine)", "Biến đổi Phối cảnh (Perspective)", "Tăng cường (Augmentation)"
    ])

    with tab_basic:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            st.markdown("**Thao tác cơ bản:**")
            mode = st.radio(
                "Thao tác cơ bản:",
                ["Lật ảnh (Flip)", "Thay đổi kích thước (Resize)", "Cắt ảnh (Crop)"],
                label_visibility="collapsed"
            )

            if mode == "Lật ảnh (Flip)":
                st.markdown("**Hướng lật:**")
                flip_mode = st.selectbox(
                    "Hướng lật:",
                    ["horizontal (Ngang)", "vertical (Dọc)", "both (Cả hai)"],
                    label_visibility="collapsed"
                )
                clean_mode = flip_mode.split()[0]
                result = flip_image(img, mode=clean_mode)

            elif mode == "Thay đổi kích thước (Resize)":
                st.markdown("**Tỷ lệ co giãn (%):**")
                scale_pct = st.slider("Tỷ lệ co giãn (%):", 20, 200, 100, step=10, label_visibility="collapsed")
                st.markdown("**Phương pháp nội suy:**")
                interp = st.selectbox("Phương pháp nội suy:", ["bilinear", "nearest", "bicubic", "area"], label_visibility="collapsed")
                new_w, new_h = int(w * scale_pct / 100), int(h * scale_pct / 100)
                result = resize_image(img, new_w, new_h, interpolation=interp)

            else:
                st.markdown("**Tọa độ X bắt đầu:**")
                crop_x = st.slider("Tọa độ X bắt đầu:", 0, w - 20, 20, label_visibility="collapsed")
                st.markdown("**Tọa độ Y bắt đầu:**")
                crop_y = st.slider("Tọa độ Y bắt đầu:", 0, h - 20, 20, label_visibility="collapsed")
                st.markdown("**Độ rộng (Width):**")
                crop_w = st.slider("Độ rộng (Width):", 20, w - crop_x, w // 2, label_visibility="collapsed")
                st.markdown("**Chiều cao (Height):**")
                crop_h = st.slider("Chiều cao (Height):", 20, h - crop_y, h // 2, label_visibility="collapsed")
                result = crop_image(img, crop_x, crop_y, crop_w, crop_h)

        with col_view:
            image_compare_view(
                img, result,
                f"Ảnh gốc ({w}x{h})",
                f"Kết quả ({result.shape[1]}x{result.shape[0]})"
            )

    with tab_affine:
        col_a_ctrl, col_a_view = st.columns([1, 2])
        with col_a_ctrl:
            st.markdown("**Góc xoay (độ):**")
            angle = st.slider("Góc xoay (độ):", -180.0, 180.0, 0.0, step=5.0, label_visibility="collapsed")
            st.markdown("**Tỷ lệ co giãn (scale):**")
            scale = st.slider("Tỷ lệ co giãn (scale):", 0.3, 2.5, 1.0, step=0.1, label_visibility="collapsed")
            st.markdown("**Tịnh tiến trục X (tx px):**")
            tx = st.slider("Tịnh tiến trục X (tx px):", -100.0, 100.0, 0.0, step=5.0, label_visibility="collapsed")
            st.markdown("**Tịnh tiến trục Y (ty px):**")
            ty = st.slider("Tịnh tiến trục Y (ty px):", -100.0, 100.0, 0.0, step=5.0, label_visibility="collapsed")

            transformed, rot_mat = affine_transform(img, angle, scale, tx, ty)
            display_matrix(rot_mat, "Ma trận biến đổi Afin 2x3")

        with col_a_view:
            image_compare_view(img, transformed, "Ảnh gốc", f"Biến đổi Afin ({angle}°)")

    with tab_persp:
        col_p_ctrl, col_p_view = st.columns([1, 2])
        with col_p_ctrl:
            st.markdown("**Độ nghiêng phối cảnh (px):**")
            offset = st.slider("Độ nghiêng phối cảnh (px):", 0, w // 3, 30, step=5, label_visibility="collapsed")
            src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            dst_pts = np.float32([
                [offset, offset],
                [w - 1 - offset, 0],
                [w - 1, h - 1],
                [0, h - 1 - offset]
            ])
            warped, persp_mat = perspective_transform(img, src_pts, dst_pts)
            display_matrix(persp_mat, "Ma trận biến đổi phối cảnh 3x3")

        with col_p_view:
            image_compare_view(img, warped, "Ảnh gốc", "Ảnh phối cảnh")

    with tab_aug:
        col_aug_seed, _ = st.columns([1, 3])
        with col_aug_seed:
            st.markdown("**Giá trị Seed ngẫu nhiên:**")
            seed_val = st.number_input("Giá trị Seed ngẫu nhiên:", value=42, step=1, label_visibility="collapsed")

        aug_list = generate_augmentations(img, n=8, seed=int(seed_val))
        cols = st.columns(4)
        for idx, (aug_img, desc) in enumerate(aug_list):
            cols[idx % 4].image(aug_img, caption=desc, width=170)
