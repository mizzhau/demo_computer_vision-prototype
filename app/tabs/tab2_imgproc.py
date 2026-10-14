import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.imgproc.color import (
    convert_color_space,
    adjust_brightness_contrast,
    compute_histogram,
    apply_clahe,
    apply_hist_equalization,
    extract_pixel_neighborhood
)
from app.components.compare import image_compare_view


def render():
    st.subheader("Image Processing Lab")

    img = st.session_state.get("input_image")
    if img is None:
        st.warning("Vui lòng chọn hoặc tải ảnh tại phần dataset.")
        return

    tab_colorspace, tab_adj, tab_hist, tab_pixel = st.tabs([
        "Không gian màu", "Độ sáng & Tương phản", "Histogram", "Tra cứu pixel"
    ])

    with tab_colorspace:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Chọn không gian màu biểu diễn:**")
            space = st.selectbox(
                "Không gian màu:",
                ["RGB (Chuẩn)", "BGR (Kênh đảo)", "Grayscale (Ảnh xám)", "HSV (Màu, Bão hòa, Sáng)"],
                label_visibility="collapsed"
            )

        target_code = "RGB"
        if "BGR" in space:
            target_code = "BGR"
        elif "Gray" in space:
            target_code = "GRAY"
        elif "HSV" in space:
            target_code = "HSV"

        converted = convert_color_space(img, target_code)

        with col2:
            st.image(
                converted,
                caption=f"{space} - Shape: {converted.shape}",
                width=300
            )

        st.markdown("**Phân tách 3 kênh màu:**")
        if target_code == "RGB":
            view_mode = st.radio(
                "Chế độ hiển thị kênh màu:",
                ["Ánh màu trực quan", "Thang xám cường độ (0-255)"],
                horizontal=True
            )

            c_r, c_g, c_b = st.columns(3)
            if "Ánh màu" in view_mode:
                red_channel = np.zeros_like(img)
                red_channel[:, :, 0] = img[:, :, 0]

                green_channel = np.zeros_like(img)
                green_channel[:, :, 1] = img[:, :, 1]

                blue_channel = np.zeros_like(img)
                blue_channel[:, :, 2] = img[:, :, 2]

                c_r.image(red_channel, caption="Kênh Đỏ", width=220)
                c_g.image(green_channel, caption="Kênh Lục", width=220)
                c_b.image(blue_channel, caption="Kênh Lam", width=220)
            else:
                c_r.image(img[:, :, 0], caption="Kênh Đỏ (0-255)", clamp=True, width=220)
                c_g.image(img[:, :, 1], caption="Kênh Lục (0-255)", clamp=True, width=220)
                c_b.image(img[:, :, 2], caption="Kênh Lam (0-255)", clamp=True, width=220)

        elif target_code == "HSV":
            c_h, c_s, c_v = st.columns(3)
            c_h.image(converted[:, :, 0], caption="Kênh Hue", clamp=True, width=220)
            c_s.image(converted[:, :, 1], caption="Kênh Saturation", clamp=True, width=220)
            c_v.image(converted[:, :, 2], caption="Kênh Value", clamp=True, width=220)

    with tab_adj:
        col_ctrl, col_view = st.columns([1, 2])
        with col_ctrl:
            st.markdown("**Hệ số tương phản (α):**")
            alpha = st.slider("Hệ số tương phản (α)", 0.1, 3.0, 1.0, step=0.1, label_visibility="collapsed")
            st.markdown("**Độ lệch sáng (β):**")
            beta = st.slider("Độ lệch sáng (β)", -100, 100, 0, step=5, label_visibility="collapsed")
            st.latex(rf"I_{{out}} = \text{{clip}}({alpha:.1f} \cdot I_{{in}} + {beta}, 0, 255)")

        adjusted = adjust_brightness_contrast(img, alpha=alpha, beta=beta)
        with col_view:
            image_compare_view(img, adjusted, "Ảnh gốc", f"Đã biến đổi (α={alpha}, β={beta})")

    with tab_hist:
        col_h_ctrl, col_h_view = st.columns([1, 2])
        with col_h_ctrl:
            st.markdown("**Phương pháp xử lý Histogram:**")
            method = st.radio(
                "Phương pháp xử lý Histogram",
                ["Histogram gốc", "Cân bằng lược đồ (Hist Equalization)", "CLAHE"],
                label_visibility="collapsed"
            )
            clip_lim = 2.0
            if method == "CLAHE":
                st.markdown("**Clip Limit:**")
                clip_lim = st.slider("Clip Limit", 1.0, 10.0, 2.0, step=0.5, label_visibility="collapsed")

        if method == "Cân bằng lược đồ (Hist Equalization)":
            proc_img = apply_hist_equalization(img)
        elif method == "CLAHE":
            proc_img = apply_clahe(img, clip_limit=clip_lim)
        else:
            proc_img = img

        with col_h_view:
            image_compare_view(img, proc_img, "Ảnh trước", "Ảnh sau xử lý")

        hists = compute_histogram(proc_img)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=hists["r"], mode="lines", name="Kênh Đỏ (R)", line=dict(color="#E53935")))
        fig.add_trace(go.Scatter(y=hists["g"], mode="lines", name="Kênh Lục (G)", line=dict(color="#43A047")))
        fig.add_trace(go.Scatter(y=hists["b"], mode="lines", name="Kênh Lam (B)", line=dict(color="#1E88E5")))
        fig.add_trace(go.Scatter(y=hists["gray"], mode="lines", name="Mức xám", line=dict(color="#757575", dash="dash")))
        fig.update_layout(title="Lược đồ tần suất Histogram", xaxis_title="Cường độ pixel (0-255)", yaxis_title="Số lượng", margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with tab_pixel:
        h, w = img.shape[:2]
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.markdown("**Tọa độ X (chiều rộng):**")
            px_x = st.number_input("Tọa độ X", min_value=0, max_value=w - 1, value=w // 2, label_visibility="collapsed")
            st.markdown("**Tọa độ Y (chiều cao):**")
            px_y = st.number_input("Tọa độ Y", min_value=0, max_value=h - 1, value=h // 2, label_visibility="collapsed")
            nh = extract_pixel_neighborhood(img, px_x, px_y, size=5)
            st.write(f"- **RGB:** `{nh['rgb']}`")
            st.write(f"- **BGR:** `{nh['bgr']}`")
            st.write(f"- **HSV:** `{nh['hsv']}`")

        with col_p2:
            st.markdown(f"**Ma trận 5x5 kênh Green tại ({px_x}, {px_y}):**")
            st.dataframe(nh["region_rgb"][:, :, 1], use_container_width=True)
