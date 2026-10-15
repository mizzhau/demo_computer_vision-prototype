import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cv2


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
        "Không gian màu", "Độ sáng & tương phản", "Histogram", "Tra cứu pixel"
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

        st.divider()

        def make_hist_figure(target_img: np.ndarray, chart_title: str) -> go.Figure:
            hists = compute_histogram(target_img)
            f = go.Figure()
            f.add_trace(go.Scatter(y=hists["r"], mode="lines", name="Kênh Đỏ (R)", line=dict(color="#E53935", width=1.5)))
            f.add_trace(go.Scatter(y=hists["g"], mode="lines", name="Kênh Lục (G)", line=dict(color="#43A047", width=1.5)))
            f.add_trace(go.Scatter(y=hists["b"], mode="lines", name="Kênh Lam (B)", line=dict(color="#1E88E5", width=1.5)))
            f.add_trace(go.Scatter(y=hists["gray"], mode="lines", name="Mức xám", line=dict(color="#757575", dash="dash", width=1.5)))
            f.update_layout(
                title=dict(text=chart_title, font=dict(size=14)),
                xaxis_title="Cường độ pixel (0-255)",
                yaxis_title="Số lượng pixel",
                margin=dict(t=45, b=30, l=40, r=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            return f

        if method == "Histogram gốc":
            fig_orig = make_hist_figure(img, "Lược đồ tần suất Histogram đa kênh (RGB & Mức xám)")
            st.plotly_chart(fig_orig, use_container_width=True)
        else:
            col_h_fig1, col_h_fig2 = st.columns(2)
            with col_h_fig1:
                fig_before = make_hist_figure(img, "1. Lược đồ Histogram đa kênh — Ảnh gốc")
                st.plotly_chart(fig_before, use_container_width=True)
            with col_h_fig2:
                fig_after = make_hist_figure(proc_img, f"2. Lược đồ Histogram đa kênh — Sau {method}")
                st.plotly_chart(fig_after, use_container_width=True)

        st.divider()
        st.markdown("#### Lược đồ Histogram chuyên biệt thang mức xám (Grayscale Histogram)")

        gray_before = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(proc_img, cv2.COLOR_RGB2GRAY)

        def make_gray_bar_chart(gray_mat: np.ndarray, title_text: str, bar_color: str) -> go.Figure:
            h = cv2.calcHist([gray_mat], [0], None, [256], [0, 256]).flatten()
            f = go.Figure(go.Bar(
                x=list(range(256)),
                y=h,
                marker=dict(color=bar_color),
                name="Pixel count"
            ))
            f.update_layout(
                title=dict(text=title_text, font=dict(size=13)),
                xaxis=dict(title="Thang mức xám (0: Đen ➔ 255: Trắng)", range=[-1, 256], dtick=32),
                yaxis_title="Số lượng pixel",
                margin=dict(t=40, b=30, l=40, r=20),
                height=280
            )
            return f

        if method == "Histogram gốc":
            fig_gray = make_gray_bar_chart(gray_before, "Phân bố thang mức xám — Ảnh gốc (Dạng cột Bar)", "#475569")
            st.plotly_chart(fig_gray, use_container_width=True)
        else:
            col_gb1, col_gb2 = st.columns(2)
            with col_gb1:
                fig_gb = make_gray_bar_chart(gray_before, "Thang mức xám ảnh gốc (Trước cân bằng)", "#64748B")
                st.plotly_chart(fig_gb, use_container_width=True)
            with col_gb2:
                fig_ga = make_gray_bar_chart(gray_after, f"Thang mức xám sau khi {method}", "#16A34A")
                st.plotly_chart(fig_ga, use_container_width=True)

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
