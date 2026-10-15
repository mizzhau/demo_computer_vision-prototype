import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

from config import BASE_DIR, MANIFEST_PATH, TARGET_CLASSES, CLASS_DISPLAY_NAMES, MAX_IMAGE_SIDE_PX
from app.state import set_active_image


@st.cache_data
def get_dataset_distribution_chart(csv_path_str: str):
    df_chart = pd.read_csv(csv_path_str)
    fig = px.histogram(
        df_chart, x="label", color="split",
        barmode="group",
        title="Phân phối số lượng ảnh theo loài và tập chia",
        color_discrete_map={"train": "#2E7D32", "val": "#F9A825", "test": "#1E88E5"}
    )
    fig.update_layout(xaxis_title="Loài cây", yaxis_title="Số lượng", margin=dict(t=40, b=40))
    return fig


def render():
    st.subheader("Dataset Explorer")

    df = st.session_state.get("manifest_df")
    if df is None and MANIFEST_PATH.exists():
        df = pd.read_csv(MANIFEST_PATH)
        st.session_state["manifest_df"] = df

    tab_sample, tab_upload, tab_stats = st.tabs([
        "Chọn ảnh mẫu", "Tải ảnh", "Thống kê"
    ])

    with tab_sample:
        selected_class = st.selectbox(
            "Chọn loài cây:",
            TARGET_CLASSES,
            format_func=lambda x: f"{CLASS_DISPLAY_NAMES.get(x, x)} ({x})"
        )

        if df is not None:
            class_df = df[df["label"] == selected_class].head(12)
            cols = st.columns(6)
            for idx, (_, row) in enumerate(class_df.iterrows()):
                col = cols[idx % 6]
                img_path = BASE_DIR / row["path"]
                if img_path.exists():
                    with Image.open(img_path) as pil_img:
                        thumb = pil_img.copy()
                    col.image(thumb, width=130)
                    if col.button("Chọn", key=f"btn_sel_{row['image_id']}", use_container_width=True):
                        with Image.open(img_path) as pil_img:
                            full_rgb = np.array(pil_img.convert("RGB"))
                        set_active_image(full_rgb, {
                            "source": "dataset",
                            "label": selected_class,
                            "file_name": img_path.name,
                            "path": str(img_path),
                            "seg_path": row.get("seg_path"),
                            "is_upload": False
                        })
                        st.toast(f"Đã chọn ảnh {img_path.name} thành công!", icon="✅")
                        st.rerun()

    with tab_upload:
        uploaded_file = st.file_uploader("Chọn tập tin ảnh:", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
                w, h = pil_img.size
                if max(w, h) > MAX_IMAGE_SIDE_PX:
                    scale = MAX_IMAGE_SIDE_PX / float(max(w, h))
                    new_w, new_h = int(w * scale), int(h * scale)
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

                img_rgb = np.array(pil_img)
                col_u1, col_u2 = st.columns([1, 2])
                with col_u1:
                    st.image(img_rgb, width=280)
                with col_u2:
                    st.info("Ảnh người dùng tải lên sẽ được áp dụng cho toàn bộ các tab xử lý.")
                    if st.button("Áp dụng ảnh này", type="primary"):
                        set_active_image(img_rgb, {
                            "source": "user_upload",
                            "label": "user_upload",
                            "file_name": uploaded_file.name,
                            "is_upload": True
                        })
                        st.toast(f"Đã áp dụng ảnh tải lên: {uploaded_file.name}!", icon="✅")
                        st.rerun()
            except Exception as e:
                st.error(f"Lỗi khi đọc ảnh: {e}")

    with tab_stats:
        if df is not None and MANIFEST_PATH.exists():
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Tổng số ảnh", f"{len(df):,}")
            with col_s2:
                st.metric("Số lượng loài", f"{df['label'].nunique()}")
            with col_s3:
                st.metric("Kích thước", f"{df['width'].iloc[0]} x {df['height'].iloc[0]}")
            with col_s4:
                st.metric("Train / Val / Test", "70% / 15% / 15%")

            fig = get_dataset_distribution_chart(str(MANIFEST_PATH))
            st.plotly_chart(fig, use_container_width=True)
