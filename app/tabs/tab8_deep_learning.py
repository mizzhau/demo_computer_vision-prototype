import streamlit as st
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image

from config import METRICS_PATH, ARTIFACTS_DIR, TARGET_CLASSES, CLASS_DISPLAY_NAMES, MAX_IMAGE_SIDE_PX
from core.models.dl_stubs import (
    apply_manual_convolution,
    extract_cnn_feature_maps,
    predict_cnn
)


@st.cache_data
def get_model_comparison_df_cached(dl_metrics_str: str, ml_metrics_str: str) -> pd.DataFrame:
    cmp_rows = []
    dl_p = Path(dl_metrics_str)
    ml_p = Path(ml_metrics_str)

    if dl_p.exists():
        with open(dl_p, "r", encoding="utf-8") as f:
            dl_m = json.load(f)
        for k, v in dl_m.items():
            cmp_rows.append({
                "Họ mô hình": "Deep Learning (CNN)",
                "Kiến trúc": "ResNet-18" if k == "resnet18" else "Classic TinyCNN",
                "Đặc trưng": "Học tự động (End-to-End)",
                "Accuracy": f"{v['accuracy']*100:.2f}%",
                "Macro F1": f"{v['macro_f1']*100:.2f}%",
                "Inference (ms/ảnh)": v["infer_ms_per_image"],
                "Kích thước (MB)": v["model_size_mb"]
            })

    if ml_p.exists():
        with open(ml_p, "r", encoding="utf-8") as f:
            ml_m = json.load(f)
        for k, v in ml_m.items():
            cmp_rows.append({
                "Họ mô hình": "Classical ML",
                "Kiến trúc": v["algo_name"].upper(),
                "Đặc trưng": "Thủ công (Shape/Color)" if v["feature_type"] == "shape_color" else "Thủ công (HOG)",
                "Accuracy": f"{v['accuracy']*100:.2f}%",
                "Macro F1": f"{v['f1_macro']*100:.2f}%",
                "Inference (ms/ảnh)": v["infer_ms_per_image"],
                "Kích thước (MB)": v["model_size_mb"]
            })

    if cmp_rows:
        return pd.DataFrame(cmp_rows).sort_values(by="Accuracy", ascending=False)
    return pd.DataFrame()


def render():
    st.subheader("Deep Learning & Model Comparison")

    img = st.session_state.get("input_image")
    meta = st.session_state.get("input_meta", {})
    if img is None:
        st.warning("Vui lòng chọn hoặc tải ảnh tại phần dataset.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    tab_conv, tab_fmaps, tab_infer, tab_compare = st.tabs([
        "Minh họa tích chập", "CNN Feature Maps", "Phân loại CNN", "So sánh với ML cổ điển"
    ])

    with tab_conv:
        col_k_ctrl, col_k_view = st.columns([1, 2])
        with col_k_ctrl:
            st.markdown("**Chọn ma trận tích chập (Kernel 3x3):**")
            k_preset = st.selectbox(
                "Kernel tích chập:",
                ["Sobel X (Dò cạnh dọc)", "Sobel Y (Dò cạnh ngang)", "Sharpen (Làm sắc nét)", "Blur (Làm mờ trung bình)", "Ridge Detection"]
            )
            if "Sobel X" in k_preset:
                kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            elif "Sobel Y" in k_preset:
                kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            elif "Sharpen" in k_preset:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            elif "Blur" in k_preset:
                kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            else:
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

            st.markdown("**Phép gộp (Pooling):**")
            pool_type = st.radio("Loại Pooling:", ["Max Pooling (2x2)", "Average Pooling (2x2)"], horizontal=True)
            p_code = "avg" if "Average" in pool_type else "max"

            fmap, pooled = apply_manual_convolution(gray, kernel, pool_type=p_code)

        with col_k_view:
            c1, c2, c3 = st.columns(3)
            c1.image(gray, caption=f"Ảnh đầu vào ({gray.shape[1]}x{gray.shape[0]})", clamp=True, width=190)
            c2.image(fmap, caption=f"Feature Map ({fmap.shape[1]}x{fmap.shape[0]})", clamp=True, width=190)
            c3.image(pooled, caption=f"Sau Pooling ({pooled.shape[1]}x{pooled.shape[0]})", clamp=True, width=190)

    with tab_fmaps:
        col_fm_ctrl, col_fm_view = st.columns([1, 3])
        with col_fm_ctrl:
            st.markdown("**Mô hình trích xuất:**")
            fm_model = st.selectbox("Chọn mô hình:", ["tinycnn", "resnet18"], format_func=lambda x: "Classic TinyCNN (3 khối Conv)" if x == "tinycnn" else "ResNet-18")
            btn_extract = st.button("Trích xuất Feature Maps", type="primary")

        with col_fm_view:
            if btn_extract or f"fmaps_{fm_model}" in st.session_state:
                if btn_extract:
                    with st.spinner("Đang forward pass qua các lớp tích chập..."):
                        fmaps_list = extract_cnn_feature_maps(img, model_name=fm_model)
                        st.session_state[f"fmaps_{fm_model}"] = fmaps_list
                else:
                    fmaps_list = st.session_state[f"fmaps_{fm_model}"]

                for layer_info in fmaps_list:
                    st.markdown(f"**Lớp: {layer_info['layer_name']}**")
                    act = layer_info["activation"]
                    num_channels = min(8, act.shape[0])
                    cols = st.columns(num_channels)
                    for ch in range(num_channels):
                        ch_img = act[ch]
                        ch_norm = ((ch_img - ch_img.min()) / (ch_img.max() - ch_img.min() + 1e-7) * 255).astype(np.uint8)
                        cols[ch].image(ch_norm, caption=f"Ch {ch+1}", clamp=True, width=75)

    with tab_infer:
        st.markdown("**1. Chọn nguồn ảnh đầu vào:**")
        dl_src_mode = st.radio(
            "Nguồn ảnh:",
            ["Ảnh hiện tại (Dataset Explorer)", "Tải ảnh từ máy tính (User Upload)"],
            horizontal=True,
            key="tab8_src_mode"
        )

        col_dl_src_view, col_inf_ctrl, col_inf_res = st.columns([1, 1, 1.2])

        active_dl_img = None
        active_dl_meta = {}

        if dl_src_mode == "Ảnh hiện tại (Dataset Explorer)":
            active_dl_img = img
            active_dl_meta = meta
            with col_dl_src_view:
                st.markdown("**Ảnh đầu vào (Dataset):**")
                if active_dl_img is not None:
                    st.image(active_dl_img, width=210)
                    d_name = CLASS_DISPLAY_NAMES.get(meta.get("label"), meta.get("label", "N/A"))
                    st.caption(f"{meta.get('file_name', 'dataset')} ({d_name})")
                else:
                    st.warning("Chưa có ảnh được chọn.")
        else:
            with col_dl_src_view:
                st.markdown("**Tải ảnh từ máy tính:**")
                uploaded_file = st.file_uploader(
                    "Chọn tập tin ảnh:",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="tab8_upload_input",
                    label_visibility="collapsed"
                )
                if uploaded_file is not None:
                    try:
                        pil_img = Image.open(uploaded_file).convert("RGB")
                        w, h = pil_img.size
                        if max(w, h) > MAX_IMAGE_SIDE_PX:
                            scale = MAX_IMAGE_SIDE_PX / float(max(w, h))
                            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
                        active_dl_img = np.array(pil_img)
                        active_dl_meta = {
                            "source": "user_upload",
                            "label": "user_upload",
                            "file_name": uploaded_file.name,
                            "is_upload": True
                        }
                        st.image(active_dl_img, width=210)
                        st.caption(f"{uploaded_file.name} ({active_dl_img.shape[1]}x{active_dl_img.shape[0]} px)")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file ảnh: {e}")
                else:
                    st.info("Vui lòng tải lên 1 tệp ảnh lá cây.")

        with col_inf_ctrl:
            st.markdown("**2. Cấu hình mô hình:**")
            st.markdown("**Mô hình CNN:**")
            dl_model_name = st.selectbox(
                "Mô hình:",
                ["resnet18", "tinycnn"],
                format_func=lambda x: "ResNet-18 (Pretrained Fine-tuned)" if x == "resnet18" else "Classic TinyCNN (3 khối Conv)",
                key="tab8_dl_model_select",
                label_visibility="collapsed"
            )
            btn_dl_pred = st.button("Chạy dự đoán CNN", type="primary", disabled=(active_dl_img is None))

        with col_inf_res:
            st.markdown("**3. Kết quả phân loại:**")
            dl_cache_key = f"dl_pred_cache_{dl_src_mode}_{dl_model_name}"
            if (btn_dl_pred or dl_cache_key in st.session_state) and active_dl_img is not None:
                if btn_dl_pred:
                    with st.spinner("Đang suy luận bằng PyTorch CNN..."):
                        dl_res = predict_cnn(active_dl_img, model_name=dl_model_name)
                        st.session_state[dl_cache_key] = dl_res
                else:
                    dl_res = st.session_state[dl_cache_key]

                pred_display = CLASS_DISPLAY_NAMES.get(dl_res["label"], dl_res["label"])
                st.markdown(f"**Kết quả:** `{pred_display}`")

                if active_dl_meta.get("label") and active_dl_meta.get("label") != "user_upload":
                    actual_display = CLASS_DISPLAY_NAMES.get(active_dl_meta["label"], active_dl_meta["label"])
                    if active_dl_meta["label"] == dl_res["label"]:
                        st.success(f"Dự đoán đúng (Thực tế: {actual_display})")
                    else:
                        st.error(f"Dự đoán sai (Thực tế: {actual_display})")
                else:
                    st.info("Ảnh người dùng tải lên (chưa gắn nhãn nhị phân trước)")

                st.write(f"- **Độ trễ suy luận:** `{dl_res['latency_ms']} ms`")

                st.markdown("**Top-3 xác suất cao nhất:**")
                for rank, (cls_name, prob) in enumerate(dl_res["top_k"], 1):
                    d_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
                    st.progress(prob, text=f"{d_name}: {prob*100:.1f}%")
            elif active_dl_img is None:
                st.write("*(Chưa có ảnh đầu vào hợp lệ)*")
            else:
                st.write("*(Bấm 'Chạy dự đoán CNN' để xem kết quả)*")

    with tab_compare:
        st.markdown("**Bảng so sánh Deep Learning (CNN) vs Classical ML trên cùng một Test Set:**")
        dl_metrics_path = ARTIFACTS_DIR / "dl" / "dl_metrics.json"
        cmp_df = get_model_comparison_df_cached(str(dl_metrics_path), str(METRICS_PATH))
        if not cmp_df.empty:
            st.dataframe(cmp_df, use_container_width=True)
