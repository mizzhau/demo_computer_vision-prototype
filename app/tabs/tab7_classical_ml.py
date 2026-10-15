import streamlit as st
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image

from config import METRICS_PATH, TARGET_CLASSES, CLASS_DISPLAY_NAMES, MAX_IMAGE_SIDE_PX
from core.models.ml_pipeline import predict_single_image
from core.imgproc.segment import apply_threshold, apply_morphology, extract_foreground_leaf
from core.features.shape import extract_largest_contour, draw_geometric_overlays
from core.features.hog_feat import compute_hog_features
from app.components.metrics_view import plot_confusion_matrix, display_classification_summary
from app.state import require_active_image


def show_preprocessing_pipeline(img_rgb: np.ndarray, feature_type: str):
    st.markdown("**Các bước xử lý ảnh trước khi đưa vào mô hình:**")
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if feature_type == "shape_color":
        mask_otsu, th_val = apply_threshold(gray, method="otsu")
        mask_clean = apply_morphology(mask_otsu, op_name="closing", kernel_size=5)
        clean_mask, foreground, _ = extract_foreground_leaf(img_rgb, mask_clean)
        contour = extract_largest_contour(clean_mask)
        geom_overlay = draw_geometric_overlays(img_rgb, contour)

        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_rgb, caption="1. Ảnh gốc RGB", width=180)
        c2.image(gray, caption="2. Mức xám (Grayscale)", clamp=True, width=180)
        c3.image(mask_clean, caption=f"3. Mặt nạ Otsu (Ngưỡng {th_val:.0f})", clamp=True, width=180)
        c4.image(geom_overlay, caption="4. Contour & Bounding Box", width=180)

    else:
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        _, hog_img = compute_hog_features(img_rgb)

        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_rgb, caption="1. Ảnh gốc RGB", width=180)
        c2.image(gray, caption="2. Mức xám (Grayscale)", clamp=True, width=180)
        c3.image(resized, caption="3. Resize chuẩn 128x128", clamp=True, width=180)
        c4.image(hog_img, caption="4. Bản đồ Gradient HOG", width=180)


@st.cache_data
def load_ml_metrics_cached(path_str: str) -> dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def render():
    st.subheader("Classical Machine Learning")

    img = st.session_state.get("input_image")
    meta = st.session_state.get("input_meta", {})
    if img is None:
        st.warning("Vui lòng chọn hoặc tải ảnh tại phần dataset.")
        return

    if not METRICS_PATH.exists():
        st.error("Chưa có metrics.json. Vui lòng chạy script scripts/train_ml.py trước.")
        return

    metrics_all = load_ml_metrics_cached(str(METRICS_PATH))

    tab_eval, tab_predict, tab_compare = st.tabs([
        "Đánh giá mô hình", "Dự đoán ảnh hiện tại", "So sánh toàn bộ"
    ])

    with tab_eval:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Bộ đặc trưng đầu vào:**")
            ftype = st.selectbox(
                "Bộ đặc trưng đầu vào:",
                ["shape_color", "hog"],
                format_func=lambda x: "Hình học + Màu sắc (Shape & Color 51D)" if x == "shape_color" else "HOG (1764D)",
                label_visibility="collapsed"
            )
        with col_c2:
            st.markdown("**Thuật toán phân loại:**")
            algo = st.selectbox(
                "Thuật toán phân loại:",
                ["rf", "svm_rbf", "svm_linear", "knn"],
                format_func=lambda x: {
                    "rf": "Random Forest",
                    "svm_rbf": "SVM (RBF Kernel)",
                    "svm_linear": "SVM (Linear)",
                    "knn": "KNN (k=5)"
                }.get(x, x),
                label_visibility="collapsed"
            )

        model_key = f"{ftype}_{algo}"
        model_metrics = metrics_all.get(model_key)

        if model_metrics:
            display_classification_summary(model_metrics)
            st.write("")
            labels_display = [CLASS_DISPLAY_NAMES.get(c, c) for c in TARGET_CLASSES]
            plot_confusion_matrix(model_metrics["confusion_matrix"], labels_display)

    with tab_predict:
        st.markdown("**1. Chọn nguồn ảnh đầu vào:**")
        src_mode = st.radio(
            "Nguồn ảnh:",
            ["Ảnh hiện tại (Dataset Explorer)", "Tải ảnh từ máy tính (User Upload)"],
            horizontal=True,
            key="tab7_src_mode"
        )

        col_src_view, col_p_ctrl, col_p_res = st.columns([1, 1, 1.2])

        active_pred_img = None
        active_pred_meta = {}

        if src_mode == "Ảnh hiện tại (Dataset Explorer)":
            active_pred_img = img
            active_pred_meta = meta
            with col_src_view:
                st.markdown("**Ảnh đầu vào (Dataset):**")
                if active_pred_img is not None:
                    st.image(active_pred_img, width=210)
                    d_name = CLASS_DISPLAY_NAMES.get(meta.get("label"), meta.get("label", "N/A"))
                    st.caption(f"{meta.get('file_name', 'dataset')} ({d_name})")
                else:
                    st.warning("Chưa có ảnh được chọn.")
        else:
            with col_src_view:
                st.markdown("**Tải ảnh từ máy tính:**")
                uploaded_file = st.file_uploader(
                    "Chọn tập tin ảnh:",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="tab7_upload_input",
                    label_visibility="collapsed"
                )
                if uploaded_file is not None:
                    try:
                        pil_img = Image.open(uploaded_file).convert("RGB")
                        w, h = pil_img.size
                        if max(w, h) > MAX_IMAGE_SIDE_PX:
                            scale = MAX_IMAGE_SIDE_PX / float(max(w, h))
                            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
                        active_pred_img = np.array(pil_img)
                        active_pred_meta = {
                            "source": "user_upload",
                            "label": "user_upload",
                            "file_name": uploaded_file.name,
                            "is_upload": True
                        }
                        st.image(active_pred_img, width=210)
                        st.caption(f"{uploaded_file.name} ({active_pred_img.shape[1]}x{active_pred_img.shape[0]} px)")
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file ảnh: {e}")
                else:
                    st.info("Vui lòng tải lên 1 tệp ảnh lá cây.")

        with col_p_ctrl:
            st.markdown("**2. Cấu hình mô hình:**")
            st.markdown("**Đặc trưng:**")
            p_ftype = st.selectbox("Đặc trưng:", ["shape_color", "hog"], key="p_ftype", label_visibility="collapsed")
            st.markdown("**Thuật toán:**")
            p_algo = st.selectbox(
                "Thuật toán:",
                ["rf", "svm_rbf", "svm_linear", "knn"],
                format_func=lambda x: {
                    "rf": "Random Forest",
                    "svm_rbf": "SVM (RBF Kernel)",
                    "svm_linear": "SVM (Linear)",
                    "knn": "KNN (k=5)"
                }.get(x, x),
                key="p_algo",
                label_visibility="collapsed"
            )
            btn_run = st.button("Chạy dự đoán", type="primary")
            if btn_run and active_pred_img is None:
                require_active_image()

        with col_p_res:
            st.markdown("**3. Kết quả phân loại:**")
            cache_key = f"last_pred_tab7_{src_mode}"
            if (btn_run or cache_key in st.session_state) and active_pred_img is not None:
                if btn_run:
                    with st.spinner("Đang trích xuất đặc trưng và dự đoán..."):
                        pred_res = predict_single_image(active_pred_img, feature_type=p_ftype, algo_name=p_algo)
                        st.session_state[cache_key] = pred_res
                else:
                    pred_res = st.session_state[cache_key]

                pred_display = CLASS_DISPLAY_NAMES.get(pred_res["label"], pred_res["label"])
                st.markdown(f"**Kết quả:** `{pred_display}`")

                if active_pred_meta.get("label") and active_pred_meta.get("label") != "user_upload":
                    actual_display = CLASS_DISPLAY_NAMES.get(active_pred_meta["label"], active_pred_meta["label"])
                    if active_pred_meta["label"] == pred_res["label"]:
                        st.success(f"Dự đoán đúng (Thực tế: {actual_display})")
                    else:
                        st.error(f"Dự đoán sai (Thực tế: {actual_display})")
                else:
                    st.info("Ảnh người dùng tải lên (chưa gắn nhãn nhị phân trước)")

                st.write(f"- **Độ trễ suy luận:** `{pred_res['latency_ms']} ms`")

                st.markdown("**Top-3 xác suất cao nhất:**")
                for rank, (cls_name, prob) in enumerate(pred_res["top_k"], 1):
                    d_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
                    st.progress(prob, text=f"{d_name}: {prob*100:.1f}%")
            elif active_pred_img is None:
                st.write("*(Chưa có ảnh đầu vào hợp lệ)*")
            else:
                st.write("*(Bấm 'Chạy dự đoán' để xem kết quả)*")

        if active_pred_img is not None:
            st.divider()
            show_preprocessing_pipeline(active_pred_img, p_ftype)

    with tab_compare:
        cmp_rows = []
        for k, v in metrics_all.items():
            cmp_rows.append({
                "Mô hình": k,
                "Đặc trưng": "Shape + Color" if v["feature_type"] == "shape_color" else "HOG",
                "Thuật toán": v["algo_name"].upper(),
                "Accuracy": f"{v['accuracy']*100:.2f}%",
                "Macro F1": f"{v['f1_macro']*100:.2f}%",
                "Inference (ms/ảnh)": v["infer_ms_per_image"],
                "Kích thước (MB)": v["model_size_mb"]
            })
        cmp_df = pd.DataFrame(cmp_rows).sort_values(by="Accuracy", ascending=False)
        st.dataframe(cmp_df, use_container_width=True)
