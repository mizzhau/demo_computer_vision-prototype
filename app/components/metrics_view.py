import streamlit as st
import plotly.express as px
import pandas as pd


def plot_confusion_matrix(cm_data: list, labels: list[str]):
    fig = px.imshow(
        cm_data,
        x=labels,
        y=labels,
        color_continuous_scale="Greens",
        text_auto=True,
        labels=dict(x="Dự đoán (Predicted)", y="Thực tế (Actual)", color="Số lượng")
    )
    fig.update_layout(
        title="Confusion Matrix (Ma trận nhầm lẫn trên Test Set)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


def display_classification_summary(metrics_dict: dict):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics_dict.get('accuracy', 0)*100:.2f}%")
    with col2:
        st.metric("Macro F1", f"{metrics_dict.get('f1_macro', 0)*100:.2f}%")
    with col3:
        st.metric("Inference/Ảnh", f"{metrics_dict.get('infer_ms_per_image', 0):.2f} ms")
    with col4:
        st.metric("Kích thước Model", f"{metrics_dict.get('model_size_mb', 0):.2f} MB")
