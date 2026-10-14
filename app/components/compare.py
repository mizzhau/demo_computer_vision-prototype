import streamlit as st
import numpy as np


def image_compare_view(
    before_img: np.ndarray,
    after_img: np.ndarray,
    before_caption: str = "Ảnh gốc",
    after_caption: str = "Ảnh kết quả"
):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{before_caption}**")
        st.image(before_img, width=320)
    with col2:
        st.markdown(f"**{after_caption}**")
        st.image(after_img, width=320)
