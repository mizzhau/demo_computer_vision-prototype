import streamlit as st
import numpy as np


def display_matrix(matrix: np.ndarray, title: str = "Ma trận biến đổi"):
    st.write(f"**{title}:**")
    formatted_rows = []
    for row in matrix:
        formatted_rows.append(" & ".join([f"{val:.4f}" for val in row]))
    latex_str = r"\begin{bmatrix}" + r" \\ ".join(formatted_rows) + r"\end{bmatrix}"
    st.latex(latex_str)
