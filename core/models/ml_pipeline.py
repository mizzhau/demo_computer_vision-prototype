import time
from pathlib import Path
import cv2
import joblib
import numpy as np

from core.imgproc.color import compute_histogram
from core.imgproc.segment import apply_threshold, apply_morphology, extract_foreground_leaf
from core.features.shape import extract_largest_contour, compute_shape_features
from core.features.color_hist import compute_color_features
from core.features.hog_feat import compute_hog_features
from config import ARTIFACTS_DIR, ML_MODELS_DIR, CONFIDENCE_THRESHOLD, TARGET_CLASSES


def extract_features_from_image(img_rgb: np.ndarray, feature_type: str = "shape_color") -> np.ndarray:
    if feature_type == "hog":
        feat, _ = compute_hog_features(img_rgb)
        return feat

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask_otsu, _ = apply_threshold(gray, method="otsu")
    mask_clean = apply_morphology(mask_otsu, op_name="closing", kernel_size=5)
    clean_mask, _, _ = extract_foreground_leaf(img_rgb, mask_clean)

    contour = extract_largest_contour(clean_mask)
    shape_dict = compute_shape_features(contour)
    color_dict = compute_color_features(img_rgb, mask=clean_mask)

    geom_vec = [
        shape_dict["area"],
        shape_dict["perimeter"],
        shape_dict["aspect_ratio"],
        shape_dict["extent"],
        shape_dict["solidity"],
        shape_dict["circularity"],
    ] + shape_dict["hu_moments"]

    color_vec = color_dict["hist_hsv_32d"] + [
        color_dict["channel_stats"]["r_mean"],
        color_dict["channel_stats"]["r_std"],
        color_dict["channel_stats"]["g_mean"],
        color_dict["channel_stats"]["g_std"],
        color_dict["channel_stats"]["b_mean"],
        color_dict["channel_stats"]["b_std"],
    ]

    return np.array(geom_vec + color_vec, dtype=np.float32)


import functools


@functools.lru_cache(maxsize=16)
def load_ml_model(feature_type: str, algo_name: str):
    model_path = ML_MODELS_DIR / f"{feature_type}_{algo_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
    return joblib.load(model_path)


def predict_single_image(
    img_rgb: np.ndarray,
    feature_type: str = "shape_color",
    algo_name: str = "rf"
) -> dict:
    t0 = time.perf_counter()
    pipeline = load_ml_model(feature_type, algo_name)
    feat_vec = extract_features_from_image(img_rgb, feature_type=feature_type)
    feat_vec_2d = feat_vec.reshape(1, -1)

    if hasattr(pipeline, "predict_proba"):
        probs_array = pipeline.predict_proba(feat_vec_2d)[0]
    else:
        pred_label_id = pipeline.predict(feat_vec_2d)[0]
        probs_array = np.zeros(len(TARGET_CLASSES))
        probs_array[pred_label_id] = 1.0

    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(probs_array))
    pred_label = TARGET_CLASSES[pred_id]
    max_prob = float(probs_array[pred_id])

    probs_dict = {
        cls_name: float(round(p, 4))
        for cls_name, p in zip(TARGET_CLASSES, probs_array)
    }

    top_indices = np.argsort(probs_array)[::-1][:3]
    top_k = [[TARGET_CLASSES[i], float(round(probs_array[i], 4))] for i in top_indices]

    return {
        "model_id": f"{algo_name}_{feature_type}",
        "label": pred_label,
        "label_id": pred_id,
        "probs": probs_dict,
        "top_k": top_k,
        "uncertain": max_prob < CONFIDENCE_THRESHOLD,
        "latency_ms": float(round(latency_ms, 2))
    }
