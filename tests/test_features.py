import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.features.shape import compute_shape_features
from core.features.hog_feat import compute_hog_features


def test_circularity_synthetic_circle():
    img_size = 300
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (150, 150)
    radius = 60
    cv2.circle(canvas, center, radius, 255, -1)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) > 0
    feats = compute_shape_features(contours[0])

    assert 0.85 <= feats["circularity"] <= 1.05, f"Hình tròn phải có độ tròn xấp xỉ 1.0, thực tế: {feats['circularity']}"
    assert 0.90 <= feats["aspect_ratio"] <= 1.10, f"Hình tròn phải có aspect ratio xấp xỉ 1.0, thực tế: {feats['aspect_ratio']}"


def test_feature_extraction_deterministic():
    rng = np.random.default_rng(42)
    sample_img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)

    feat1, _ = compute_hog_features(sample_img)
    feat2, _ = compute_hog_features(sample_img)

    np.testing.assert_array_equal(feat1, feat2, "Trích xuất đặc trưng trên cùng một ảnh phải cho kết quả giống hệt nhau.")
