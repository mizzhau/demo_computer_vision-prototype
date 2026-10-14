import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import TARGET_CLASSES, ML_MODELS_DIR
from core.models.ml_pipeline import predict_single_image
from core.models.dl_stubs import apply_manual_convolution, predict_cnn
from core.models.modern_stubs import run_clip_zero_shot


def test_ml_prediction_schema_and_probabilities():
    rng = np.random.default_rng(42)
    dummy_img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    model_files = list(ML_MODELS_DIR.glob("*.joblib"))
    assert len(model_files) > 0, "Phải có ít nhất một model joblib đã huấn luyện."

    res = predict_single_image(dummy_img, feature_type="shape_color", algo_name="rf")

    assert "label" in res and res["label"] in TARGET_CLASSES
    assert "label_id" in res and 0 <= res["label_id"] < len(TARGET_CLASSES)
    assert "probs" in res and len(res["probs"]) == len(TARGET_CLASSES)
    assert "latency_ms" in res and res["latency_ms"] >= 0.0

    prob_sum = sum(res["probs"].values())
    assert abs(prob_sum - 1.0) < 1e-3, f"Tổng xác suất phải bằng 1.0, thực tế: {prob_sum}"


def test_deep_learning_tab8_and_tab9_stub():
    dummy_gray = np.zeros((100, 100), dtype=np.uint8)
    dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_kernel = np.ones((3, 3), dtype=np.float32)

    # Tab 8 convolution
    conv_out, pooled_out = apply_manual_convolution(dummy_gray, dummy_kernel)
    assert conv_out is not None and isinstance(conv_out, np.ndarray)
    assert pooled_out is not None and isinstance(pooled_out, np.ndarray)

    # Tab 8 prediction (TinyCNN & ResNet-18)
    res_tiny = predict_cnn(dummy_rgb, model_name="tinycnn")
    assert res_tiny["label"] in TARGET_CLASSES
    assert "probs" in res_tiny and len(res_tiny["probs"]) == len(TARGET_CLASSES)

    # Tab 9 stub remains pass
    assert run_clip_zero_shot(dummy_rgb) is None, "Stub Tab 9 phải trả về None (pass)."
