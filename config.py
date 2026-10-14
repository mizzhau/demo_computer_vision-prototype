import os
import sys
from pathlib import Path

try:
    import threadpoolctl
    orig_find = threadpoolctl.ThreadpoolController._find_libraries_on_windows

    def safe_find(self):
        try:
            orig_find(self)
        except Exception:
            pass

    threadpoolctl.ThreadpoolController._find_libraries_on_windows = safe_find
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_COLOR_DIR = BASE_DIR / "dataset" / "raw" / "color"
RAW_SEG_DIR = BASE_DIR / "dataset" / "raw" / "segmented"
SUBSET_DIR = DATA_DIR / "plantvillage_subset"
SAMPLE_DIR = DATA_DIR / "sample"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ML_MODELS_DIR = ARTIFACTS_DIR / "ml"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 350
MIN_SAMPLES_PER_CLASS = 300

CONFIDENCE_THRESHOLD = 0.50
MAX_UPLOAD_SIZE_MB = 5
MAX_IMAGE_SIDE_PX = 1024

TARGET_CLASSES = [
    "Apple___healthy",
    "Blueberry___healthy",
    "Corn_(maize)___healthy",
    "Grape___healthy",
    "Pepper,_bell___healthy",
    "Tomato___healthy"
]

CLASS_DISPLAY_NAMES = {
    "Apple___healthy": "Táo (Apple)",
    "Blueberry___healthy": "Việt quất (Blueberry)",
    "Corn_(maize)___healthy": "Bắp / Ngô (Corn)",
    "Grape___healthy": "Nho (Grape)",
    "Pepper,_bell___healthy": "Ớt chuông (Pepper bell)",
    "Tomato___healthy": "Cà chua (Tomato)"
}
