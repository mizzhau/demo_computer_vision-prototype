import os
import sys
import time
import json
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

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, MANIFEST_PATH, ML_MODELS_DIR, METRICS_PATH, TARGET_CLASSES
from core.models.ml_pipeline import extract_features_from_image
from core.evaluation.metrics import compute_classification_metrics


def train_and_evaluate_ml():
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading manifest.csv...")
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest tại: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    feature_types = ["shape_color", "hog"]
    extracted_features = {}
    cache_dir = BASE_DIR / "data" / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for ftype in feature_types:
        train_cache_x = cache_dir / f"{ftype}_train_x.npy"
        train_cache_y = cache_dir / f"{ftype}_train_y.npy"
        test_cache_x = cache_dir / f"{ftype}_test_x.npy"
        test_cache_y = cache_dir / f"{ftype}_test_y.npy"

        if (
            train_cache_x.exists() and train_cache_y.exists() and
            test_cache_x.exists() and test_cache_y.exists()
        ):
            print(f"Loading cached features for {ftype}...")
            X_train = np.load(train_cache_x)
            y_train = np.load(train_cache_y)
            X_test = np.load(test_cache_x)
            y_test = np.load(test_cache_y)
        else:
            print(f"\n--- Extracting {ftype} features for Train & Test sets ---")
            X_train, y_train = [], []
            for idx, row in train_df.iterrows():
                img_path = BASE_DIR / row["path"]
                with Image.open(img_path) as pil_img:
                    img_rgb = np.array(pil_img.convert("RGB"))
                feat = extract_features_from_image(img_rgb, feature_type=ftype)
                X_train.append(feat)
                y_train.append(row["label_id"])

            X_test, y_test = [], []
            for idx, row in test_df.iterrows():
                img_path = BASE_DIR / row["path"]
                with Image.open(img_path) as pil_img:
                    img_rgb = np.array(pil_img.convert("RGB"))
                feat = extract_features_from_image(img_rgb, feature_type=ftype)
                X_test.append(feat)
                y_test.append(row["label_id"])

            X_train, y_train = np.array(X_train), np.array(y_train)
            X_test, y_test = np.array(X_test), np.array(y_test)

            np.save(train_cache_x, X_train)
            np.save(train_cache_y, y_train)
            np.save(test_cache_x, X_test)
            np.save(test_cache_y, y_test)

        extracted_features[ftype] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }
        print(f"Completed {ftype}: Shape = {extracted_features[ftype]['X_train'].shape}")

    models_config = {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_linear": SVC(kernel="linear", probability=True, random_state=42),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    all_metrics = {}

    for ftype in feature_types:
        data = extracted_features[ftype]
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_te, y_te = data["X_test"], data["y_test"]

        for algo_name, clf in models_config.items():
            model_key = f"{ftype}_{algo_name}"
            print(f"\nTraining [{model_key}]...")

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])

            t_train_start = time.perf_counter()
            pipeline.fit(X_tr, y_tr)
            t_train = time.perf_counter() - t_train_start

            t_infer_start = time.perf_counter()
            preds = pipeline.predict(X_te)
            t_infer = (time.perf_counter() - t_infer_start) / len(X_te) * 1000.0

            metrics = compute_classification_metrics(y_te, preds, TARGET_CLASSES)

            save_path = ML_MODELS_DIR / f"{model_key}.joblib"
            joblib.dump(pipeline, save_path)
            model_size_mb = float(save_path.stat().st_size / (1024 * 1024))

            all_metrics[model_key] = {
                "model_id": model_key,
                "feature_type": ftype,
                "algo_name": algo_name,
                "split": "test",
                "n_test": len(X_te),
                "accuracy": round(metrics["accuracy"], 4),
                "precision_macro": round(metrics["precision_macro"], 4),
                "recall_macro": round(metrics["recall_macro"], 4),
                "f1_macro": round(metrics["f1_macro"], 4),
                "infer_ms_per_image": round(t_infer, 2),
                "model_size_mb": round(model_size_mb, 2),
                "confusion_matrix": metrics["confusion_matrix"],
                "per_class": metrics["per_class"]
            }

            print(f"[{model_key}] Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | Latency: {t_infer:.2f} ms/img")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print(f"\nAll models saved to {ML_MODELS_DIR}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train_and_evaluate_ml()
