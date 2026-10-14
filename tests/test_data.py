import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MANIFEST_PATH, TARGET_CLASSES, MIN_SAMPLES_PER_CLASS


def test_manifest_existence_and_structure():
    assert MANIFEST_PATH.exists(), "File manifest.csv phải tồn tại."
    df = pd.read_csv(MANIFEST_PATH)
    required_cols = {"image_id", "path", "label", "label_id", "split", "group_id", "width", "height", "sha1"}
    assert required_cols.issubset(set(df.columns)), f"Manifest thiếu các cột bắt buộc: {required_cols - set(df.columns)}"


def test_classes_and_sample_counts():
    df = pd.read_csv(MANIFEST_PATH)
    unique_labels = sorted(df["label"].unique().tolist())
    assert unique_labels == sorted(TARGET_CLASSES), "Danh sách nhãn trong manifest phải khớp TARGET_CLASSES."

    for label in TARGET_CLASSES:
        count = len(df[df["label"] == label])
        assert count >= MIN_SAMPLES_PER_CLASS, f"Lớp {label} có {count} mẫu, nhỏ hơn ngưỡng tối thiểu {MIN_SAMPLES_PER_CLASS}."


def test_no_data_leakage_between_splits():
    df = pd.read_csv(MANIFEST_PATH)
    train_groups = set(df[df["split"] == "train"]["group_id"])
    val_groups = set(df[df["split"] == "val"]["group_id"])
    test_groups = set(df[df["split"] == "test"]["group_id"])

    assert len(train_groups.intersection(val_groups)) == 0, "Rò rỉ nhóm giữa train và validation!"
    assert len(train_groups.intersection(test_groups)) == 0, "Rò rỉ nhóm giữa train và test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Rò rỉ nhóm giữa validation và test!"
