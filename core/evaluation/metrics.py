import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    p = (pred_mask > 0).astype(np.uint8)
    g = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2.0 * intersection) / (p.sum() + g.sum())) if (p.sum() + g.sum()) > 0 else 0.0

    return iou, dice


def compute_classification_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray, labels: list[str]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist()

    per_class = {}
    for idx, name in enumerate(labels):
        binary_true = (np.array(y_true) == idx).astype(int)
        binary_pred = (np.array(y_pred) == idx).astype(int)
        per_class[name] = {
            "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
            "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
            "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
            "support": int(binary_true.sum())
        }

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "per_class": per_class
    }
