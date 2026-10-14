import cv2
import numpy as np


def extract_largest_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def compute_shape_features(contour: np.ndarray | None) -> dict:
    if contour is None or len(contour) < 3:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "aspect_ratio": 0.0,
            "extent": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "hu_moments": [0.0] * 7,
            "bbox": [0, 0, 0, 0]
        }

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    moments = cv2.moments(contour)
    raw_hu = cv2.HuMoments(moments).flatten()
    log_hu = []
    for h_val in raw_hu:
        if h_val != 0:
            val = -1.0 * np.copysign(1.0, h_val) * np.log10(abs(h_val))
        else:
            val = 0.0
        log_hu.append(float(val))

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "hu_moments": log_hu,
        "bbox": [x, y, w, h]
    }


def draw_geometric_overlays(
    img_rgb: np.ndarray, contour: np.ndarray | None
) -> np.ndarray:
    canvas = img_rgb.copy()
    if contour is None:
        return canvas

    cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)

    hull = cv2.convexHull(contour)
    cv2.drawContours(canvas, [hull], -1, (255, 255, 0), 2)

    if len(contour) >= 5:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(canvas, [box], 0, (255, 0, 255), 2)

    return canvas
