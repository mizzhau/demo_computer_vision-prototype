import cv2
import numpy as np


def apply_threshold(
    img_gray: np.ndarray,
    method: str = "otsu",
    threshold_value: int = 127,
    invert: bool = False
) -> tuple[np.ndarray, float]:
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method.lower() == "global":
        ret, mask = cv2.threshold(img_gray, threshold_value, 255, thresh_type)
        return mask, float(ret)
    elif method.lower() == "otsu":
        flag = thresh_type | cv2.THRESH_OTSU
        ret, mask = cv2.threshold(img_gray, 0, 255, flag)
        return mask, float(ret)
    elif method.lower() == "adaptive_mean":
        mask = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            thresh_type, 11, 2
        )
        return mask, 0.0
    elif method.lower() == "adaptive_gaussian":
        mask = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type, 11, 2
        )
        return mask, 0.0
    else:
        raise ValueError(f"Không hỗ trợ phương pháp phân ngưỡng: {method}")


def apply_morphology(
    mask: np.ndarray,
    op_name: str = "opening",
    kernel_shape: str = "ellipse",
    kernel_size: int = 5,
    iterations: int = 1
) -> np.ndarray:
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    k_shape = shape_map.get(kernel_shape.lower(), cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(k_shape, (kernel_size, kernel_size))

    op_map = {
        "erosion": cv2.MORPH_ERODE,
        "dilation": cv2.MORPH_DILATE,
        "opening": cv2.MORPH_OPEN,
        "closing": cv2.MORPH_CLOSE
    }
    op = op_map.get(op_name.lower(), cv2.MORPH_OPEN)
    return cv2.morphologyEx(mask, op, kernel, iterations=iterations)


def extract_foreground_leaf(
    img_rgb: np.ndarray, mask: np.ndarray, keep_largest_only: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_mask = mask.copy()

    if keep_largest_only:
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            filtered_mask = np.zeros_like(clean_mask)
            cv2.drawContours(filtered_mask, [largest], -1, 255, thickness=-1)
            clean_mask = filtered_mask

    foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=clean_mask)

    overlay = img_rgb.copy()
    colored_mask = np.zeros_like(img_rgb)
    colored_mask[clean_mask > 0] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    return clean_mask, foreground, overlay
