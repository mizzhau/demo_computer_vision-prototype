import cv2
import numpy as np


def convert_color_space(img_rgb: np.ndarray, space: str) -> np.ndarray:
    if space.upper() == "RGB":
        return img_rgb
    elif space.upper() == "BGR":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif space.upper() == "GRAY":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    elif space.upper() == "HSV":
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    else:
        raise ValueError(f"Không hỗ trợ không gian màu: {space}")


def adjust_brightness_contrast(
    img_rgb: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    adjusted = alpha * img_rgb.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def compute_histogram(img_rgb: np.ndarray) -> dict:
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    return {
        "r": hist_r,
        "g": hist_g,
        "b": hist_b,
        "gray": hist_gray
    }


def apply_clahe(
    img_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
    )
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


def apply_hist_equalization(img_rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


def extract_pixel_neighborhood(
    img_rgb: np.ndarray, x: int, y: int, size: int = 5
) -> dict:
    h, w = img_rgb.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    half = size // 2
    y_min, y_max = max(0, y - half), min(h, y + half + 1)
    x_min, x_max = max(0, x - half), min(w, x + half + 1)

    region_rgb = img_rgb[y_min:y_max, x_min:x_max]
    bgr_val = img_rgb[y, x][::-1].tolist()
    rgb_val = img_rgb[y, x].tolist()

    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hsv_val = hsv_img[y, x].tolist()

    return {
        "x": x,
        "y": y,
        "rgb": rgb_val,
        "bgr": bgr_val,
        "hsv": hsv_val,
        "region_rgb": region_rgb
    }
