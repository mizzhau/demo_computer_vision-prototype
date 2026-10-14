import cv2
import numpy as np


def apply_sobel(img_gray: np.ndarray, ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    abs_x = np.clip(np.abs(sobel_x), 0, 255).astype(np.uint8)
    abs_y = np.clip(np.abs(sobel_y), 0, 255).astype(np.uint8)
    abs_mag = np.clip(magnitude, 0, 255).astype(np.uint8)

    return abs_x, abs_y, abs_mag


def apply_laplacian(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=ksize)
    return np.clip(np.abs(lap), 0, 255).astype(np.uint8)


def apply_canny(
    img_gray: np.ndarray, threshold1: int = 50, threshold2: int = 150
) -> np.ndarray:
    return cv2.Canny(img_gray, threshold1, threshold2)
