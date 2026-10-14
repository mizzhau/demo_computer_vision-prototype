import cv2
import numpy as np


def add_gaussian_noise(
    img_rgb: np.ndarray, sigma: float = 25.0, seed: int = 42
) -> np.ndarray:
    if sigma <= 0:
        return img_rgb.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img_rgb.shape).astype(np.float32)
    noisy = img_rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    img_rgb: np.ndarray, amount: float = 0.05, seed: int = 42
) -> np.ndarray:
    if amount <= 0:
        return img_rgb.copy()
    rng = np.random.default_rng(seed)
    noisy = img_rgb.copy()
    h, w = img_rgb.shape[:2]
    num_noisy = int(amount * h * w)

    y_coords = rng.integers(0, h, num_noisy // 2)
    x_coords = rng.integers(0, w, num_noisy // 2)
    noisy[y_coords, x_coords] = [255, 255, 255]

    y_coords = rng.integers(0, h, num_noisy // 2)
    x_coords = rng.integers(0, w, num_noisy // 2)
    noisy[y_coords, x_coords] = [0, 0, 0]

    return noisy


def apply_mean_filter(img_rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.blur(img_rgb, (k, k))


def apply_gaussian_blur(
    img_rgb: np.ndarray, kernel_size: int = 5, sigma: float = 1.5
) -> np.ndarray:
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.GaussianBlur(img_rgb, (k, k), sigmaX=sigma)


def apply_median_filter(img_rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(img_rgb, k)


def apply_bilateral_filter(
    img_rgb: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0
) -> np.ndarray:
    return cv2.bilateralFilter(img_rgb, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def compute_mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
    diff = img_a.astype(np.float64) - img_b.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    mse = compute_mse(img_a, img_b)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))
