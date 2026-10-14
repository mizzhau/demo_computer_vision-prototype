import cv2
import numpy as np


def compute_color_features(img_rgb: np.ndarray, mask: np.ndarray | None = None) -> dict:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    hist_h = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()

    norm_h = hist_h / (hist_h.sum() + 1e-7)
    norm_s = hist_s / (hist_s.sum() + 1e-7)
    norm_v = hist_v / (hist_v.sum() + 1e-7)

    if mask is not None and mask.sum() > 0:
        pixels_r = img_rgb[:, :, 0][mask > 0]
        pixels_g = img_rgb[:, :, 1][mask > 0]
        pixels_b = img_rgb[:, :, 2][mask > 0]
    else:
        pixels_r = img_rgb[:, :, 0].flatten()
        pixels_g = img_rgb[:, :, 1].flatten()
        pixels_b = img_rgb[:, :, 2].flatten()

    r_mean, r_std = float(np.mean(pixels_r)), float(np.std(pixels_r))
    g_mean, g_std = float(np.mean(pixels_g)), float(np.std(pixels_g))
    b_mean, b_std = float(np.mean(pixels_b)), float(np.std(pixels_b))

    combined_hist = np.concatenate([norm_h, norm_s, norm_v]).tolist()

    return {
        "hist_hsv_32d": combined_hist,
        "channel_stats": {
            "r_mean": r_mean, "r_std": r_std,
            "g_mean": g_mean, "g_std": g_std,
            "b_mean": b_mean, "b_std": b_std
        }
    }
