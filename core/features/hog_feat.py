import cv2
import numpy as np
from skimage.feature import hog


def compute_hog_features(
    img_rgb: np.ndarray, target_size: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    fd, hog_image = hog(
        resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True
    )

    hog_image_rescaled = (hog_image / (hog_image.max() + 1e-7) * 255).astype(np.uint8)
    return fd, hog_image_rescaled
