import cv2
import numpy as np


def resize_image(
    img_rgb: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: str = "bilinear"
) -> np.ndarray:
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }
    flag = interp_map.get(interpolation.lower(), cv2.INTER_LINEAR)
    return cv2.resize(img_rgb, (target_width, target_height), interpolation=flag)


def crop_image(
    img_rgb: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    img_h, img_w = img_rgb.shape[:2]
    x = max(0, min(img_w - 1, x))
    y = max(0, min(img_h - 1, y))
    w = max(1, min(img_w - x, w))
    h = max(1, min(img_h - y, h))
    return img_rgb[y:y + h, x:x + w]


def flip_image(img_rgb: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    if mode == "horizontal":
        return cv2.flip(img_rgb, 1)
    elif mode == "vertical":
        return cv2.flip(img_rgb, 0)
    elif mode == "both":
        return cv2.flip(img_rgb, -1)
    return img_rgb


def affine_transform(
    img_rgb: np.ndarray,
    angle_deg: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    center = (w / 2.0, h / 2.0)

    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty

    transformed = cv2.warpAffine(
        img_rgb, rot_mat, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return transformed, rot_mat


def perspective_transform(
    img_rgb: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    matrix = cv2.getPerspectiveTransform(
        src_points.astype(np.float32), dst_points.astype(np.float32)
    )
    warped = cv2.warpPerspective(
        img_rgb, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return warped, matrix


def generate_augmentations(
    img_rgb: np.ndarray, n: int = 8, seed: int = 42
) -> list[tuple[np.ndarray, str]]:
    rng = np.random.default_rng(seed)
    h, w = img_rgb.shape[:2]
    augmented_list = []

    for i in range(n):
        angle = rng.uniform(-30, 30)
        scale = rng.uniform(0.85, 1.15)
        tx = rng.uniform(-15, 15)
        ty = rng.uniform(-15, 15)

        transformed, _ = affine_transform(img_rgb, angle, scale, tx, ty)

        if rng.random() > 0.5:
            transformed = cv2.flip(transformed, 1)

        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-15, 15)
        adjusted = np.clip(alpha * transformed.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        desc = f"Aug #{i+1}: rot={angle:.1f}°, sc={scale:.2f}"
        augmented_list.append((adjusted, desc))

    return augmented_list
