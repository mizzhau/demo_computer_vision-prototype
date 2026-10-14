import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.imgproc.color import adjust_brightness_contrast, convert_color_space
from core.imgproc.geometry import flip_image, affine_transform
from core.imgproc.filters import add_gaussian_noise, compute_mse, compute_psnr
from core.evaluation.metrics import compute_iou_dice


def test_brightness_contrast_identity():
    sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    res = adjust_brightness_contrast(sample_img, alpha=1.0, beta=0.0)
    np.testing.assert_array_equal(sample_img, res, "Khi alpha=1, beta=0 thì ảnh ra phải bằng ảnh gốc.")


def test_flip_twice_identity():
    sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    flipped_1 = flip_image(sample_img, mode="horizontal")
    flipped_2 = flip_image(flipped_1, mode="horizontal")
    np.testing.assert_array_equal(sample_img, flipped_2, "Lật ngang 2 lần phải khôi phục ảnh ban đầu.")


def test_affine_identity():
    sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    transformed, mat = affine_transform(sample_img, angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0)
    expected_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(mat, expected_mat, atol=1e-5, err_msg="Ma trận xoay 0 độ phải là ma trận đơn vị 2x3.")


def test_noise_and_metrics_zero():
    sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    noisy = add_gaussian_noise(sample_img, sigma=0.0)
    mse = compute_mse(sample_img, noisy)
    psnr = compute_psnr(sample_img, noisy)
    assert mse == 0.0, "Với sigma=0, MSE phải bằng 0."
    assert psnr == float("inf"), "Với sigma=0, PSNR phải là vô cùng."


def test_iou_dice_perfect_overlap():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    iou, dice = compute_iou_dice(mask, mask)
    assert iou == 1.0, "Hai mask giống hệt nhau phải có IoU = 1.0"
    assert dice == 1.0, "Hai mask giống hệt nhau phải có Dice = 1.0"
