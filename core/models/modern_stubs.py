import numpy as np


def run_clip_zero_shot(img_rgb: np.ndarray, prompt_template: str = "a photo of a {species} leaf") -> dict:
    pass


def run_sam2_segmentation(img_rgb: np.ndarray, point_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
    pass


def run_mobilevit_inference(img_rgb: np.ndarray) -> list[tuple[str, float]]:
    pass


def run_resnet18_baseline_inference(img_rgb: np.ndarray) -> list[tuple[str, float]]:
    pass
