import time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np

from config import ARTIFACTS_DIR, TARGET_CLASSES, CONFIDENCE_THRESHOLD
from scripts.train_dl import TinyCNN


DL_DIR = ARTIFACTS_DIR / "dl"
DEVICE = torch.device("cpu")

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


import functools


@functools.lru_cache(maxsize=4)
def load_dl_model(model_name: str = "resnet18") -> nn.Module:
    if model_name == "tinycnn":
        model = TinyCNN(num_classes=len(TARGET_CLASSES))
        pt_path = DL_DIR / "tinycnn_leaf.pt"
        if pt_path.exists():
            model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    else:
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(TARGET_CLASSES))
        pt_path = DL_DIR / "resnet18_leaf.pt"
        if pt_path.exists():
            model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model


def apply_manual_convolution(img_gray: np.ndarray, kernel: np.ndarray, pool_type: str = "max") -> tuple[np.ndarray, np.ndarray]:
    import cv2
    convolved = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
    convolved = np.clip(convolved, 0, 255).astype(np.uint8)

    h, w = convolved.shape
    new_h, new_w = h // 2, w // 2
    interp = cv2.INTER_AREA if pool_type == "avg" else cv2.INTER_NEAREST
    pooled = cv2.resize(convolved, (new_w, new_h), interpolation=interp)
    return convolved, pooled


def extract_cnn_feature_maps(img_rgb: np.ndarray, model_name: str = "tinycnn") -> list[dict]:
    model = load_dl_model(model_name)
    pil_img = Image.fromarray(img_rgb)
    tensor = _TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    feature_maps = []
    hooks = []

    def get_hook(name):
        def hook(m, i, o):
            feature_maps.append({
                "layer_name": name,
                "activation": o.detach().cpu().numpy()[0]
            })
        return hook

    if model_name == "tinycnn":
        hooks.append(model.conv_block1.register_forward_hook(get_hook("Conv Block 1 (16 channels)")))
        hooks.append(model.conv_block2.register_forward_hook(get_hook("Conv Block 2 (32 channels)")))
        hooks.append(model.conv_block3.register_forward_hook(get_hook("Conv Block 3 (64 channels)")))
    else:
        hooks.append(model.conv1.register_forward_hook(get_hook("Initial Conv1 (64 channels)")))
        hooks.append(model.layer1.register_forward_hook(get_hook("ResNet Layer 1 (64 channels)")))
        hooks.append(model.layer2.register_forward_hook(get_hook("ResNet Layer 2 (128 channels)")))

    with torch.no_grad():
        _ = model(tensor)

    for h in hooks:
        h.remove()

    return feature_maps


def predict_cnn(img_rgb: np.ndarray, model_name: str = "resnet18") -> dict:
    t0 = time.perf_counter()
    model = load_dl_model(model_name)
    pil_img = Image.fromarray(img_rgb)
    tensor = _TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    pred_id = int(np.argmax(probs))
    pred_label = TARGET_CLASSES[pred_id]
    max_prob = float(probs[pred_id])

    probs_dict = {
        cls_name: float(round(float(p), 4))
        for cls_name, p in zip(TARGET_CLASSES, probs)
    }
    top_indices = np.argsort(probs)[::-1][:3]
    top_k = [[TARGET_CLASSES[i], float(round(float(probs[i]), 4))] for i in top_indices]

    return {
        "model_id": model_name,
        "label": pred_label,
        "label_id": pred_id,
        "probs": probs_dict,
        "top_k": top_k,
        "uncertain": max_prob < CONFIDENCE_THRESHOLD,
        "latency_ms": float(round(latency_ms, 2))
    }
