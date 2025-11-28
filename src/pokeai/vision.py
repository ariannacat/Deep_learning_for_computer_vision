"""
Vision utilities for Pokémon species recognition.

This module is a cleaned-up version of your notebook code, focused ONLY on
inference (no training, no evaluation reports).

It supports:
- Torchvision models (ResNet, VGG, etc.) with CV-ensemble across folds
- YOLOv8-CLS models with CV-ensemble across folds

Main entry:

    pred_name, conf = predict_image("crop_own_sprite.png")

Requirements:
- You have already trained models and saved:
  - artifacts/classes.txt              (one label per line)
  - artifacts/best_<MODEL_NAME_SAFE>_foldK.pth
    OR YOLO runs under artifacts/yolo_cls_runs/
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Union, Optional
from .models import make_model, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .constants import DEVICE, ARTIFACTS, DATA_DIR
from pokeai.config import load_config
from .io_utils import load_classes_txt

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  
    YOLO = None

# ============================================================
# Class vocabulary from artifacts/classes.txt
# ============================================================

CLASSES: List[str] = load_classes_txt(ARTIFACTS / "classes.txt")
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ============================================================
# Torchvision transforms 
# ============================================================

eval_tfms = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ============================================================
# Checkpoint paths 
# ============================================================

def fold_name(fold_idx: int, model_name_safe) -> str:
    return f"{model_name_safe}_fold{fold_idx}"

def fold_ckpt_path(fold_idx: int, model_name_safe) -> Path:
    """
    Path of the per-fold checkpoint for Torchvision models.
    Matches your notebook:

        artifacts/best_<model_name_safe>_foldK.pth
    """
    return ARTIFACTS / f"best_{fold_name(fold_idx, model_name_safe)}.pth"

# ============================================================
# Helper: resolve input image path
# ============================================================

def _resolve_image_path(path_or_rel: Union[str, Path]) -> Path:
    """
    Try:
      1) interpret as a direct path
      2) if not found, prepend DATASET_DIR (for dataset images)
    """
    p = Path(path_or_rel)
    if p.is_file():
        return p

    candidate = DATA_DIR / path_or_rel
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Image not found: {path_or_rel} "
        f"(tried '{p}' and '{candidate}')"
    )

# ============================================================
# Main API: predict_image
# ============================================================

def predict_image(path_or_rel: Union[str, Path], cfg: Dict[str, Any]) -> Tuple[str, float]:
    """
    Predict a single image class using config-driven model settings.

    cfg["model"] is expected to contain:
      - backend: "torchvision" or "yolo"
      - name: model name, e.g. "resnet50" or "yolov8n-cls"
      - folds: number of folds (int)
      - yolo_name (optional): base name for YOLO runs, default "pokemon_yolo_cls"
    """
    img_path = _resolve_image_path(path_or_rel)
    
    model_cfg = cfg.get("model", {})
    MODEL_NAME = model_cfg["name"]              
    N_FOLDS = int(model_cfg["folds"])           
    
    USE_YOLO = USE_YOLO = model_cfg["backend"] == "yolo"

    YOLO_NAME = model_cfg.get("yolo_name", "pokemon_yolo_cls")

    if USE_YOLO:
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO is not installed. "
                "Install with: pip install ultralytics"
            )
        return _predict_yolo_ensemble(img_path, YOLO_NAME, N_FOLDS)

    return _predict_torchvision_ensemble(img_path, MODEL_NAME, N_FOLDS)

# ============================================================
# Torchvision CV-ensemble prediction
# ============================================================

def _predict_torchvision_ensemble(img_path: Path, model_name: str, n_folds: int) -> Tuple[str, float]:
    """
    Mirror of your notebook logic:
    - apply eval_tfms
    - for each fold:
        - build model with make_model(MODEL_NAME, len(CLASSES), pretrained=False)
        - load fold checkpoint
        - compute logits
    - average logits across folds
    - apply softmax to get confidence
    """
    img = Image.open(img_path).convert("RGB")
    x = eval_tfms(img).unsqueeze(0).to(DEVICE)

    model_name_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)

    logits_list = []

    for fold_id in range(1, n_folds + 1):
        ckpt_path = fold_ckpt_path(fold_id, model_name_safe)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold_id}: {ckpt_path}"
            )

        model = make_model(model_name, len(CLASSES), pretrained=False).to(DEVICE).eval()
        sd = torch.load(ckpt_path, map_location=DEVICE)
        # In your test code you loaded directly with strict=True
        model.load_state_dict(sd, strict=True)

        with torch.no_grad():
            logits = model(x)
        logits_list.append(logits.cpu().numpy())

    # Average logits across folds: [N=1, C]
    logits_mean = np.mean(np.stack(logits_list, axis=0), axis=0)
    pred_idx = int(np.argmax(logits_mean))

    # Compute confidence via softmax
    logits_tensor = torch.tensor(logits_mean)
    probs = torch.softmax(logits_tensor, dim=1)[0]
    conf = float(probs[pred_idx])

    pred_name = IDX_TO_CLASS.get(pred_idx, f"class_{pred_idx}")
    return pred_name, conf


# ============================================================
# YOLOv8-CLS CV-ensemble prediction
# ============================================================

def _best_yolo_weights_for_fold(fold_id: int, yolo_name: str) -> Path:
    """
    Find YOLO best.pt for a given fold, mirroring your naming:

        run_name = f"{YOLO_NAME}_fold{fold_id}"
        YOLO_RUNS / run_name / "weights" / "best.pt"
    """
    YOLO_RUNS = ARTIFACTS / "yolo_cls_runs"
    run_name = f"{yolo_name}_fold{fold_id}"
    direct = YOLO_RUNS / run_name / "weights" / "best.pt"
    if direct.is_file():
        return direct

    candidates = list(YOLO_RUNS.glob(f"{run_name}*/weights/best*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find YOLO best weights for fold {fold_id}. "
            f"Tried: {direct} and {YOLO_RUNS}/{run_name}*/weights/best*.pt"
        )
    return candidates[0]


def _predict_yolo_ensemble(img_path: Path, yolo_name: str, n_folds: int) -> Tuple[str, float]:
    """
    Mirror of your notebook's YOLO ensemble logic:
    - for each fold, load YOLO best.pt
    - get probabilities for the image
    - average probabilities across folds
    - take argmax + confidence
    """
    if YOLO is None:
        raise ImportError("Ultralytics YOLO is not installed.")

    probs_list = []

    for fold_id in range(1, n_folds + 1):
        best_weights = _best_yolo_weights_for_fold(fold_id, yolo_name)
        yolo_model = YOLO(str(best_weights))
        r = yolo_model.predict(str(img_path), verbose=False)
        probs_list.append(r[0].probs.data.cpu().numpy())

    probs_mean = np.mean(np.stack(probs_list, axis=0), axis=0)  # [C]
    top1_idx = int(np.argmax(probs_mean))
    conf = float(probs_mean[top1_idx])

    if not CLASSES:
        pred_name = f"class_{top1_idx}"
    else:
        pred_name = CLASSES[top1_idx]

    return pred_name, conf
