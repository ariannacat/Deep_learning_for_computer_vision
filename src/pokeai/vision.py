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

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

# ============================================================
# Config / paths 
# ============================================================

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image + normalization
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Artifacts root (where classes.txt, folds, checkpoints live)
ARTIFACTS = Path(os.getenv("POKEAI_ARTIFACTS", "artifacts")).resolve()

# Dataset dir is only used as *fallback* if the given image path is relative
DATASET_DIR = Path(os.getenv("POKEAI_DATASET_DIR", "data/dataset")).resolve()

# Cross-validation config
N_FOLDS = int(os.getenv("N_FOLDS", "5"))

# Model name (same as in your notebook)
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18")  # e.g. vgg16, resnet50, yolov8n-cls
MODEL_NAME_SAFE = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME)

# YOLO config
USE_YOLO = MODEL_NAME.startswith("yolov8") and MODEL_NAME.endswith("-cls")
YOLO_NAME = os.getenv("YOLO_NAME", "pokemon_yolo_cls")
YOLO_RUNS = ARTIFACTS / "yolo_cls_runs"

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

def fold_name(fold_idx: int) -> str:
    return f"{MODEL_NAME_SAFE}_fold{fold_idx}"

def fold_ckpt_path(fold_idx: int) -> Path:
    """
    Path of the per-fold checkpoint for Torchvision models.
    Matches your notebook:

        artifacts/best_<MODEL_NAME_SAFE>_foldK.pth
    """
    return ARTIFACTS / f"best_{fold_name(fold_idx)}.pth"


# ============================================================
# Model builder 
# ============================================================

def make_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()

    # --- ResNet family ---
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    # --- VGG family (vgg11/13/16/19 with or without _bn) ---
    if name.startswith("vgg"):
        ctor = getattr(models, name)  # e.g. models.vgg16, models.vgg16_bn
        weights = None
        if pretrained:
            enum_name = name.replace("vgg", "VGG") + "_Weights"
            try:
                weights_enum = getattr(models, enum_name)
                weights = getattr(weights_enum, "DEFAULT", None) or getattr(weights_enum, "IMAGENET1K_V1", None)
            except Exception:
                weights = None
        try:
            m = ctor(weights=weights)
        except TypeError:
            m = ctor(pretrained=pretrained)  # older torchvision fallback

        # replace last Linear in classifier
        last_linear_idx = None
        for i in reversed(range(len(m.classifier))):
            if isinstance(m.classifier[i], nn.Linear):
                last_linear_idx = i; break
        if last_linear_idx is None:
            raise ValueError("Unexpected VGG classifier structure.")
        in_f = m.classifier[last_linear_idx].in_features
        m.classifier[last_linear_idx] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unsupported model: {name}")

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

    candidate = DATASET_DIR / path_or_rel
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Image not found: {path_or_rel} "
        f"(tried '{p}' and '{candidate}')"
    )

# ============================================================
# Main API: predict_image
# ============================================================

def predict_image(path_or_rel: Union[str, Path]) -> Tuple[str, float]:
    """
    Predict a single image class using your original logic:

      • Torchvision (ResNet/VGG/etc.) → averages logits over CV folds
      • YOLOv8-CLS → averages probabilities over CV folds

    Returns:
        (pred_name, confidence)
    """
    img_path = _resolve_image_path(path_or_rel)

    if USE_YOLO:
        if YOLO is None:
            raise ImportError(
                "Ultralytics YOLO is not installed. "
                "Install with: pip install ultralytics"
            )
        return _predict_yolo_ensemble(img_path)

    return _predict_torchvision_ensemble(img_path)

# ============================================================
# Torchvision CV-ensemble prediction
# ============================================================

def _predict_torchvision_ensemble(img_path: Path) -> Tuple[str, float]:
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

    logits_list = []

    for fold_id in range(1, N_FOLDS + 1):
        ckpt_path = fold_ckpt_path(fold_id)
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold_id}: {ckpt_path}"
            )

        model = make_model(MODEL_NAME, len(CLASSES), pretrained=False).to(DEVICE).eval()
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

def _best_yolo_weights_for_fold(fold_id: int) -> Path:
    """
    Find YOLO best.pt for a given fold, mirroring your naming:

        run_name = f"{YOLO_NAME}_fold{fold_id}"
        YOLO_RUNS / run_name / "weights" / "best.pt"
    """
    run_name = f"{YOLO_NAME}_fold{fold_id}"
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


def _predict_yolo_ensemble(img_path: Path) -> Tuple[str, float]:
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

    for fold_id in range(1, N_FOLDS + 1):
        best_weights = _best_yolo_weights_for_fold(fold_id)
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
