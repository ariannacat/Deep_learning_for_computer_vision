"""
Shared vision model utilities (Torchvision architectures).

Used by:
- training scripts (training/train_models.py)
- runtime inference (pokeai.vision)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# Common image size + normalization (same as notebook)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build a Torchvision classifier and adapt the final layer to num_classes.
    Matches the logic from your notebook.
    """
    name = name.lower()

    # --- ResNet family ---
    if name == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
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
                weights = getattr(weights_enum, "DEFAULT", None) or getattr(
                    weights_enum, "IMAGENET1K_V1", None
                )
            except Exception:
                weights = None
        try:
            m = ctor(weights=weights)
        except TypeError:
            # older torchvision fallback
            m = ctor(pretrained=pretrained)

        # replace last Linear in classifier
        last_linear_idx = None
        for i in reversed(range(len(m.classifier))):
            if isinstance(m.classifier[i], nn.Linear):
                last_linear_idx = i
                break
        if last_linear_idx is None:
            raise ValueError("Unexpected VGG classifier structure.")
        in_f = m.classifier[last_linear_idx].in_features
        m.classifier[last_linear_idx] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unsupported model: {name}")

