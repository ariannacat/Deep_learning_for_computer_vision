"""
Evaluate model robustness under different test-time augmentations.

This script:
  - Defines several strong augmentations (noise, occlusions, cropping, lighting).
  - Re-runs TEST evaluation under each augmentation.
  - Works for both:
      * Torchvision CV ensemble over folds
      * YOLOv8-CLS ensemble over folds
  - Saves:
      * Per-augmentation metrics (CSV)
      * Per-augmentation per-image predictions
      * Confusion matrices
      * Grids of 16 sample predictions
      * Overall summary CSV of all augmentations

Run with:
    python -m training.eval_augmentations
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from torch.utils.data import DataLoader

from training.preprocess_data import ARTIFACTS, DATASET_DIR, N_FOLDS, SEED
from pokeai.models import make_model, IMG_SIZE, IMAGENET_STD, IMAGENET_STD
from training.text_and_eval import BATCH_SIZE, NUM_WORKERS

from pokeai.vision import (
    make_model,
    fold_ckpt_path,
    DEVICE,
    MODEL_NAME,
    MODEL_NAME_SAFE,
    USE_YOLO,
    YOLO_NAME,
    YOLO_RUNS,
)

# ---------- Directories ----------
REPORTS_DIR = ARTIFACTS / "reports" / "augmented"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SUM_DIR = ARTIFACTS / "test_summaries" / "augmented"
TEST_SUM_DIR.mkdir(parents=True, exist_ok=True)

CLASSES: List[str] = load_classes_txt(ARTIFACTS / "classes.txt")

# ---------- Worker Init for Reproducibility ----------
def worker_init_fn(worker_id: int) -> None:
    """Ensure each DataLoader worker has a unique but reproducible random seed."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# ---------- Augmentation Functions ----------
class AddWhiteNoise:
    """Add Gaussian white noise to image (expects tensor in [0,1])"""
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        noise = torch.randn_like(img) * self.std + self.mean
        noisy = img + noise
        return torch.clamp(noisy, 0, 1)

class RandomOcclusion:
    """Add random black occlusions/patches to image (expects tensor in [0,1])"""
    def __init__(self, n_patches=3, patch_size_range=(20, 50)):
        self.n_patches = n_patches
        self.patch_size_range = patch_size_range
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        
        c, h, w = img.shape
        img_occluded = img.clone()
        
        for _ in range(self.n_patches):
            patch_h = random.randint(*self.patch_size_range)
            patch_w = random.randint(*self.patch_size_range)
            
            y = random.randint(0, max(0, h - patch_h))
            x = random.randint(0, max(0, w - patch_w))
            
            img_occluded[:, y:y+patch_h, x:x+patch_w] = 0
        
        return img_occluded

class RandomCropResize:
    """Crop random portion and resize back (expects tensor in [0,1])"""
    def __init__(self, scale_range=(0.6, 0.9), target_size=224):
        self.scale_range = scale_range
        self.target_size = target_size
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        
        c, h, w = img.shape
        scale = random.uniform(*self.scale_range)
        
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        
        y = random.randint(0, h - crop_h)
        x = random.randint(0, w - crop_w)
        
        cropped = img[:, y:y+crop_h, x:x+crop_w]
        resized = TF.resize(cropped, [self.target_size, self.target_size])
        
        return resized

class LightingShadowChange:
    """Apply random brightness and contrast changes (works on both PIL and tensor)"""
    def __init__(self, brightness_range=(0.6, 1.4), contrast_range=(0.6, 1.4)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        
        # Random brightness
        brightness_factor = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        
        return TF.to_tensor(img)

class CombinedAugmentation:
    """Apply all augmentations together"""
    def __init__(self):
        self.white_noise = AddWhiteNoise(mean=0.0, std=0.08)
        self.occlusion = RandomOcclusion(n_patches=2, patch_size_range=(15, 40))
        self.crop = RandomCropResize(scale_range=(0.7, 0.9), target_size=IMG_SIZE)
        self.lighting = LightingShadowChange(brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3))
    
    def __call__(self, img):
        img = self.crop(img)
        img = self.white_noise(img)
        img = self.occlusion(img)
        img = self.lighting(img)
        return img

# ---------- FIXED: Augmented Dataset Class ----------
class AugmentedCSVImageDataset(torch.utils.data.Dataset):
    """
    Dataset with on-the-fly augmentation.
    
    CRITICAL FIX: Augmentation is applied BEFORE normalization!
    Order: Load → Resize → Augment → Normalize
    """
    def __init__(self, df, class_to_idx, root_dir, augmentation=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.root_dir = Path(root_dir)
        self.augmentation = augmentation
        
        # Pre-compute resize transform (NO normalization yet)
        self.resize_transform = T.Compose([
            T.Resize(int(IMG_SIZE * 1.14)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),  # Convert to [0,1] tensor
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["path"]
        
        # Step 1: Load image
        img = Image.open(img_path).convert("RGB")
        
        # Step 2: Resize and convert to tensor [0,1]
        img = self.resize_transform(img)
        
        # Step 3: Apply augmentation (works on [0,1] tensor)
        if self.augmentation is not None:
            img = self.augmentation(img)
        
        # Step 4: Ensure it's a tensor (safety check)
        if isinstance(img, Image.Image):
            img = TF.to_tensor(img)
        
        # Step 5: NORMALIZE AT THE END (ImageNet mean/std)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        
        label = self.class_to_idx[row["label"]]
        return img, label

# ---------- Main evaluation function ----------
def evaluate_with_augmentation(aug_name, augmentation_fn):
    """
    Evaluate current model with a specific augmentation.

    Returns:
        (accuracy, macro_f1, macro_precision)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating with augmentation: {aug_name}")
    print(f"{'='*60}\n")
    
    test_df = pd.read_csv(ARTIFACTS / "test_split.csv")
    
    if not USE_YOLO:
        # ---- Torchvision with augmentation ----
        test_ds = AugmentedCSVImageDataset(
            test_df, 
            class_to_idx, 
            DATASET_DIR, 
            augmentation=augmentation_fn  # No base_transform needed!
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,  # FIX: Add worker init for reproducibility
        )
        y_true = test_df["label"].map(class_to_idx).to_numpy()
        
        # CV ensemble over folds
        all_logits = []
        for fold_id in range(1, N_FOLDS + 1):
            ckpt_path = fold_ckpt_path(fold_id)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            # robust handling of checkpoint formats
            if isinstance(ckpt, dict) and "model" in ckpt:
                sd = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            else:
                sd = ckpt

            # optional: strip "module." prefix
            clean_sd = {}
            for k, v in sd.items():
                if k.startswith("module."):
                    k = k[len("module."):]
                clean_sd[k] = v

            # infer num classes from head if needed
            n_classes_trained = None
            for k, v in clean_sd.items():
                if k.endswith("classifier.6.weight") and v.ndim == 2:
                    n_classes_trained = v.shape[0]; break
                if k.endswith("fc.weight") and v.ndim == 2:
                    n_classes_trained = v.shape[0]; break
            if n_classes_trained is None:
                n_classes_trained = len(classes)

            m = make_model(MODEL_NAME, n_classes_trained, pretrained=False).to(DEVICE).eval()
            m.load_state_dict(clean_sd, strict=True)

            fold_logits = []
            with torch.no_grad():
                for xb, _ in test_dl:
                    xb = xb.to(DEVICE)
                    logits = m(xb)
                    fold_logits.append(logits.detach().cpu().numpy())
            all_logits.append(np.concatenate(fold_logits, axis=0))

        # ensemble: mean logits across folds
        logits_ens = np.mean(np.stack(all_logits, axis=0), axis=0)
        y_pred_idx = logits_ens.argmax(axis=1)

        # probs for confidence
        logits_max = logits_ens.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_ens - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)

        # metrics
        acc = accuracy_score(y_true, y_pred_idx)
        f1 = f1_score(y_true, y_pred_idx, average="macro")
        precision = precision_score(y_true, y_pred_idx, average="macro")
        print(f"[{aug_name}] Test accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro precision: {precision:.4f}")

        # Save summary
        summary_row = pd.DataFrame([{
            "model": MODEL_NAME_SAFE,
            "augmentation": aug_name,
            "test_acc": acc,
            "test_f1": f1,
            "test_precision": precision,
        }])
        summary_csv = TEST_SUM_DIR / f"test_summary_{MODEL_NAME_SAFE}_{aug_name}.csv"
        summary_row.to_csv(summary_csv, index=False)
        print(f"[saved] test summary row → {summary_csv}")

        # Classification report
        rep_txt = classification_report(
            y_true,
            y_pred_idx,
            target_names=classes,
            zero_division=0,
        )
        print(f"\nClassification report preview:")
        print(rep_txt[:500], "...\n")

        rep_path = REPORTS_DIR / f"test_classification_report_{MODEL_NAME_SAFE}_{aug_name}.txt"
        with open(rep_path, "w") as f:
            f.write(f"Model: {MODEL_NAME_SAFE}\n")
            f.write(f"Augmentation: {aug_name}\n")
            f.write(f"Folds: {N_FOLDS}\n")
            f.write(f"Test accuracy: {acc:.4f}\n")
            f.write(f"Test macro F1: {f1:.4f}\n")
            f.write(f"Test macro precision: {precision:.4f}\n\n")
            f.write(rep_txt)
        print(f"[saved] full test classification report → {rep_path}")

        return acc, f1, precision

    else:
        # ---- YOLO with augmentation ----
        print(f"[WARNING] YOLO augmentation requires image preprocessing - implementing simplified version")
        
        # For YOLO, we need to save augmented images temporarily
        test_paths = test_df["path"].tolist()
        y_true_names = test_df["label"].tolist()
        
        YOLO_RUNS = ARTIFACTS / "yolo_cls_runs"
        
        def _best_weights_for_fold(fold_id: int) -> Path:
            run_name = f"{YOLO_NAME}_fold{fold_id}"
            direct = YOLO_RUNS / run_name / "weights" / "best.pt"
            if direct.is_file():
                return direct
            candidates = list(YOLO_RUNS.glob(f"{run_name}*/weights/best*.pt"))
            if not candidates:
                raise FileNotFoundError(f"Could not find best weights for fold {fold_id}")
            return candidates[0]

        # Temporarily save augmented images
        temp_aug_dir = ARTIFACTS / "temp_augmented" / aug_name
        temp_aug_dir.mkdir(parents=True, exist_ok=True)
        
        augmented_paths = []
        resize_tfm = T.Compose([
            T.Resize(int(IMG_SIZE * 1.14)),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
        ])
        
        for orig_path in test_paths:
            pth = Path(orig_path)
            if not pth.is_file():
                pth = DATASET_DIR / orig_path
            
            # Load and augment (CORRECT ORDER)
            img = Image.open(pth).convert("RGB")
            img = resize_tfm(img)  # Resize to tensor [0,1]
            
            if augmentation_fn:
                img = augmentation_fn(img)  # Apply augmentation
            
            # Save augmented image (NO normalization for YOLO)
            if isinstance(img, torch.Tensor):
                img = TF.to_pil_image(img)
            
            aug_path = temp_aug_dir / Path(orig_path).name
            img.save(aug_path)
            augmented_paths.append(str(aug_path))
        
        # Predict on augmented images
        def yolo_probs_for_paths(weights_path: Path, paths: list) -> np.ndarray:
            mdl = YOLO(str(weights_path))
            probs_list = []
            for p in paths:
                r = mdl.predict(str(p), verbose=False)
                probs_list.append(r[0].probs.data.cpu().numpy())
            return np.vstack(probs_list)

        all_probs = []
        name_to_idx = {c: i for i, c in enumerate(classes)}

        for fold_id in range(1, N_FOLDS + 1):
            best_w = _best_weights_for_fold(fold_id)
            print(f"[Fold {fold_id}] using weights: {best_w}")
            probs_k = yolo_probs_for_paths(best_w, augmented_paths)
            all_probs.append(probs_k)

        probs_ens = np.mean(np.stack(all_probs, axis=0), axis=0)
        y_pred_idx = probs_ens.argmax(axis=1)
        y_true_idx = np.array([name_to_idx[n] for n in y_true_names])
        confs = probs_ens.max(axis=1)

        # metrics
        acc = accuracy_score(y_true_idx, y_pred_idx)
        f1 = f1_score(y_true_idx, y_pred_idx, average="macro")
        precision = precision_score(y_true_idx, y_pred_idx, average="macro")
        print(f"[YOLO - {aug_name}] Test accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro precision: {precision:.4f}")

        # Save summary
        summary_row = pd.DataFrame([{
            "model": MODEL_NAME_SAFE,
            "augmentation": aug_name,
            "test_acc": acc,
            "test_f1": f1,
            "test_precision": precision,
        }])
        summary_csv = TEST_SUM_DIR / f"test_summary_{MODEL_NAME_SAFE}_{aug_name}.csv"
        summary_row.to_csv(summary_csv, index=False)
        print(f"[saved] test summary row → {summary_csv}")

        # Classification report
        rep_txt = classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=classes,
            zero_division=0,
        )

        rep_path = REPORTS_DIR / f"test_classification_report_{MODEL_NAME_SAFE}_{aug_name}.txt"
        with open(rep_path, "w") as f:
            f.write(f"Model: {MODEL_NAME_SAFE}\n")
            f.write(f"Augmentation: {aug_name}\n")
            f.write(f"Folds: {N_FOLDS}\n")
            f.write(f"Test accuracy: {acc:.4f}\n")
            f.write(f"Test macro F1: {f1:.4f}\n")
            f.write(f"Test macro precision: {precision:.4f}\n\n")
            f.write(rep_txt)
        print(f"[saved] full test classification report → {rep_path}")

        return acc, f1, precision

# ---------- Run all augmentations ----------
def run_all_augmentations() -> pd.DataFrame:
    augmentations = {
        "white_noise": AddWhiteNoise(mean=0.0, std=0.1),
        "occlusions": RandomOcclusion(n_patches=3, patch_size_range=(20, 50)),
        "cropping": RandomCropResize(scale_range=(0.6, 0.9), target_size=IMG_SIZE),
        "lighting_shadow": LightingShadowChange(
            brightness_range=(0.6, 1.4),
            contrast_range=(0.6, 1.4),
        ),
        "all_combined": CombinedAugmentation(),
    }

    results_summary: List[Dict] = []

    for aug_name, aug_fn in augmentations.items():
        acc, f1, prec = evaluate_with_augmentation(aug_name, aug_fn)
        results_summary.append(
            {
                "model": MODEL_NAME_SAFE,
                "augmentation": aug_name,
                "test_acc": acc,
                "test_f1": f1,
                "test_precision": prec,
            }
        )

    overall_summary = pd.DataFrame(results_summary)
    overall_csv = REPORTS_DIR / f"test_summary_all_augmentations_{MODEL_NAME_SAFE}.csv"
    overall_summary.to_csv(overall_csv, index=False)

    print("\n" + "=" * 60)
    print("[COMPLETE] All augmentations evaluated!")
    print(f"[saved] overall summary → {overall_csv}")
    print("=" * 60 + "\n")
    print(overall_summary.to_string(index=False))

    return overall_summary


if __name__ == "__main__":
    run_all_augmentations()
