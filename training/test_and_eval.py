"""
Evaluate trained models on the held-out test set, using CV ensemble
(just like in the notebook), WITHOUT training.

It assumes:
- training.preprocess_data has been run
- training.train_models has been run for the chosen MODEL_NAME
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    classification_report,
    confusion_matrix,
)

from training.train_models import YOLO_RUNS, YOLO_NAME
from pokeai.constants import DEVICE
from training.train_constants import SEED, BATCH_SIZE, NUM_WORKERS, MODEL_NAME
from training.preprocess_data import ARTIFACTS, DATASET_DIR, FOLDS_DIR, N_FOLDS
from pokeai.models import make_model, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from pokeai.io_utils import load_classes_txt

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# Config
# ============================================================

MODEL_NAME_SAFE = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME)
USE_YOLO = MODEL_NAME.startswith("yolov8") and MODEL_NAME.endswith("-cls")

REPORTS_DIR = ARTIFACTS / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SUM_DIR = ARTIFACTS / "test_summaries"
TEST_SUM_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Classes & dataset
# ============================================================


CLASSES: List[str] = load_classes_txt(ARTIFACTS / "classes.txt")
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

eval_tfms = transforms.Compose(
    [
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int], root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.root / row["path"]).convert("RGB")
        y = self.class_to_idx[row["label"]]
        if self.transform:
            img = self.transform(img)
        return img, y

# ============================================================
# Evaluation
# ============================================================

def evaluator():

    # ---------- Prepare test set ----------
    test_df = pd.read_csv(ARTIFACTS / "test_split.csv")
    if not USE_YOLO:
        # Torchvision test loader (build once)
        test_ds  = CSVImageDataset(test_df, class_to_idx, DATASET_DIR, transform=eval_tfms)
        test_dl  = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        y_true   = test_df["label"].map(class_to_idx).to_numpy()
    else:
        # YOLO needs file paths and string labels
        test_paths   = test_df["path"].tolist()
        y_true_names = test_df["label"].tolist()

    # ---------- CV ensemble over folds ----------
    if not USE_YOLO:
        # ---- Torchvision: average logits over fold checkpoints ----
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
        logits_ens = np.mean(np.stack(all_logits, axis=0), axis=0)   # [N, C]
        y_pred_idx = logits_ens.argmax(axis=1)

        # probs for confidence
        logits_max = logits_ens.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_ens - logits_max)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)

        # metrics (now with precision)
        acc       = accuracy_score(y_true, y_pred_idx)
        f1        = f1_score(y_true, y_pred_idx, average="macro")
        precision = precision_score(y_true, y_pred_idx, average="macro")
        print(f"[CV ensemble] Test accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro precision: {precision:.4f}")

        # ----- Single-row summary CSV for model comparison -----
        summary_row = pd.DataFrame([{
            "model": MODEL_NAME_SAFE,
            "test_acc": acc,
            "test_f1": f1,
            "test_precision": precision,
        }])

        # ----- Classification report -----
        rep_txt = classification_report(
            y_true,
            y_pred_idx,
            target_names=classes,
            zero_division=0,
        )
        print("\nClassification report (head):")
        print(rep_txt[:1000], "...\n")

    else:
        # ---- YOLOv8-CLS: average probabilities across folds ----

        def _best_weights_for_fold(fold_id: int) -> Path:
            run_name = f"{YOLO_NAME}_fold{fold_id}"
            direct = YOLO_RUNS / run_name / "weights" / "best.pt"
            if direct.is_file():
                return direct
            candidates = list(YOLO_RUNS.glob(f"{run_name}*/weights/best*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"Could not find best weights for fold {fold_id}. "
                    f"Tried: {direct} and {YOLO_RUNS}/{run_name}*/weights/best*.pt"
                )
            return candidates[0]

        def yolo_probs_for_paths(weights_path: Path, paths: list[str]) -> np.ndarray:
            mdl = YOLO(str(weights_path))
            probs_list = []
            for p in paths:
                pth = Path(p)
                if not pth.is_file():
                    pth = DATASET_DIR / p
                r = mdl.predict(str(pth), verbose=False)
                probs_list.append(r[0].probs.data.cpu().numpy())
            return np.vstack(probs_list)  # [N, C]

        # collect per-fold probs
        all_probs = []
        name_to_idx = {c: i for i, c in enumerate(classes)}

        for fold_id in range(1, N_FOLDS + 1):
            best_w = _best_weights_for_fold(fold_id)
            print(f"[Fold {fold_id}] using weights: {best_w}")
            probs_k = yolo_probs_for_paths(best_w, test_paths)
            all_probs.append(probs_k)

        probs_ens = np.mean(np.stack(all_probs, axis=0), axis=0)   # [N, C]
        y_pred_idx = probs_ens.argmax(axis=1)
        y_true_idx = np.array([name_to_idx[n] for n in y_true_names])
        confs = probs_ens.max(axis=1)

        # metrics (with precision)
        acc       = accuracy_score(y_true_idx, y_pred_idx)
        f1        = f1_score(y_true_idx, y_pred_idx, average="macro")
        precision = precision_score(y_true_idx, y_pred_idx, average="macro")
        print(f"[YOLO CV ensemble] Test accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Macro precision: {precision:.4f}")

        # ----- Single-row summary CSV for model comparison -----
        summary_row = pd.DataFrame([{
            "model": MODEL_NAME_SAFE,
            "test_acc": acc,
            "test_f1": f1,
            "test_precision": precision,
        }])

        # ----- Classification report -----
        rep_txt = classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=classes,
            zero_division=0,
        )
        print("\nClassification report (head):")
        print(rep_txt[:1000], "...\n")

    return summary_row, rep_txt

if __name__ == "__main__":
    summary_row, rep_txt = evaluator()

    summary_csv = TEST_SUM_DIR / f"test_summary_{MODEL_NAME_SAFE}.csv"
    summary_row.to_csv(summary_csv, index=False)
    print(f"[saved] test summary row → {summary_csv}")

    rep_path = REPORTS_DIR / f"test_classification_report_{MODEL_NAME_SAFE}.txt"
    with open(rep_path, "w") as f:
        f.write(f"Model: {MODEL_NAME_SAFE}\n")
        f.write(f"Folds: {N_FOLDS}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro F1: {f1:.4f}\n")
        f.write(f"Test macro precision: {precision:.4f}\n\n")
        f.write(rep_txt)
    print(f"[saved] full test classification report → {rep_path}")
