"""
Train models and evaluate on held-out test set, following the same logic
as the notebook:

- If MODEL_NAME is a Torchvision model (resnet/vgg):
    * Train N_FOLDS models with CV
    * Save best fold checkpoints in artifacts/
    * Run CV ensemble on test_split.csv and save metrics
- If MODEL_NAME is YOLOv8-CLS:
    * Materialize fold-specific YOLO folders
    * Train YOLO per fold
    * Run CV ensemble on test set and save metrics
"""

from __future__ import annotations

import os
import re
import time, shutil
from pathlib import Path
from typing import List, Dict, Tuple


import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    classification_report,
    confusion_matrix,
)

# Training constants
from training.train_constants import SEED, BATCH_SIZE, NUM_WORKERS, LR, WEIGHT_DECAY, EPOCHS, VAL_PATIENCE, WARMUP_EPOCHS, MODEL_NAME

# Shared from preprocessing
from training.preprocess_data import ARTIFACTS, DATASET_DIR, FOLDS_DIR, N_FOLDS

# Shared model builder
from pokeai.models import make_model, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

# Shared class loader
from pokeai.io_utils import load_classes_txt

# YOLO (optional)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# Config
# ============================================================

MODEL_NAME_SAFE = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL_NAME)

USE_YOLO = MODEL_NAME.startswith("yolov8") and MODEL_NAME.endswith("-cls")

FOLDS_ROOT = ARTIFACTS / "folds"

# YOLO dirs
YOLO_ROOT = ARTIFACTS / "yolo_cls_data"
YOLO_RUNS = ARTIFACTS / "yolo_cls_runs"
YOLO_NAME = os.getenv("YOLO_NAME", "pokemon_yolo_cls")
YOLO_ROOT.mkdir(parents=True, exist_ok=True)
YOLO_RUNS.mkdir(parents=True, exist_ok=True)

# ============================================================
# Classes & transforms
# ============================================================

CLASSES: List[str] = load_classes_txt(ARTIFACTS / "classes.txt")
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

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
# Fold helpers: torchvision and YOLO
# ============================================================

def fold_name(fold_idx: int) -> str:
    return f"{MODEL_NAME_SAFE}_fold{fold_idx}"


def fold_ckpt_path(fold_idx: int) -> Path:
    return ARTIFACTS / f"best_{fold_name(fold_idx)}.pth"

# -------- Torchvision: build loaders for a given fold --------
def build_tv_loaders_for_fold(fold_id: int):
    fold_dir = Path(FOLDS_ROOT) / f"fold_{fold_id}"
    train_df = pd.read_csv(fold_dir / "train.csv")
    val_df   = pd.read_csv(fold_dir / "val.csv")
    test_df  = pd.read_csv(ARTIFACTS / "test_split.csv")

    train_ds = CSVImageDataset(train_df, class_to_idx, DATASET_DIR, transform=train_tfms)
    val_ds   = CSVImageDataset(val_df,   class_to_idx, DATASET_DIR, transform=eval_tfms)
    test_ds  = CSVImageDataset(test_df,  class_to_idx, DATASET_DIR, transform=eval_tfms)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return train_dl, val_dl, test_dl, len(train_ds), len(val_ds), len(test_ds)

# ------- Torchvision: CV-ready builders ----------------------
num_classes = len(classes)

def init_torchvision_for_fold(fold_id: int):
    """
    Build a fresh torchvision model and training objects for a given fold.
    Returns: model, criterion, optimizer, scheduler, scaler, ckpt_path
    """
    # model
    model = make_model(MODEL_NAME, num_classes, pretrained=True).to(DEVICE)

    # optional: freeze VGG features for warmup (head-only), unfreeze in loop later
    if MODEL_NAME.startswith("vgg"):
        print(f"[Fold {fold_id}] Freezing VGG feature extractor for warmup.")
        for p in model.features.parameters():
            p.requires_grad = False

    # loss / optim / sched / amp
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    ckpt_path = fold_ckpt_path(fold_id)
    print(f"[Fold {fold_id}] Model ready: {MODEL_NAME} | pretrained={USE_PRETRAINED} | classes={num_classes}")
    return model, criterion, optimizer, scheduler, scaler, ckpt_path

# -------- YOLO: materialize per-fold folder layout --------
def materialize_yolo_fold(fold_id: int):
    import shutil
    from tqdm.auto import tqdm

    fold_dir = Path(FOLDS_ROOT) / f"fold_{fold_id}"
    train_df = pd.read_csv(fold_dir / "train.csv")
    val_df   = pd.read_csv(fold_dir / "val.csv")

    yolo_cv_dir    = ARTIFACTS / "yolo_folds"
    yolo_fold_root = yolo_cv_dir / f"fold_{fold_id}"
    yolo_train     = yolo_fold_root / "train"
    yolo_val       = yolo_fold_root / "val"

    if yolo_fold_root.exists():
        shutil.rmtree(yolo_fold_root)
    yolo_train.mkdir(parents=True, exist_ok=True)
    yolo_val.mkdir(parents=True, exist_ok=True)

    def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
    def _copy_file(src: Path, dst: Path):
        _ensure_dir(dst.parent)
        if not dst.exists(): shutil.copy2(src, dst)

    def _mat(df, split_root: Path):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copy → {split_root.name}"):
            rel = Path(row["path"])
            src = (DATASET_DIR / rel) if not rel.is_absolute() else rel
            if src.exists():
                dst = split_root / str(row["label"]) / src.name
                _copy_file(src, dst)

    _mat(train_df, yolo_train)
    _mat(val_df,   yolo_val)
    return yolo_fold_root  # contains train/ and val/

# ============================================================
# Training
# ============================================================

def trainer():
    fold_summaries = []

    if not (MODEL_NAME.startswith("yolov8") and MODEL_NAME.endswith("-cls")):
        # ================= TORCHVISION 5-FOLD TRAINING =================

        for fold_id in range(1, N_FOLDS + 1):
            print(f"\n==================== Fold {fold_id}/{N_FOLDS} ====================")
            fold_dir = FOLDS_ROOT / f"fold_{fold_id}"
            assert fold_dir.exists(), f"Missing fold dir: {fold_dir}"

            # Load per-fold splits
            train_df = pd.read_csv(fold_dir / "train.csv")
            val_df   = pd.read_csv(fold_dir / "val.csv")

            # Build datasets/dataloaders for this fold
            train_ds = CSVImageDataset(train_df, class_to_idx, DATASET_DIR, transform=train_tfms)
            val_ds   = CSVImageDataset(val_df,   class_to_idx, DATASET_DIR, transform=eval_tfms)

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
            val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

            # Fresh model/optim/sched/scaler per fold
            model, criterion, optimizer, scheduler, scaler, ckpt_path = init_torchvision_for_fold(fold_id)

            # Train loop (with optional VGG unfreeze)
            best_val = float("inf")
            best_sd  = None
            wait     = 0
            did_unfreeze = False

            def run_epoch(dataloader, train: bool):
                model.train(train)
                total_loss, total_correct, total = 0.0, 0, 0
                for xb, yb in dataloader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = model(xb)
                        loss   = criterion(logits, yb)
                    if train:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    total_loss   += loss.item() * yb.size(0)
                    total_correct += (logits.argmax(1) == yb).sum().item()
                    total        += yb.size(0)
                return total_loss/total, total_correct/total

            for epoch in range(1, EPOCHS + 1):
                # Unfreeze VGG backbone after warmup (optional)
                if (not did_unfreeze) and MODEL_NAME.startswith("vgg") and epoch == WARMUP_EPOCHS + 1:
                    print(f"[Fold {fold_id}] Unfreezing VGG backbone at epoch {epoch}.")
                    for p in model.features.parameters():
                        p.requires_grad = True
                    optimizer = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
                    did_unfreeze = True

                t0 = time.time()
                tr_loss, tr_acc = run_epoch(train_dl, train=True)
                va_loss, va_acc = run_epoch(val_dl,   train=False)
                scheduler.step(epoch + va_loss)

                print(f"[Fold {fold_id}] Epoch {epoch:02d} | "
                    f"train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                    f"val_loss {va_loss:.4f} acc {va_acc:.3f} | "
                    f"{time.time()-t0:.1f}s")

                # Early stopping on val loss
                if va_loss < best_val - 1e-4:
                    best_val = va_loss; wait = 0
                    best_sd = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= VAL_PATIENCE:
                        print(f"[Fold {fold_id}] Early stopping.")
                        break

            # Save best checkpoint for this fold
            if best_sd:
                model.load_state_dict(best_sd)
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Fold {fold_id}] Saved best weights to: {ckpt_path}")

            fold_summaries.append({
                "model": MODEL_NAME_SAFE,
                "fold": fold_id,
                "val_loss": best_val,
                "val_acc": va_acc   # add this metric
            })

        print("\nCV summary (val_loss):", fold_summaries)

    else:
        # ================= YOLOv8-CLS 5-FOLD TRAINING =================
        # Assumes per-fold CSVs exist under artifacts/folds/fold_k/train.csv & val.csv
        from tqdm.auto import tqdm

        YOLO_CV_DIR = ARTIFACTS / "yolo_folds"
        YOLO_CV_DIR.mkdir(parents=True, exist_ok=True)

        def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
        def _copy_file(src: Path, dst: Path):
            _ensure_dir(dst.parent)
            if not dst.exists():
                shutil.copy2(src, dst)

        def materialize_split(df, split_root: Path, src_root: Path):
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copy → {split_root.name}"):
                rel = Path(row["path"])
                src = (src_root / rel) if not rel.is_absolute() else rel
                if src.exists():
                    dst = split_root / str(row["label"]) / src.name
                    _copy_file(src, dst)

        for fold_id in range(1, N_FOLDS + 1):
            print(f"\n==================== YOLO Fold {fold_id}/{N_FOLDS} ====================")
            fold_dir = FOLDS_ROOT / f"fold_{fold_id}"
            train_df = pd.read_csv(fold_dir / "train.csv")
            val_df   = pd.read_csv(fold_dir / "val.csv")

            # Prepare YOLO folder layout for this fold
            YOLO_FOLD_ROOT = YOLO_CV_DIR / f"fold_{fold_id}"
            YOLO_TRAIN = YOLO_FOLD_ROOT / "train"
            YOLO_VAL   = YOLO_FOLD_ROOT / "val"
            if YOLO_FOLD_ROOT.exists():
                shutil.rmtree(YOLO_FOLD_ROOT)
            YOLO_TRAIN.mkdir(parents=True, exist_ok=True)
            YOLO_VAL.mkdir(parents=True, exist_ok=True)

            materialize_split(train_df, YOLO_TRAIN, DATASET_DIR)
            materialize_split(val_df,   YOLO_VAL,   DATASET_DIR)
            print(f"[Fold {fold_id}] YOLO data ready at {YOLO_FOLD_ROOT}")

            # Train YOLO for this fold (per-fold run name)
            run_name = f"{YOLO_NAME}_fold{fold_id}"
            yolo_model = YOLO(YOLO_WEIGHTS)
            yolo_model.train(
                data=str(YOLO_FOLD_ROOT),           # has train/ and val/
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                workers=NUM_WORKERS,
                device=(0 if DEVICE.type == "cuda" else "cpu"),
                project=str(YOLO_RUNS),
                name=run_name,
                exist_ok=True,
                verbose=True,
            )
            print(f"[Fold {fold_id}] YOLO best weights → {YOLO_RUNS / run_name / 'weights' / 'best.pt'}")

            run_dir = YOLO_RUNS / run_name
            results_csv = run_dir / "results.csv"

            val_acc = None
            val_loss = None

            if results_csv.exists():
                df_res = pd.read_csv(results_csv)

                # Try common column names for classification metrics
                acc_cols = [c for c in df_res.columns if c.lower() in
                            {"metrics/accuracy_top1", "accuracy_top1", "top1_acc", "val/acc", "val_acc", "acc"}]
                loss_cols = [c for c in df_res.columns if c.lower() in
                            {"val/loss", "loss", "metrics/loss", "val_loss"}]

                # Pick best top-1 acc across epochs if available; otherwise last epoch
                if acc_cols:
                    val_acc = float(df_res[acc_cols[0]].max())  # best accuracy
                if loss_cols:
                    val_loss = float(df_res[loss_cols[0]].min())  # best (lowest) val loss

                # Fallbacks if columns weren't found: try last row with any sensible names
                if val_acc is None and results_csv.exists():
                    last = df_res.tail(1)
                    for c in df_res.columns:
                        if "acc" in c.lower():
                            val_acc = float(last[c].iloc[0]); break
                for c in df_res.columns if results_csv.exists() else []:
                    if "loss" in c.lower():
                        val_loss = float(df_res[c].min()); break

            # Make sure we always append something
            if val_acc is None: val_acc = float("nan")
            if val_loss is None: val_loss = float("nan")

            fold_summaries.append({
                "model": MODEL_NAME_SAFE,
                "fold": fold_id,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            print(f"[Fold {fold_id}] YOLO metrics → acc(top1)={val_acc:.4f}  loss={val_loss:.4f}")
    return fold_summaries

if __name__ == "__main__":
    summaries = trainer()

    # Save summaries under artifacts/fold_summaries
    fs_dir = ARTIFACTS / "fold_summaries"
    fs_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(summaries)
    out_path = fs_dir / f"{MODEL_NAME_SAFE}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[Train Done] Saved fold summaries → {out_path}")

