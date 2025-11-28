"""
Preprocess Pokémon sprite dataset:

- Index data/dataset/ (class folders → label)
- Remove unreadable/duplicate images
- Enforce minimum samples per class
- Create train_dev / test split (stratified)
- Create K stratified folds over train_dev
- Save:
    artifacts/dataset_index.csv
    artifacts/dedup_index.csv
    artifacts/train_dev_split.csv
    artifacts/test_split.csv
    artifacts/folds/fold_k/train.csv, val.csv
    artifacts/classes.txt
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.auto import tqdm
from pokeai.constants import DATA_DIR, ARTIFACTS

# ------------------------
# Reproducibility
# ------------------------
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)

# ------------------------
# Paths & config
# ------------------------
DATASET_DIR = Path(os.getenv("POKEAI_DATASET_DIR", DATA_DIR / "dataset")).resolve()
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FOLDS_DIR = ARTIFACTS / "folds"
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = int(os.getenv("N_FOLDS", "5"))
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ============================================================
# 1. Index dataset & filter unreadable files
# ============================================================

def index_dataset() -> pd.DataFrame:
    assert DATASET_DIR.exists(), f"Dataset dir not found: {DATASET_DIR}"

    classes = sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found under {DATASET_DIR}")

    print(f"[index] Found {len(classes)} classes")
    rows = []
    for c in tqdm(classes, desc="Indexing"):
        for fp in (DATASET_DIR / c).rglob("*"):
            if fp.is_file() and fp.suffix.lower() in EXTS:
                rel = fp.relative_to(DATASET_DIR).as_posix()
                rows.append({"path": rel, "label": c})
    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    print(f"[index] Total images (raw): {len(df)}")
    return df

def safe_size(p):
    try:
        with Image.open(DATASET_DIR / p) as im:
            im = im.convert("RGB")
            return im.size  # (w, h)
    except Exception:
        return (None, None)

def add_sizes_and_save(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas(desc="Sizing")
    sizes = df["path"].progress_apply(safe_size)
    df["width"] = sizes.apply(lambda s: s[0])
    df["height"] = sizes.apply(lambda s: s[1])

    unopenable = int(df["width"].isna().sum())
    print(f"[index] Unopenable images: {unopenable}")
    df = df.dropna(subset=["width", "height"]).reset_index(drop=True)

    out = ARTIFACTS / "dataset_index.csv"
    df.to_csv(out, index=False)
    print(f"[index] Saved dataset_index.csv → {out}")
    return df

# ============================================================
# 2. Deduplicate by hash
# ============================================================

def deduplicate_by_hash(df: pd.DataFrame) -> pd.DataFrame:
    import hashlib

    def file_md5(rel: str) -> str:
        h = hashlib.md5()
        with open(DATASET_DIR / rel, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    tqdm.pandas(desc="Hashing")
    df["hash"] = df["path"].progress_apply(file_md5)

    before = len(df)
    df = df.drop_duplicates(subset=["hash"]).reset_index(drop=True)
    removed = before - len(df)
    print(f"[dedup] Unique images: {len(df)} (removed {removed} duplicates)")

    out = ARTIFACTS / "dedup_index.csv"
    df.to_csv(out, index=False)
    print(f"[dedup] Saved dedup_index.csv → {out}")
    return df

# ============================================================
# 3. Label sanity + global splits
# ============================================================

def label_sanity_and_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    MIN_PER_CLASS = int(os.getenv("MIN_PER_CLASS", "2"))
    MAX_UPSAMPLE = int(os.getenv("MAX_UPSAMPLE", "1000"))
    SMALL_CLASS_STRATEGY = os.getenv("SMALL_CLASS_STRATEGY", "filter").lower()

    label_counts = df["label"].value_counts()
    too_small = label_counts[label_counts < MIN_PER_CLASS]

    if len(too_small) > 0:
        print("[sanity] Classes with too few samples (global):", dict(too_small))
        if SMALL_CLASS_STRATEGY == "filter":
            before = len(df)
            df = df[~df["label"].isin(too_small.index)].reset_index(drop=True)
            after = len(df)
            print(
                f"[sanity] Filtered rare classes (<{MIN_PER_CLASS}). "
                f"Rows: {before} -> {after}."
            )
        elif SMALL_CLASS_STRATEGY == "upsample":
            frames = [df]
            for cls, cnt in too_small.items():
                need = min(MIN_PER_CLASS - cnt, MAX_UPSAMPLE)
                add = df[df["label"] == cls].sample(n=need, replace=True, random_state=SEED)
                frames.append(add)
            df = (
                pd.concat(frames, ignore_index=True)
                .sample(frac=1.0, random_state=SEED)
                .reset_index(drop=True)
            )
            print(f"[sanity] Upsampled minority classes to at least {MIN_PER_CLASS}.")
        else:
            raise ValueError("SMALL_CLASS_STRATEGY must be 'filter' or 'upsample'")

    # 1) train_dev/test split
    train_dev_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["label"], random_state=SEED
    )

    # Ensure each class in train_dev has at least N_FOLDS samples
    td_counts = train_dev_df["label"].value_counts()
    too_small_td = td_counts[td_counts < N_FOLDS]

    if len(too_small_td) > 0:
        print(
            f"[sanity] Classes with <{N_FOLDS} items in train_dev "
            f"(for StratifiedKFold):",
            dict(too_small_td),
        )

        TD_SMALL_CLASS_STRATEGY = os.getenv("TD_SMALL_CLASS_STRATEGY", "filter").lower()
        if TD_SMALL_CLASS_STRATEGY == "filter":
            before = len(train_dev_df)
            train_dev_df = train_dev_df[
                ~train_dev_df["label"].isin(too_small_td.index)
            ].reset_index(drop=True)
            after = len(train_dev_df)
            print(
                f"[sanity] train_dev filtered (<{N_FOLDS}). "
                f"Rows: {before} -> {after}."
            )
        elif TD_SMALL_CLASS_STRATEGY == "upsample":
            frames = [train_dev_df]
            for cls, cnt in too_small_td.items():
                need = N_FOLDS - cnt
                add = train_dev_df[train_dev_df["label"] == cls].sample(
                    n=need, replace=True, random_state=SEED
                )
                frames.append(add)
            train_dev_df = (
                pd.concat(frames, ignore_index=True)
                .sample(frac=1.0, random_state=SEED)
                .reset_index(drop=True)
            )
            print(f"[sanity] Upsampled train_dev classes to at least {N_FOLDS}.")
        else:
            raise ValueError("TD_SMALL_CLASS_STRATEGY must be 'filter' or 'upsample'")

    # Ensure no class is only in test
    tv_labels = set(train_dev_df["label"].unique())
    test_only = sorted(list(set(test_df["label"].unique()) - tv_labels))
    if test_only:
        moved_rows = []
        for lab in test_only:
            cand = test_df[test_df["label"] == lab].head(1)
            if len(cand) == 1:
                moved_rows.append(cand.iloc[0])
                test_df = test_df.drop(cand.index)
        if moved_rows:
            train_dev_df = pd.concat([train_dev_df, pd.DataFrame(moved_rows)], ignore_index=True)

    # Save splits
    train_dev_df.to_csv(ARTIFACTS / "train_dev_split.csv", index=False)
    test_df.to_csv(ARTIFACTS / "test_split.csv", index=False)
    print("[splits] Saved train_dev_split.csv and test_split.csv")
    return train_dev_df, test_df


# ============================================================
# 4. Build folds + classes.txt
# ============================================================

def build_folds(train_dev_df: pd.DataFrame) -> tuple[List[str], Dict[str, int]]:
    classes = sorted(train_dev_df["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # classes.txt
    (ARTIFACTS / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    print(f"[classes] {len(classes)} classes saved to {ARTIFACTS/'classes.txt'}")

    # Stratified K-fold over train_dev
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_id, (tr_idx, va_idx) in enumerate(
        kf.split(train_dev_df["path"], train_dev_df["label"]), start=1
    ):
        df_tr = train_dev_df.iloc[tr_idx].reset_index(drop=True)
        df_va = train_dev_df.iloc[va_idx].reset_index(drop=True)

        fold_dir = FOLDS_DIR / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        df_tr.to_csv(fold_dir / "train.csv", index=False)
        df_va.to_csv(fold_dir / "val.csv", index=False)
        print(f"[folds] fold_{fold_id}: train={len(df_tr)}, val={len(df_va)}")

    return classes, class_to_idx


# ============================================================
# Main
# ============================================================

def main():
    print(f"[config] DATASET_DIR = {DATASET_DIR}")
    print(f"[config] ARTIFACTS   = {ARTIFACTS}")
    print(f"[config] N_FOLDS     = {N_FOLDS}")

    df = index_dataset()
    df = add_sizes_and_save(df)
    df = deduplicate_by_hash(df)
    train_dev_df, test_df = label_sanity_and_split(df)
    classes, _ = build_folds(train_dev_df)

    print(
        f"[done] Train_dev={len(train_dev_df)}, Test={len(test_df)}, "
        f"Classes={len(classes)}"
    )
    print(f"[done] Folds stored under: {FOLDS_DIR}")


if __name__ == "__main__":
    main()

