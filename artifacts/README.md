# Artifacts

This directory contains all **generated outputs** produced during dataset preprocessing,
training, evaluation, and robustness testing. Nothing in this folder is committed to
GitHub except this README and optional `.gitkeep` files — all heavy files (checkpoints,
YOLO runs, reports, metrics) are created locally.

The structure is automatically populated by the training scripts in
`training/`, and is consumed at inference time by `pokeai.vision` and the
main pipeline.

---

## Folder structure
```kotlin
artifacts/
├─ classes.txt
│
├─ train_dev_split.csv
├─ test_split.csv
│
├─ folds/
│   ├─ fold_1/
│   │   ├─ train.csv
│   │   └─ val.csv
│   ├─ fold_2/
│   └─ ...
│
├─ fold_summaries/
│   └─ resnet18.csv
│
├─ best_resnet18_fold1.pth
├─ best_resnet18_fold2.pth
│   ... (Torchvision weights — ignored by git)
│
├─ yolo_folds/
│   ├─ fold_1/
│   │   ├─ train/<class>/<img>.png
│   │   └─ val/<class>/<img>.png
│   ├─ fold_2/
│   └─ ...
│
├─ yolo_cls_runs/
│   ├─ pokemon_yolo_cls_fold1/
│   │   └─ weights/best.pt
│   ├─ pokemon_yolo_cls_fold2/
│   └─ ...
│
├─ reports/
│   ├─ test_grid_resnet18.png
│   ├─ test_classification_report_resnet18.txt
│   ├─ test_confusion_matrix_resnet18.csv
│   ├─ test_summary_aggregated.csv
│   └─ augmented/
│        ├─ test_grid_resnet18_white_noise.png
│        └─ ...
│
└─ test_summaries/
    ├─ test_summary_resnet18.csv
    └─ augmented/
        ├─ test_summary_resnet18_white_noise.csv
        └─ ...
```

## What belongs here?

This folder is intended to store **all outputs** generated automatically by:

- `training/preprocess_data.py`  
- `training/train_models.py`  
- `training/test_and_eval.py`  
- `training/eval_augmentations.py`  
- `training/model_comparison.py`  
- `pokeai.vision` (during inference)

These include:

### **Dataset metadata & splits**
- `classes.txt`
- `train_dev_split.csv`
- `test_split.csv`
- `folds/fold_k/train.csv`
- `folds/fold_k/val.csv`

### **Model checkpoints**
- `best_<model>_foldK.pth` (Torchvision)
- `yolo_cls_runs/<run>/weights/best.pt` (YOLOv8-CLS)

### **Evaluation outputs**
- per-model test summary CSVs
- confusion matrices
- per-image prediction CSVs
- rendered prediction grids
- classification reports
- aggregated summary across models

### **Robustness / augmentation outputs**
Stored under `reports/augmented` and `test_summaries/augmented`.

---

## What should NOT be committed?

Large binaries or datasets:

- `.pth` / `.pt` model weights  
- YOLO runs (`yolo_cls_runs/…/weights/best.pt`)  
- images copied into `yolo_folds/`  
- prediction grids (`*.png`)  

These are ignored by `.gitignore`.

---

## How other modules use this folder

- **pipeline + vision**  
  Load class names from `classes.txt` and model weights from
  `artifacts/best_<model>_foldK.pth` or YOLO runs.

- **train_models.py**  
  Saves checkpoints and validation summaries here.

- **test_and_eval.py**  
  Reads fold checkpoints and writes evaluation reports here.

- **model_comparison.py**  
  Aggregates all CSVs in `test_summaries/`.

---

## Tips

- If you move this folder, set the environment variable  
  `POKEAI_ARTIFACTS=/path/to/new/location`  
  so `vision.py` and training scripts find it correctly.

- If you want to share your trained models, do **not** push them to GitHub.
  Instead, upload them to HuggingFace, Dropbox, Google Drive, or similar,
  and provide a download script (e.g. `scripts/download_models.sh`).





