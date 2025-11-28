# Training Suite

This folder contains all the scripts required to **prepare data**, **train models**, **evaluate performance**, **compare models**, and **test robustness to augmentations**  
for the Pokémon species classifier used by the main `pokeai` pipeline.

The pipeline followed is:

> preprocess → train → evaluate → compare → robustness testing

All outputs (splits, weights, metrics, plots) are written under `artifacts/`.

---

## 📁 Files in this folder

### **1. `preprocess_data.py`**
Builds the dataset index and all cross-validation splits.

This script:
- indexes the image dataset under `data/dataset/`
- removes unreadable / duplicate images
- enforces minimum samples per class
- creates:
  - `artifacts/train_dev_split.csv`
  - `artifacts/test_split.csv`
  - `artifacts/folds/fold_k/train.csv`, `val.csv`
  - `artifacts/classes.txt` (class vocabulary)

**Run:**
```bash
python -m training.preprocess_data
```

---

### **2. `train_models.py`**
Trains a classifier with **5-fold cross-validation**.

Supports:
- **Torchvision** models (ResNet, VGG, …)
- **YOLOv8-CLS** classification models

The script:
- trains one model per fold  
- performs early stopping  
- handles VGG warmup + unfreeze  
- saves per-fold checkpoints
- saves per-fold validation summaries to:
  - `artifacts/fold_summaries/<MODEL_NAME>.csv`

**Select model & train:**
```bash
export MODEL_NAME=resnet18
python -m training.train_models
```

Other options:
```bash
export MODEL_NAME=resnet50
export MODEL_NAME=vgg16_bn
export MODEL_NAME=yolov8n-cls
```

---

### **3. `test_and_eval.py`**
Performs **test-set evaluation** using the trained cross-validation ensemble.

This script:
- loads the best checkpoint for each fold  
- ensembles their predictions  
- evaluates on `artifacts/test_split.csv`
- saves:
  - test metrics (`accuracy`, `macro-F1`, `macro-precision`)  
  - per-image predictions  
  - classification report  
  - confusion matrix (CSV)  
  - prediction grid image  

Saved to:
```
artifacts/test_summaries/
artifacts/reports/
```

**Run:**
```bash
export MODEL_NAME=resnet18
python -m training.test_and_eval
```

---

### **4. `eval_augmentation.py`**

Evaluates **model robustness** under several **test-time augmentations**:

- White noise  
- Random occlusions  
- Random crop-and-resize  
- Lighting/contrast changes  
- A combined augmentation

For each augmentation, it:
- ensembles predictions across folds  
- computes metrics (ACC, F1, Precision)  
- saves per-image predictions  
- saves confusion matrices  
- produces a 4×4 prediction grid  
- writes summary CSVs

Outputs go to:

```
artifacts/reports/augmented/
artifacts/test_summaries/augmented/
```

**Run:**
```bash
export MODEL_NAME=resnet18
python -m training.eval_agumentation
```

---

### **5. `model_comparison.py`**
Aggregates **all test summaries** from `test_and_eval.py`  
(and optionally from augmented robustness tests) and provides:

- a summary table comparing models  
- an aggregated CSV  

Uses all CSVs under:

```
artifacts/test_summaries/
```

**Run:**
```bash
python -m training.model_comparison
```

---

## 📂 Output layout (all auto-generated)

```
artifacts/
│
├─ classes.txt
├─ train_dev_split.csv
├─ test_split.csv
│
├─ folds/
│   ├─ fold_1/
│   │   ├─ train.csv
│   │   └─ val.csv
│   └─ ...
│
├─ best_resnet18_fold1.pth               # Torchvision weights
├─ yolo_cls_runs/pokemon_yolo_cls_fold1/
│       └─ weights/best.pt               # YOLOv8 weights
│
├─ test_summaries/
│   ├─ test_summary_resnet18.csv
│   └─ augmented/...
│
└─ reports/
    ├─ test_predictions_resnet18.csv
    ├─ test_grid_resnet18.png
    └─ augmented/...
```

---

## 📝 Notes

- These scripts **do not ship data or weights**: you generate them locally.
- GPU strongly recommended for YOLO and VGG models.
- All settings (paths, hyperparameters, dataset root, etc.) are configurable via
  environment variables or by editing the scripts.

--- 
