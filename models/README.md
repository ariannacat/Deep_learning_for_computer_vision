# Artifacts

This directory will store all trained model weights and related artifacts.

## Structure
models/
├─ yolo_cls/
│ ├─ yolov8s-pokemon.pt # (future YOLOv8 small model)
│ └─ yolov8m-pokemon.pt # (future YOLOv8 medium model)
├─ torchvision/
│ ├─ resnet50/
│ │ ├─ resnet50-pokemon_best.pt
│ │ └─ class_to_idx.json
│ └── vgg16_bn/
│   ├─ vgg16_bn-pokemon_best.pt
│   └─ class_to_idx.json
└─ exports/
├─ onnx/
└─ torchscript/


## Notes
- This folder is **currently empty** — you’ll add weights here after training or downloading.
- Do **not commit** model weights to GitHub (`.gitignore` already excludes them).
- Each classifier will include:
  - its `.pt` weight file,
  - and a `class_to_idx.json` mapping file.
- The folder structure allows you to manage multiple backends (YOLO, Torchvision, etc.)
