#!/bin/bash
# Usage:
#   chmod +x scripts/download_models.sh    # make executable (run once)
#   ./scripts/download_models.sh           # run script
#
# This script downloads model weights. 

echo "Downloading model weights..."
mkdir -p models/

# Example (later you will replace these with real URLs)
# gdown --id 1hZ3ABCDE -O models/yolo_cls/yolov8s-pokemon.pt
# gdown --id 1gY6FGHIJ -O models/torchvision/resnet50/resnet50-pokemon_best.pt

echo "Done."

