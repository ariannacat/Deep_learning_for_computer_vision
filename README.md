# Pokémon Battle Advisor — Full Pipeline 

This repository implements a complete AI assistant for Pokémon battles:

1. Parse a battle screenshot (YOLO-style UI parser + OCR)  
2. Extract sprites, HP bars, and move names  
3. Recognize both Pokémon using a Torchvision or YOLO-CLS classifier  
4. Load cleaned decision datasets  
5. Compute the best move using a Pokémon knowledge base  
6. Output the chosen move and reasoning

---

## Quick Start (Run the Example End-to-End)

Make sure you are in the project root:

```bash
cd ~/Deep_learning_for_computer_vision
```
1. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the package + dependencies (editable mode)
```bash
pip install -e .
```
3. Install Tesseract OCR (required for text extraction):
* MacOS:
```bash
brew install tesseract
````
* Windows:
Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki 

4. Run the full pipeline on the example image
```bash
pokeai run --image data/example.png --config configs/default.yaml
```

This will:
* Parse the screenshot
* Extract the sprites
* Recognize both Pokémon
* OCR the available moves
* Load the decision CSVs (preprocessed automatically)
* Compute the best move
* Print a structured JSON-like output with all components

### Quick Debug: See Only Which Pokémon Are Recognized

You can use the built-in debug command:

```bash
pokeai recognize --image data/example.png --config configs/default.yaml
```
## Model Selection (Torchvision / YOLO)

### Training Your Own Models 

This repository includes a `training/` directory with all the code needed to:
* train Torchvision classifiers (ResNet, VGG, etc.)
* train YOLOv8-CLS models
* generate folds and evaluation reports
* export weights compatible with the main pipeline

If you want to train your own models, follow the instructions inside:

```markdown
training/
    torchvision/
    yolo_cls/
    utils/
```

Trained weights should be saved using the naming convention:

```markdown
best_<MODEL_NAME>_foldK.pth                 (Torchvision)
yolo_cls_runs/<NAME>_foldK/weights/best.pt  (YOLO-CLS)
```

### Using Your Own Weights (without retraining)

If you already have pretrained weights, you do not need to overwrite the repository’s artifacts/ folder.

Instead, create your own directory anywhere on your filesystem, e.g.:

```swift
/home/user/my_pokeai_models/
    classes.txt
    best_resnet50_fold1.pth
    best_resnet50_fold2.pth
    ...
    yolo_cls_runs/
```

Then simply point the pipeline to your folder by exporting:

```bash
export POKEAI_ARTIFACTS="/home/user/my_pokeai_models"
```
This environment variable tells the system:
* where classes.txt is located
* where to look for weight files
* where YOLO run folders live
  
The vision system reads model settings from your config file:
```yaml
model:
  backend: "torchvision"      # "torchvision" or "yolo"
  name: "resnet50"            # e.g. vgg16, resnet18, resnet50, yolov8n-cls
  folds: 5                    # number of CV folds you trained
  dir: "artifacts"            # where weights live, ignored if POKEAI_ARTIFACTS is set
  yolo_name: "pokemon_yolo_cls"
```
You can switch models simply by changing the config.

All weights must be in:
```bash
POKEAI_ARTIFACTS/
  best_resnet50_fold1.pth
  best_resnet50_fold2.pth
  ...
  yolo_cls_runs/pokemon_yolo_cls_fold1/weights/best.pt
  ...
  classes.txt
```
# Data Requirements
Place your datasets under:
```kotlin
data/
  csv/
    pokemon_data.csv
    moves.csv
    movesets.csv
```
The loader automatically:
* cleans move power/accuracy (power_clean, accuracy_clean)
* computes weakness/resistance counters in Pokémon data
* normalizes names

# Project Structure
(aggiorno domani non ho tempo)

# License
MIT License (see LICENSE file)


