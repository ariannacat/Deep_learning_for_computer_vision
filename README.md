# Pokémon Battle Assistant

This project combines computer vision and game logic to automatically read a Pokémon battle screenshot and decide the **best move** to use.

---

## Overview
**Pipeline:**
1. Parse a screenshot (OCR + cropping)
2. Recognize active Pokémon using YOLO or Torchvision (chosen model)
3. Extract HP and available moves
4. Compute damage multipliers and choose the most effective move

---

## Structure
- `src/pokeai/`: Core Python package (logic, OCR, decision)
- `cli/`: Command-line entry point (`pokeai` command)
- `scripts/`: Utilities and examples
- `configs/`: Model and threshold configuration files
- `tests/`: Unit and integration tests

---

## Installation
```bash
pip install -e .```

---

## Usage
```bash
pokeai run data/samples/example.png --config configs/default.yaml```

## Setup
Before running the project, make sure the helper scripts are executable.
Run these commands once from the project root:
```bash
chmod +x scripts/download_models.sh
chmod +x scripts/setup_tesseract.sh
chmod +x scripts/dev_install.sh```

````bash
pip install -e ".[dev]"```

Install Tesseract OCR (required for text extraction):

* MacOS:
````bash
brew install tesseract```

* Windows:
Download the installer from: https://github.com/UB-Mannheim/tesseract/wiki 

Finally, download the model weights (YOLO/Torchvision) stored externally:
````bash
./scripts/download_models.sh```

## License
MIT License (see LICENSE file)


