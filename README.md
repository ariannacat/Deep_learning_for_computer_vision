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
pip install -e .

---

## Usage
```bash
pokeai run data/samples/example.png --config configs/default.yaml

## License
MIT License (see LICENSE file)


