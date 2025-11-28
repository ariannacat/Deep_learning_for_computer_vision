"""
Global constants for the pokeai package.
"""

from pathlib import Path
import os

# Directories
CONFIGS_DIR = Path(os.getenv("POKEAI_CONFIGS_DIR", "configs")).resolve()
DATA_DIR = Path(os.getenv("POKEAI_DATA_DIR", "data")).resolve()
ARTIFACTS = Path(os.getenv("POKEAI_ARTIFACTS", "artifacts")).resolve()

# Device
try:
    import torch
    DEVICE = torch.device(os.getenv("POKEAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
except Exception:
    DEVICE = "cpu"

# Default
DEFAULT_CONFIG_PATH = Path(os.getenv("POKEAI_DEFAULT_CONFIG", CONFIGS_DIR / "default.yaml")).resolve()
DEFAULT_POKEMON_CSV = Path(os.getenv("POKEAI_POKEMON_CSV", "data/csv/pokemon_data.csv")).resolve()
DEFAULT_MOVESET_CSV = Path(os.getenv("POKEAI_MOVESET_CSV", "data/csv/movesets.csv")).resolve()
DEFAULT_MOVE_DETAILS_CSV = Path(os.getenv("POKEAI_MOVE_DETAILS_CSV", "data/csv/moves.csv")).resolve()
