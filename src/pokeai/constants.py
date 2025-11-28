"""
Global constants for the pokeai package.
"""

from pathlib import Path
import os
import torch

# Directories
PROJECT_ROOT = Path(os.getenv("POKEAI_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS_DIR = Path(os.getenv("POKEAI_CONFIGS_DIR", PROJECT_ROOT / "configs"))
DATA_DIR = Path(os.getenv("POKEAI_DATA_DIR", "data")).resolve()
ARTIFACTS = Path(os.getenv("POKEAI_ARTIFACTS", "artifacts")).resolve()

# Device
DEVICE = torch.device(os.getenv("POKEAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

# Default
DEFAULT_CONFIG_PATH = Path(os.getenv("POKEAI_DEFAULT_CONFIG", CONFIGS_DIR / "default.yaml"))
DEFAULT_POKEMON_CSV = Path(os.getenv("POKEAI_POKEMON_CSV", "data/csv/pokemon_data.csv"))
DEFAULT_MOVESET_CSV = Path(os.getenv("POKEAI_MOVESET_CSV", "data/csv/movesets.csv"))
DEFAULT_MOVE_DETAILS_CSV = Path(os.getenv("POKEAI_MOVE_DETAILS_CSV", "data/csv/moves.csv"))
