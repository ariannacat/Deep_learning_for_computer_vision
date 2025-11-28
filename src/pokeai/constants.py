"""
Global constants for the pokeai package.
"""

from pathlib import Path
import os
import torch

PROJECT_ROOT = Path(os.getenv("POKEAI_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS_DIR = Path(os.getenv("POKEAI_CONFIGS_DIR", PROJECT_ROOT / "configs"))
DATASET_DIR = Path(os.getenv("POKEAI_DATASET_DIR", PROJECT_ROOT / "data"))
DEFAULT_CONFIG_PATH = Path(os.getenv("POKEAI_DEFAULT_CONFIG", CONFIGS_DIR / "default.yaml"))

DEVICE = torch.device(os.getenv("POKEAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

ARTIFACTS = Path(os.getenv("POKEAI_ARTIFACTS", PROJECT_ROOT / "artifacts")).resolve()
