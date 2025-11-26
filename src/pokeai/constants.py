"""
Global constants for the pokeai package.
"""

from pathlib import Path

# Root paths (can be useful later)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Default config file used by CLI and scripts
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"

# You can add other global thresholds/constants later, e.g.:
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
