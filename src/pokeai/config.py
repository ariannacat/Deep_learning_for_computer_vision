"""
Configuration loading utilities for pokeai.

Right now this is intentionally simple:
- load_config(path) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .constants import DEFAULT_CONFIG_PATH


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a Python dict.

    If no path is provided, uses DEFAULT_CONFIG_PATH from constants.py.
    """
    if path is None:
        cfg_path = Path(DEFAULT_CONFIG_PATH)
    else:
        cfg_path = Path(path)

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg

