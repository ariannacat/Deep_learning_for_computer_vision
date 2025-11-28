"""
High-level pipeline to go from a battle screenshot to a move recommendation.

Steps:
1. Load decision CSVs (Pokémon, moves, movesets).
2. Initialise OCR move dictionary for the screenshot parser.
3. Load screenshot image.
4. Parse UI elements (sprites, HP bars, moves) with `screengrab_parser`.
5. Recognize our Pokémon and the opponent via `vision.predict_image`.
6. Call `decision.advisor` to choose the best move.

This is what the CLI (`cli/pokeai_cli.py`) should call.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import cv2

from . import io_utils
from .decision import advisor
from .screengrab_parser import extract_from_screenshot, set_move_dictionary
from .vision import predict_image
from .data_models import Decision
from difflib import get_close_matches
from typing import List

# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _load_decision_data():
    """
    Load the three CSVs used by the decision logic.

    Returns:
        df_pokemon, df_move_details, df_pokemon_moves
    """
    df_pokemon, df_pokemon_moves, df_move_details = io_utils.load_datasets()
    return df_pokemon, df_move_details, df_pokemon_moves


def _init_move_dictionary(df_move_details) -> None:
    """
    Initialise the global MOVE_DICTIONARY used by screengrab_parser
    from the canonical moves CSV.
    """
    if "move" not in df_move_details.columns:
        # Adapt here if your column is called differently
        raise KeyError("Expected column 'move' in df_move_details.")
    moves = df_move_details["move"].dropna().astype(str).unique().tolist()
    set_move_dictionary(moves)


def _read_image_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _extract_hp_percent(parse_result: Dict[str, Any]) -> float:
    """
    Try to read our HP percentage from the parse result.
    Falls back to 100.0 if not available.
    """
    hp_block = parse_result.get("hp", {})
    ours = hp_block.get("ours")
    if not ours:
        return 100.0
    hp_pct = ours.get("hp_percent")
    if hp_pct is None:
        return 100.0
    try:
        return float(hp_pct)
    except (TypeError, ValueError):
        return 100.0


def _recognize_pokemon_from_sprites(parse_result: Dict[str, Any], cfg) -> Tuple[str, str]:
    """
    Use the cropped sprite images created by `extract_from_screenshot`
    to predict our Pokémon and the opponent with the current model.

    Assumes `extract_from_screenshot` wrote `own_sprite_file` and
    `opp_sprite_file` on disk and stored their filenames in `parse_result`.
    """
    sprites = parse_result.get("sprites", {})
    own_path = sprites.get("own_sprite_file")
    opp_path = sprites.get("opp_sprite_file")

    if not own_path or not opp_path:
        raise RuntimeError(
            "Sprite filenames not found in parse result. "
            "Check screengrab_parser.extract_from_screenshot."
        )

    our_name, our_conf = predict_image(own_path, cfg)
    opp_name, opp_conf = predict_image(opp_path, cfg)
    return our_name, opp_name

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def run_pipeline(screenshot_path: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full pipeline: screenshot → parse → recognize → decide.

    Parameters
    ----------
    screenshot_path : str
        Path to the battle screenshot (PNG/JPG).

    Returns
    -------
    result : dict
        {
          "parse": {...},          # raw output from extract_from_screenshot
          "recognition": {...},   # recognized Pokémon + HP + OCR moves
          "decision": {...},      # serialized Decision dataclass
        }
    """

    # 1) Load decision CSVs
    df_pokemon, df_move_details, df_pokemon_moves = _load_decision_data()

    # 2) Initialise the move dictionary for OCR correction
    _init_move_dictionary(df_move_details)

    # 3) Load screenshot
    img = _read_image_bgr(screenshot_path)

    # 4) Parse UI elements (sprites, HP bars, moves)
    parse_result = extract_from_screenshot(img)

    # 5) Recognize Pokémon from sprite crops
    our_name, opp_name = _recognize_pokemon_from_sprites(parse_result, cfg)

    # 6) Infer our HP percentage
    hp_percent = _extract_hp_percent(parse_result)

    # 7) Moves from OCR (list of strings)
    moves_ocr = parse_result.get("moves", []) or []

    # 8) Build battle state for the advisor
    state = {
        "attacker_name": our_name,
        "defender_name": opp_name,
        "attacker_hp_percent": hp_percent,
        "available_moves": moves_ocr,
    }

    # 9) Call decision logic
    decision: Decision = advisor(
        state=state,
        df_pokemon=df_pokemon,
        df_pokemon_moves=df_pokemon_moves,
        df_move_details=df_move_details,
    )

    # 10) JSON-friendly output
    result: Dict[str, Any] = {
        "parse": parse_result,
        "recognition": {
            "our_pokemon": our_name,
            "opponent_pokemon": opp_name,
            "our_hp_percent": hp_percent,
            "moves_ocr": moves_ocr,
        },
        "decision": asdict(decision),
    }
    return result

