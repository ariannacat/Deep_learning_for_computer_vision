"""
Utility functions for loading CSV datasets used by the decision engine.
"""

from .preprocess_decision_data import preprocess_decision_data
from pathlib import Path
from typing import List, Union
import pandas as pd


DEFAULT_POKEMON_CSV = "data/csv/pokemon_data.csv"
DEFAULT_MOVESET_CSV = "data/csv/movesets.csv"
DEFAULT_MOVE_DETAILS_CSV = "data/csv/moves.csv"


def load_datasets(
    pokemon_path: str = DEFAULT_POKEMON_CSV,
    moveset_path: str = DEFAULT_MOVESET_CSV,
    move_details_path: str = DEFAULT_MOVE_DETAILS_CSV,
):
    """
    Load the 3 CSV files used by the decision engine.

    Returns:
        df_pokemon, df_pokemon_moves, df_move_details
    """
    p = Path(pokemon_path)
    mset = Path(moveset_path)
    md = Path(move_details_path)

    if not p.is_file():
        raise FileNotFoundError(f"Missing Pokémon dataset: {pokemon_path}")
    if not mset.is_file():
        raise FileNotFoundError(f"Missing moveset dataset: {moveset_path}")
    if not md.is_file():
        raise FileNotFoundError(f"Missing move details dataset: {move_details_path}")

    df_pokemon = pd.read_csv(p)
    df_pokemon_moves = pd.read_csv(mset)
    df_move_details = pd.read_csv(md)

    # Apply the same cleaning as in DECISION_ALGORITHM.ipynb
    df_pokemon, df_pokemon_moves, df_move_details = preprocess_decision_data(
        df_pokemon, df_pokemon_moves, df_move_details
    )

    return df_pokemon, df_pokemon_moves, df_move_details

def load_move_name_list(move_details_path: str = DEFAULT_MOVE_DETAILS_CSV) -> List[str]:
    """
    Load the list of canonical move names from the move details CSV.

    Expected CSV structure:
        column 'move' contains the canonical move name (e.g., 'Thunderbolt')
    """
    df = pd.read_csv(move_details_path)
    if "move" not in df.columns:
        raise ValueError(f"`move` column not found in {move_details_path}")

    moves = (
        df["move"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    return moves

def load_classes_txt(path: Union[str, Path]) -> List[str]:
    """
    Load class names (one per line) from a classes.txt file.

    Used by the vision module to know the index→label mapping.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"classes.txt not found at {p}")

    lines = p.read_text(encoding="utf-8").splitlines()
    classes = [ln.strip() for ln in lines if ln.strip()]
    if not classes:
        raise ValueError(f"No classes found in {p}")
    return classes

