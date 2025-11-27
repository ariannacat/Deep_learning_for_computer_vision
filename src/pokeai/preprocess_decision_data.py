"""
Utilities to preprocess the decision CSVs (Pokemon, movesets, move details)
to match the cleaning done in DECISION_ALGORITHM.ipynb.

This:
- creates power_clean and accuracy_clean in df_move_details
- adds weakness / resistance counters to df_pokemon
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def preprocess_decision_data(
    df_pokemon: pd.DataFrame,
    df_pokemon_moves: pd.DataFrame,
    df_move_details: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the same cleaning you used in DECISION_ALGORITHM.ipynb.

    Returns:
        df_pokemon_clean, df_pokemon_moves (unchanged), df_move_details_clean
    """

    # -----------------------------
    # 1) Clean move power/accuracy
    # -----------------------------
    df_move_details = df_move_details.copy()

    # Power -> numeric (turn "—" into NaN, coerce)
    df_move_details["power_clean"] = (
        df_move_details["power"]
        .replace("—", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Accuracy -> numeric percent (strip %, turn "—" into NaN)
    acc = df_move_details["accuracy"].astype(str).str.replace("%", "", regex=False)
    acc = acc.replace("—", np.nan)
    df_move_details["accuracy_clean"] = pd.to_numeric(acc, errors="coerce")

    # -----------------------------
    # 2) Weakness / resistance counters on df_pokemon
    # -----------------------------
    df_pokemon = df_pokemon.copy()

    # Weakness columns typically end with "_weakness"
    weakness_cols = [c for c in df_pokemon.columns if c.endswith("_weakness")]

    if weakness_cols:
        df_pokemon["count_weaknesses_2x"] = (df_pokemon[weakness_cols] == 2.0).sum(
            axis=1
        )
        df_pokemon["count_weaknesses_4x"] = (df_pokemon[weakness_cols] == 4.0).sum(
            axis=1
        )
        df_pokemon["count_resistances"] = (df_pokemon[weakness_cols] == 0.5).sum(axis=1)
        df_pokemon["count_immunities"] = (df_pokemon[weakness_cols] == 0.0).sum(axis=1)

    # df_pokemon_moves is not changed by this cleaning
    return df_pokemon, df_pokemon_moves, df_move_details
