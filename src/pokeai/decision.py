"""
Decision logic for choosing the best move.

This module is a cleaned-up version of the notebook logic, adapted to
work inside the pokeai package, but keeping the same structure you used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .type_chart import get_type_effectiveness
from .data_models import Decision

# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------

def get_pokemon_moveset(pokemon_name, df_pokemon_moves):
    """
    Extracts the moves that a Pokémon can learn

    Returns a list with move names
    """
    # Find the Pokemon in the dataset
    pokemon_data = df_pokemon_moves[df_pokemon_moves['species'] == pokemon_name]
    
    if pokemon_data.empty:
        return []
    
    # Take the first row
    pokemon_data = pokemon_data.iloc[0]
    
    # Extract the moves (columns move1, move2, ...)
    move_cols = [col for col in df_pokemon_moves.columns if col.startswith('move')]
    
    moves = []
    for col in move_cols:
        move_entry = pokemon_data[col]
        if pd.notna(move_entry):
            # Format: "L7 - Ember" o "Start - Tackle"
            if ' - ' in str(move_entry):
                move_name = move_entry.split(' - ')[1]
                moves.append(move_name)
    
    return list(set(moves))

def calculate_move_damage(
    move_name,
    attacker_name,
    defender_name,
    df_pokemon,
    df_move_details
):
    """
    Calculates the damage of a move
    
    Returns: damage_score (float)
    """
    
    # 1. Get move details
    move_data = df_move_details[df_move_details['move'] == move_name]
    if move_data.empty:
        return 0.0  # Move not found
    
    move_data = move_data.iloc[0]
    
    # 2. Get attacker stats
    attacker_data = df_pokemon[df_pokemon['Name'] == attacker_name]
    if attacker_data.empty:
        return 0.0
    attacker_data = attacker_data.iloc[0]
    
    # 3. Get defender stats
    defender_data = df_pokemon[df_pokemon['Name'] == defender_name]
    if defender_data.empty:
        return 0.0
    defender_data = defender_data.iloc[0]
    
    # 4. Base power (if NaN = move status, return 0)
    power = move_data['power_clean']
    if pd.isna(power) or power == 0:
        return 0.0  # Status move 
    
    # 5. Type effectiveness
    move_type = move_data['type']
    defender_type1 = defender_data['Type 1']
    defender_type2 = defender_data['Type 2']
    type_multiplier = get_type_effectiveness(move_type, defender_type1, defender_type2)
    
    # 6. STAB (Same Type Attack Bonus)
    stab = 1.0
    if move_type == attacker_data['Type 1'] or move_type == attacker_data['Type 2']:
        stab = 1.5
    
    # 7. Physical vs Special
    category = move_data['category']
    if category == 'Physical':
        attack_stat = attacker_data['Attack']
        defense_stat = defender_data['Defense']
    elif category == 'Special':
        attack_stat = attacker_data['Sp. Attack']
        defense_stat = defender_data['Sp. Defense']
    else:  # Status
        return 0.0
    
    # 8. Accuracy
    accuracy = move_data['accuracy_clean']
    if pd.isna(accuracy):
        accuracy = 100  # Default for moves without specified accuracy 
    accuracy_factor = accuracy / 100.0

    # 9. SIMPLIFIED FORMULA (inspired by Pokémon damage formula)
    # damage = power * (attack/defense) * STAB * type_effectiveness * accuracy
    damage_score = power * (attack_stat / defense_stat) * stab * type_multiplier * accuracy_factor
    
    return damage_score

# ------------------------------------------------------------
# Ranking
# ------------------------------------------------------------

def rank_all_moves(attacker_name, defender_name, attacker_hp_percent, 
                   df_pokemon, df_pokemon_moves, df_move_details):
    """
    Builds a complete RANKING of all available moves for the attacker
    
    Parameters:
    - attacker_name: name of the attacking Pokémon
    - defender_name: name of the defending Pokémon
    - attacker_hp_percent: remaining HP % (0-100)
    
    Returns: ranking DataFrame [move_name, damage_score, rank]def rank_all_moves(attacker_name, defender_name, attacker_hp_percent, 
                   df_pokemon, df_pokemon_moves, df_move_details):
    """
    
    # 1. Get all available moves for attacker
    available_moves = get_pokemon_moveset(attacker_name, df_pokemon_moves)
    
    if not available_moves:
        return pd.DataFrame(columns=['move_name', 'damage_score', 'rank'])
    
    # 2. Calculate damage for each move
    move_scores = []
    for move in available_moves:
        damage = calculate_move_damage(move, attacker_name, defender_name, 
                                      df_pokemon, df_move_details)
        move_scores.append({
            'move_name': move,
            'damage_score': damage
        })
    
    # 3. Create DataFrame and sort
    ranking_df = pd.DataFrame(move_scores)
    ranking_df = ranking_df.sort_values('damage_score', ascending=False).reset_index(drop=True)
    ranking_df['rank'] = ranking_df.index + 1
    
    # 4. ADJUST for HP (if low HP, penalizes low accuracy moves)
    if attacker_hp_percent < 30:  # If critic HP
        # Obtain moves accuracy
        for idx, row in ranking_df.iterrows():
            move_data = df_move_details[df_move_details['move'] == row['move_name']]
            if not move_data.empty:
                accuracy = move_data.iloc[0]['accuracy_clean']
                if pd.notna(accuracy) and accuracy < 90:
                    # Penalize moves with low accuracy when HP is low
                    ranking_df.at[idx, 'damage_score'] *= 0.8
        
        # Re-sort after adjustment
        ranking_df = ranking_df.sort_values('damage_score', ascending=False).reset_index(drop=True)
        ranking_df['rank'] = ranking_df.index + 1
    
    return ranking_df

# ------------------------------------------------------------
# Main chooser
# ------------------------------------------------------------

def choose_best_move(attacker_name, defender_name, attacker_hp_percent,
                     available_moves_ocr, df_pokemon, df_pokemon_moves, df_move_details):
    """
    FINAL ALGORITHM: Chooses the best move among the 4 available
    
    Parameters:
    - attacker_name: name of the attacking Pokémon
    - defender_name: name of the defending Pokémon  
    - attacker_hp_percent: HP % (0-100)
    - available_moves_ocr: list of 4 available moves 
    
    Returns: dict con {
        'best_move': name of the chosen move,
        'damage_score': damage score,
        'rank_in_full_pool': chosen move ranking,
        'available_moves_ranking'
    }
    """
    
    # 1. Creates ranking of ALL moves
    full_ranking = rank_all_moves(attacker_name, defender_name, attacker_hp_percent,
                                  df_pokemon, df_pokemon_moves, df_move_details)
    
    if full_ranking.empty:
        return None
    
    # 2. Filtrates only the 4 available moves
    available_ranking = full_ranking[full_ranking['move_name'].isin(available_moves_ocr)]
    
    if available_ranking.empty:
        # No move is found
        return {
            'best_move': available_moves_ocr[0],  
            'damage_score': 0,
            'rank_in_full_pool': 999,
            'available_moves_ranking': available_ranking,
            'full_ranking': full_ranking.head(10),
            'warning': 'OCR moves not found in the database'
        }
    
    # 3. Choose the best among the 4
    best_move_data = available_ranking.iloc[0]
    
    return {
        'best_move': best_move_data['move_name'],
        'damage_score': best_move_data['damage_score'],
        'rank_in_full_pool': int(best_move_data['rank']),
        'available_moves_ranking': available_ranking,
    }

def pokemon_battle_advisor(my_pokemon, opponent_pokemon, my_hp_percent, 
                          ocr_moves, df_pokemon, df_pokemon_moves, df_move_details, show_details=True):
    """
    POKEMON BATTLE AI ADVISOR
    
    Input:
    - my_pokemon: name of player's pokemon 
    - opponent_pokemon: name of the opponent's Pokémon 
    - my_hp_percent: player HP % (0-100)
    - ocr_moves: list of ocr-recognized moves [move1, move2, move3, move4]
    - show_details (Bool): show detailed ranking
    
    Output: best move
    """
    
    result = choose_best_move(my_pokemon, opponent_pokemon, my_hp_percent,
                             ocr_moves, df_pokemon, df_pokemon_moves, df_move_details)
    
    if result is None:
        return "Pokémon not found!"
    
    if show_details:
        print("\n" + "="*70)
        print(f"BATTLE: {my_pokemon} vs {opponent_pokemon}")
        print(f"HP: {my_hp_percent}%")
        print("="*70)
        print(f"\n4 AVAILABLE MOVES: {ocr_moves}")
        print(f"\nBEST MOVE: {result['best_move'].upper()} ")
        print(f"   Damage Score: {result['damage_score']:.2f}")
        print(f"   Rank: #{result['rank_in_full_pool']}")
        
        if 'warning' not in result:
            print(f"\nRANKING OF THE 4 MOVES:")
            print(result['available_moves_ranking'][['move_name', 'damage_score', 'rank']].to_string(index=False))
    
    return result['best_move'], result['rank_in_full_pool']

# ------------------------------------------------------------
# Package-level advisor for the rest of the pipeline
# ------------------------------------------------------------

def advisor(
    state: Dict[str, Any],
    df_pokemon: pd.DataFrame,
    df_pokemon_moves: pd.DataFrame,
    df_move_details: pd.DataFrame,
) -> Decision:
    """
    High-level function used by the pipeline / CLI.

    Expects `state` to contain:
      - "attacker_name"
      - "defender_name"
      - "attacker_hp_percent"
      - "available_moves" (list of move names)
    """
    attacker = state.get("attacker_name")
    defender = state.get("defender_name")
    hp = float(state.get("attacker_hp_percent", 100.0))
    available_moves = state.get("available_moves", [])

    result = choose_best_move(
        attacker_name=attacker,
        defender_name=defender,
        attacker_hp_percent=hp,
        available_moves_ocr=available_moves,
        df_pokemon=df_pokemon,
        df_pokemon_moves=df_pokemon_moves,
        df_move_details=df_move_details,
    )

    if result is None:
        return Decision(best_move=None, reasoning={"error": "Pokémon not found in database"})

    if "available_moves_ranking" in result and not result["available_moves_ranking"].empty:
       available_ranking_head = (
           result["available_moves_ranking"][["move_name", "damage_score", "rank"]]
           .head(10)
           .to_dict(orient="records")
       )
    else:
       available_ranking_head = []   # no ranking available

    warning = result.get("warning", None)

    return Decision(
        best_move=result["best_move"],
        reasoning={
            "damage_score": result["damage_score"],
            "rank_in_full_pool": result["rank_in_full_pool"],
            "available_moves_ranking_head": available_ranking_head,
            "warning": warning
        },
    )

