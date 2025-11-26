"""
Dataclasses used across the pokeai pipeline.

These give structure to:
- parsed screenshot data
- Pokémon and Move information
- final decision output

They are intentionally lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ------------------------------------------------------------
# Basic units
# ------------------------------------------------------------

@dataclass
class Move:
    name: str
    move_type: Optional[str] = None
    power: Optional[int] = None
    accuracy: Optional[int] = None


@dataclass
class Pokemon:
    name: str
    types: List[str] = field(default_factory=list)
    hp: Optional[int] = None          # % HP or absolute
    moves: List[Move] = field(default_factory=list)


# ------------------------------------------------------------
# Parsed screenshot result
# ------------------------------------------------------------

@dataclass
class ParseResult:
    """
    Standardized version of what extract_from_screenshot returns.
    Allows the pipeline to work even if the raw parser changes later.
    """
    active_pokemon: Optional[Pokemon] = None
    opponent_pokemon: Optional[Pokemon] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Convert dataclass to standard dict for JSON printing
        return {
            "active_pokemon": self._pokemon_to_dict(self.active_pokemon),
            "opponent_pokemon": self._pokemon_to_dict(self.opponent_pokemon),
            "raw": self.raw,
        }

    @staticmethod
    def _pokemon_to_dict(p: Optional[Pokemon]) -> Optional[Dict[str, Any]]:
        if p is None:
            return None
        return {
            "name": p.name,
            "types": p.types,
            "hp": p.hp,
            "moves": [{
                "name": m.name,
                "move_type": m.move_type,
                "power": m.power,
                "accuracy": m.accuracy,
            } for m in p.moves]
        }


# ------------------------------------------------------------
# Best-move decision result
# ------------------------------------------------------------

@dataclass
class Decision:
    best_move: Optional[str]
    reasoning: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_move": self.best_move,
            "reasoning": self.reasoning,
        }

