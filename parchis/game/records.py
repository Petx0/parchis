"""
Data records describing move/turn outcomes.

These replace the ad-hoc dicts previously threaded through Game, GameLogger,
and the RL environment: every producer built its own set of keys and every
consumer read them back with .get()/[...], with no single place documenting
what fields actually exist.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MoveInfo:
    """Result of executing a single move (entry, board move, or finish)."""
    piece: object
    old_position: Optional[int]
    new_position: int
    move_type: str  # 'enter', 'move', or 'finish'
    captured: list = field(default_factory=list)


@dataclass
class RollEntry:
    """
    One entry in a turn's roll history: either a regular dice roll or a bonus
    move triggered by a capture/finish. `is_bonus` distinguishes the two.
    """
    legal_moves_count: int
    chosen_move: Optional[tuple] = None
    move_info: Optional[MoveInfo] = None
    dice_roll: Optional[int] = None
    bonus_type: Optional[str] = None  # 'capture_bonus' or 'finish_bonus'
    bonus_squares: Optional[int] = None

    @property
    def is_bonus(self):
        return self.bonus_type is not None


@dataclass
class TurnInfo:
    """Full record of a single turn, including all rolls and bonus moves."""
    turn_number: int
    player: object
    rolls: list = field(default_factory=list)
    game_over: bool = False
    winner: object = None
    three_sixes_penalty: bool = False
    penalty_piece: object = None
    home_entry_protection: bool = False
