"""
Parchís Game Simulation Module.

This module contains the core game logic for Parchís, including:
- Game: Main game controller
- Board: Board representation and piece tracking
- Player: Player logic and move selection
- Piece: Individual piece state and movement
- Dice: Dice rolling functionality
"""

from parchis.game.game import Game
from parchis.game.board import Board
from parchis.game.player import Player
from parchis.game.piece import Piece
from parchis.game.dice import Dice

__all__ = ['Game', 'Board', 'Player', 'Piece', 'Dice']
