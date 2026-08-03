"""
Player module for Parchís game.
"""

import random
from parchis.game.piece import Piece
from parchis.game.board import Board


class Player:
    """Represents a player in the game."""

    COLORS = ["YELLOW", "BLUE", "RED", "GREEN"]

    def __init__(self, player_id, color, starting_position=None):
        """
        Initialize a player.

        Args:
            player_id: Unique identifier for the player (0-3)
            color: Player's color (YELLOW, BLUE, RED, or GREEN)
            starting_position: Starting position on the board (if None, uses default from Board)
        """
        self.player_id = player_id
        self.color = color
        self.pieces = [Piece(i, color) for i in range(4)]

        # Set starting position based on color
        if starting_position is None:
            starting_position = Board.STARTING_POSITIONS[color]
        self.starting_position = starting_position

        # Set home entry point based on color
        self.home_entry_point = Board.HOME_ENTRY_POINTS[color]

        # Start with one piece at starting square
        self.pieces[0].move_to(starting_position)

    def get_pieces_in_base(self):
        """Get all pieces currently in the base."""
        return [piece for piece in self.pieces if piece.in_base]

    def get_pieces_on_board(self):
        """Get all pieces currently on the board (not in base, not finished)."""
        return [piece for piece in self.pieces if not piece.in_base and not piece.finished]

    def get_finished_pieces(self):
        """Get all pieces that have finished the game."""
        return [piece for piece in self.pieces if piece.finished]

    def has_won(self):
        """Check if the player has won (all pieces finished)."""
        return len(self.get_finished_pieces()) == 4

    def choose_move(self, legal_moves):
        """
        Choose a move from the list of legal moves.
        In version 1, this is random selection.

        Args:
            legal_moves: List of tuples (piece, new_position)

        Returns:
            tuple: Selected (piece, new_position) or None if no moves available
        """
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def __repr__(self):
        """String representation of the player."""
        return f"Player({self.color})"
