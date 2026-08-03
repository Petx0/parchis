"""
Piece module for Parchís game.
"""


class Piece:
    """Represents a single game piece."""

    def __init__(self, piece_id, color):
        """
        Initialize a game piece.

        Args:
            piece_id: Unique identifier for the piece (0-3)
            color: Color of the piece (matches player color)
        """
        self.piece_id = piece_id
        self.color = color
        self.position = None  # None means in base, 1-76 means on board
        self.in_base = True
        self.finished = False
        self.move_order = 0  # Track when piece was last moved to current position

    def move_to(self, position):
        """
        Move the piece to a new position.

        Args:
            position: The new position (1-76)
        """
        self.position = position
        if self.in_base:
            self.in_base = False

    def send_to_base(self):
        """Send the piece back to the base (captured)."""
        self.position = None
        self.in_base = True
        self.finished = False

    def mark_finished(self):
        """Mark the piece as having reached the final position."""
        self.finished = True
        self.position = 76

    def __repr__(self):
        """String representation of the piece."""
        status = "BASE" if self.in_base else (f"POS:{self.position}" if not self.finished else "FINISHED")
        return f"{self.color}[{self.piece_id}]@{status}"
