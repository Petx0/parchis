"""
Board module for Parchís game.
"""


class Board:
    """Represents the Parchís game board."""

    # Safe squares on the main track (in absolute positions)
    # Starting squares: 5 (Yellow), 22 (Blue), 39 (Red), 56 (Green)
    # Additional safe squares: 12, 17, 29, 34, 46, 51, 63, 68
    SAFE_SQUARES = {5, 12, 17, 22, 29, 34, 39, 46, 51, 56, 63, 68}

    # Main circular track has 68 squares (1-68)
    MAIN_TRACK_SIZE = 68
    # Home column has 8 squares (69-76)
    HOME_COLUMN_SIZE = 8
    FINAL_POSITION = 76

    # Player starting positions on the main track
    STARTING_POSITIONS = {
        'YELLOW': 5,
        'BLUE': 22,
        'RED': 39,
        'GREEN': 56
    }

    # Home entry points for each player (square after which they enter home)
    HOME_ENTRY_POINTS = {
        'YELLOW': 68,
        'BLUE': 17,
        'RED': 34,
        'GREEN': 51
    }

    # Home column starting position
    HOME_COLUMN_START = 69

    # Maximum number of pieces allowed on a single square (stacking rule)
    MAX_PIECES_PER_SQUARE = 2

    def __init__(self):
        """Initialize the board."""
        # Track which pieces are at which positions
        # Key: position (1-76), Value: list of pieces at that position
        self.positions = {}
        # Global move counter to track order of piece movements
        self.move_counter = 0

    def get_pieces_at(self, position):
        """
        Get all pieces at a given position.

        Args:
            position: The position to check (1-76)

        Returns:
            list: List of pieces at that position
        """
        return self.positions.get(position, [])

    def is_safe_square(self, position):
        """
        Check if a position is a safe square.

        Args:
            position: The position to check (1-76, only applies to main track 1-68)

        Returns:
            bool: True if the position is safe
        """
        return position in self.SAFE_SQUARES

    def add_piece(self, piece, position, update_move_order=True):
        """
        Add a piece to a position on the board.

        Args:
            piece: The piece to add
            position: The position (1-76)
            update_move_order: Whether to update the piece's move_order (default True)
        """
        if position not in self.positions:
            self.positions[position] = []
        self.positions[position].append(piece)

        if update_move_order:
            self.move_counter += 1
            piece.move_order = self.move_counter

    def remove_piece(self, piece):
        """
        Remove a piece from the board.

        Args:
            piece: The piece to remove
        """
        if piece.position is not None and piece.position in self.positions:
            if piece in self.positions[piece.position]:
                self.positions[piece.position].remove(piece)
                if not self.positions[piece.position]:
                    del self.positions[piece.position]

    def enter_piece(self, piece, starting_position):
        """
        Move a piece from base onto its starting square, applying
        entry-specific capture rules. Unlike move_piece's capture rule
        (which only captures on non-safe squares), the starting square is
        always safe, so entry follows a distinct rule set:
          - 0 or 1 opponent piece alone: no capture, piece just enters.
          - 1 own + 1 opponent: capture the opponent.
          - 2 opponents: capture whichever was moved there most recently.

        Args:
            piece: The piece entering from base
            starting_position: The starting square position

        Returns:
            list: List of captured pieces
        """
        pieces_at_start = self.get_pieces_at(starting_position)
        own_pieces = [p for p in pieces_at_start if p.color == piece.color]
        opponent_pieces = [p for p in pieces_at_start if p.color != piece.color]

        captured = []

        if len(own_pieces) == 1 and len(opponent_pieces) == 1:
            captured_piece = opponent_pieces[0]
            self.remove_piece(captured_piece)
            captured_piece.send_to_base()
            captured.append(captured_piece)
        elif len(opponent_pieces) == self.MAX_PIECES_PER_SQUARE:
            # Find the piece with the highest move_order (most recent)
            most_recent = max(opponent_pieces, key=lambda p: p.move_order)
            self.remove_piece(most_recent)
            most_recent.send_to_base()
            captured.append(most_recent)

        piece.move_to(starting_position)
        self.add_piece(piece, starting_position)

        return captured

    def move_piece(self, piece, new_position):
        """
        Move a piece to a new position.

        Args:
            piece: The piece to move
            new_position: The target position (1-76)

        Returns:
            list: List of captured pieces (if any)
        """
        # Remove piece from current position
        self.remove_piece(piece)

        # Check for captures at the new position
        # Captures can only happen on main track (1-68), not safe squares or home columns (69-76)
        captured = []
        if not self.is_safe_square(new_position) and new_position < self.HOME_COLUMN_START:
            pieces_at_target = self.get_pieces_at(new_position)
            for other_piece in pieces_at_target[:]:  # Copy list to avoid modification during iteration
                if other_piece.color != piece.color:
                    captured.append(other_piece)
                    self.remove_piece(other_piece)
                    other_piece.send_to_base()

        # Update piece position
        piece.move_to(new_position)

        # Add piece to new position (unless finished)
        if new_position == self.FINAL_POSITION:
            piece.mark_finished()
        else:
            self.add_piece(piece, new_position)

        return captured

    def get_board_state(self):
        """
        Get a string representation of the current board state.

        Returns:
            str: Board state description
        """
        state_lines = ["Board State:"]
        for position in sorted(self.positions.keys()):
            pieces = self.positions[position]
            piece_strs = [f"{p.color}[{p.piece_id}]" for p in pieces]
            safe_marker = " (SAFE)" if self.is_safe_square(position) else ""
            state_lines.append(f"  Position {position}{safe_marker}: {', '.join(piece_strs)}")
        return "\n".join(state_lines)

    def __repr__(self):
        """String representation of the board."""
        return f"Board(pieces_on_board={sum(len(pieces) for pieces in self.positions.values())})"
