"""
Stateless move-legality rules for Parchís.

RuleEngine answers "what moves are legal" and "where does a move land"
purely from board state -- it never mutates the board and never tracks turn
state (dice rolls in progress, bonus chains, three-sixes counters). That
belongs to Game (parchis.game.game), which owns turn orchestration and
delegates every legality question to a RuleEngine instance.
"""

from parchis.game.board import Board
from parchis.game.constants import ENTRY_ROLL, BONUS_TURN_ROLL, ALL_OUT_BONUS_ROLL


class RuleEngine:
    """Computes legal moves, paths, and blockades from board state."""

    def __init__(self, board):
        self.board = board

    def get_blockades(self):
        """
        Get all blockade positions on the board.
        A blockade is formed when 2 pieces of the SAME color are at a safe square.
        Blockades block ALL players regardless of color (including the player who created it).

        Returns:
            set: Set of positions where blockades exist
        """
        blockades = set()
        for position in Board.SAFE_SQUARES:
            pieces_at_pos = self.board.get_pieces_at(position)
            if len(pieces_at_pos) == Board.MAX_PIECES_PER_SQUARE:
                # Check if both pieces are the same color
                if pieces_at_pos[0].color == pieces_at_pos[1].color:
                    blockades.add(position)
        return blockades

    def compute_path(self, player, start_pos, num_squares):
        """
        Compute the sequence of squares a piece passes through moving
        num_squares forward from start_pos, handling main-track wrap-around
        and home-column entry. This is the single source of truth for both
        landing-square legality and blockade-crossing checks.

        Args:
            player: The player (determines home_entry_point)
            start_pos: Current position on the main track (1-68)
            num_squares: Number of squares to advance

        Returns:
            list[int]: One entry per square advanced (length == num_squares),
                       in order. Values are main-track positions (1-68) or
                       home-column positions (Board.HOME_COLUMN_START and
                       above). Does not include start_pos itself. The path
                       may extend past Board.FINAL_POSITION; callers must
                       check for overshoot.
        """
        path = []
        pos = start_pos
        in_home = False
        for _ in range(num_squares):
            if in_home:
                pos += 1
            elif pos == player.home_entry_point:
                # Already sitting exactly on the entry point: the next
                # square forward is the first home column square.
                pos = Board.HOME_COLUMN_START
                in_home = True
            else:
                pos += 1
                if pos > Board.MAIN_TRACK_SIZE:
                    pos = 1
            path.append(pos)
        return path

    def path_crosses_blockade(self, player, start_pos, num_squares, blockades):
        """
        Check if moving num_squares forward from start_pos crosses any
        blockade (blockades only exist on main-track safe squares, so
        home-column squares in the path never match).

        Args:
            player: The player (determines home_entry_point)
            start_pos: Starting position (1-68)
            num_squares: Number of squares being advanced
            blockades: Set of blockade positions

        Returns:
            bool: True if path crosses a blockade
        """
        if not blockades:
            return False
        path = self.compute_path(player, start_pos, num_squares)
        return any(pos in blockades for pos in path)

    def get_legal_moves(self, player, dice_roll):
        """
        Get all legal moves for a player given a dice roll.

        Args:
            player: The player
            dice_roll: The dice roll value (1-6)

        Returns:
            list: List of tuples (piece, new_position, move_type)
                  move_type can be 'enter', 'move', or 'finish'
        """
        legal_moves = []

        # Special rule: If rolled a 6 and player has no pieces in base, move 7 squares
        effective_roll = dice_roll
        if dice_roll == BONUS_TURN_ROLL and len(player.get_pieces_in_base()) == 0:
            effective_roll = ALL_OUT_BONUS_ROLL

        # Option 1: If rolled a 5, can enter a new piece from base
        if dice_roll == ENTRY_ROLL:
            pieces_in_base = player.get_pieces_in_base()
            if pieces_in_base:
                # Check what's at the starting square
                pieces_at_start = self.board.get_pieces_at(player.starting_position)
                own_pieces = [p for p in pieces_at_start if p.color == player.color]
                opponent_pieces = [p for p in pieces_at_start if p.color != player.color]

                can_enter = False

                # Rule 1: Starting square is empty
                if len(pieces_at_start) == 0:
                    can_enter = True

                # Rule 2: One own piece at starting square
                elif len(own_pieces) == 1 and len(opponent_pieces) == 0:
                    can_enter = True

                # Rule 3: One opponent piece at starting square
                elif len(own_pieces) == 0 and len(opponent_pieces) == 1:
                    can_enter = True

                # Rule 4: Two own pieces - CANNOT enter
                elif len(own_pieces) == Board.MAX_PIECES_PER_SQUARE:
                    can_enter = False

                # Rule 5: One own, one opponent - CAN enter (will capture opponent)
                elif len(own_pieces) == 1 and len(opponent_pieces) == 1:
                    can_enter = True

                # Rule 6: Two opponents - CAN enter (will capture most recent)
                elif len(opponent_pieces) == Board.MAX_PIECES_PER_SQUARE:
                    can_enter = True

                if can_enter:
                    # Can enter any piece from base
                    for piece in pieces_in_base:
                        legal_moves.append((piece, player.starting_position, 'enter'))
                    # Mandatory entry: if entry is possible, ONLY allow entry moves
                    return legal_moves

        # Get all blockades on the board (they block all players)
        all_blockades = self.get_blockades()

        # Option 2: Move pieces already on board
        # (Only reached if not rolling 5 with legal entry, or no pieces in base)
        pieces_on_board = player.get_pieces_on_board()
        for piece in pieces_on_board:
            current_pos = piece.position

            # Check if piece is in home column (69-76)
            if current_pos >= Board.HOME_COLUMN_START:
                # Piece is in home column (no blockades apply)
                new_position = current_pos + effective_roll

                if new_position < Board.FINAL_POSITION:
                    # Check if destination has room (max 2 pieces)
                    if len(self.board.get_pieces_at(new_position)) < Board.MAX_PIECES_PER_SQUARE:
                        # Normal move within home column
                        legal_moves.append((piece, new_position, 'move'))
                elif new_position == Board.FINAL_POSITION:
                    # Exact landing on final position
                    legal_moves.append((piece, new_position, 'finish'))
                # else: would go beyond final position - not legal
            else:
                # Piece is on main track (1-68)
                # Calculate new position with wrapping
                new_position = self.calculate_new_position(player, current_pos, effective_roll)

                if new_position is not None:
                    # Check if destination has room (max 2 pieces)
                    pieces_at_dest = self.board.get_pieces_at(new_position)
                    if len(pieces_at_dest) < Board.MAX_PIECES_PER_SQUARE:
                        # Check if path crosses any blockade
                        if not self.path_crosses_blockade(player, current_pos, effective_roll, all_blockades):
                            legal_moves.append((piece, new_position, 'move'))

        # Special rule: If rolled a 6 and have blockades, must open one if possible
        if dice_roll == BONUS_TURN_ROLL and all_blockades:
            # Filter to only moves that open a blockade
            blockade_opening_moves = []
            for move in legal_moves:
                piece, new_pos, move_type = move
                # Check if this piece is part of a blockade
                if piece.position in all_blockades:
                    blockade_opening_moves.append(move)

            # If there are moves that open blockades, only allow those
            if blockade_opening_moves:
                legal_moves = blockade_opening_moves

        return legal_moves

    def calculate_new_position(self, player, current_pos, dice_roll):
        """
        Calculate the new position for a piece on the main track,
        handling wrapping and home entry.

        Args:
            player: The player
            current_pos: Current position (1-68)
            dice_roll: The dice roll value

        Returns:
            int or None: New position, or None if move is invalid
        """
        new_position = self.compute_path(player, current_pos, dice_roll)[-1]

        # Check if we would overshoot the finish position
        if new_position > Board.FINAL_POSITION:
            return None

        return new_position
