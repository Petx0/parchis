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
            set: Set of positions where blockades exist (a fresh copy of
            Board's cached set -- docs/V2_PLAN.md C6(b)).
        """
        return set(self.board.blockades)

    def _occupancy_count_for_move(self, position, color):
        """
        How many pieces already at `position` count toward
        Board.MAX_PIECES_PER_SQUARE for a piece of `color` moving there.

        Main-track squares (< HOME_COLUMN_START) are one shared physical
        square regardless of color, so every piece there counts -- this
        matches get_legal_moves' existing behavior for the main track,
        left unchanged.

        Home-column squares (>= HOME_COLUMN_START) are privately numbered
        per color: positions 69-75 mean a physically different square for
        each color (see docs/AZ_DESIGN.md's "home-column occupancy" fix
        write-up), so a piece of a DIFFERENT color sitting at the same
        number must not count against `color`'s own 2-piece cap there --
        only same-color pieces do. Mirrors get_blockades' own
        `pieces_at_pos[0].color == pieces_at_pos[1].color` color check,
        applied here to occupancy counting instead of blockade detection.
        """
        pieces = self.board.get_pieces_at(position)
        if position >= Board.HOME_COLUMN_START:
            pieces = [p for p in pieces if p.color == color]
        return len(pieces)

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
        if start_pos >= Board.HOME_COLUMN_START:
            # Not a case get_legal_moves produces (home-column pieces skip
            # the blockade check); kept on the path-based answer verbatim.
            path = self.compute_path(player, start_pos, num_squares)
            return any(pos in blockades for pos in path)
        # Closed form of the compute_path scan (C6(c)): the path's main-track
        # part is steps 1..min(num_squares, steps-to-home-entry) from
        # start_pos (the square after the entry point is the home column,
        # where blockades never exist), and the k-th step lands on the
        # square that is k ahead of start_pos modulo the track.
        on_track = (player.home_entry_point - start_pos) % Board.MAIN_TRACK_SIZE
        if num_squares < on_track:
            on_track = num_squares
        for blockade in blockades:
            if 0 < (blockade - start_pos) % Board.MAIN_TRACK_SIZE <= on_track:
                return True
        return False

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
        # Same moves, same order, same tuples as the original enumeration
        # (docs/V2_PLAN.md C6(c): verified byte-identical on 100k random
        # states by tests/test_rules_fast.py); rewritten to touch each
        # piece once, read Board's cached blockade set instead of rescanning
        # the safe squares, and use the closed-form path helpers above.
        board = self.board
        positions = board.positions
        color = player.color
        pieces = player.pieces
        has_pieces_in_base = any(p.in_base for p in pieces)

        # Special rule: If rolled a 6 and player has no pieces in base, move 7 squares
        effective_roll = dice_roll
        if dice_roll == BONUS_TURN_ROLL and not has_pieces_in_base:
            effective_roll = ALL_OUT_BONUS_ROLL

        # Option 1: If rolled a 5, can enter a new piece from base
        if dice_roll == ENTRY_ROLL and has_pieces_in_base:
            # Check what's at the starting square
            pieces_at_start = positions.get(player.starting_position, ())
            n_own = 0
            for p in pieces_at_start:
                if p.color == color:
                    n_own += 1
            n_opponent = len(pieces_at_start) - n_own

            can_enter = False
            # Rule 1: Starting square is empty
            if len(pieces_at_start) == 0:
                can_enter = True
            # Rule 2: One own piece at starting square
            elif n_own == 1 and n_opponent == 0:
                can_enter = True
            # Rule 3: One opponent piece at starting square
            elif n_own == 0 and n_opponent == 1:
                can_enter = True
            # Rule 4: Two own pieces - CANNOT enter
            elif n_own == Board.MAX_PIECES_PER_SQUARE:
                can_enter = False
            # Rule 5: One own, one opponent - CAN enter (will capture opponent)
            elif n_own == 1 and n_opponent == 1:
                can_enter = True
            # Rule 6: Two opponents - CAN enter (will capture most recent)
            elif n_opponent == Board.MAX_PIECES_PER_SQUARE:
                can_enter = True

            if can_enter:
                # Mandatory entry: if entry is possible, ONLY allow entry moves
                return [(piece, player.starting_position, 'enter') for piece in pieces if piece.in_base]

        # All blockades on the board (they block all players) -- Board's
        # live cached set; never mutated here.
        all_blockades = board.blockades
        legal_moves = []

        # Option 2: Move pieces already on board
        # (Only reached if not rolling 5 with legal entry, or no pieces in base)
        for piece in pieces:
            if piece.in_base or piece.finished:
                continue
            current_pos = piece.position

            # Check if piece is in home column (69-76)
            if current_pos >= Board.HOME_COLUMN_START:
                # Piece is in home column (no blockades apply)
                new_position = current_pos + effective_roll

                if new_position < Board.FINAL_POSITION:
                    # Destination room: max 2 pieces of THIS color (home-
                    # column squares are private per color -- see
                    # _occupancy_count_for_move)
                    n_same = 0
                    for p in positions.get(new_position, ()):
                        if p.color == color:
                            n_same += 1
                    if n_same < Board.MAX_PIECES_PER_SQUARE:
                        # Normal move within home column
                        legal_moves.append((piece, new_position, 'move'))
                elif new_position == Board.FINAL_POSITION:
                    # Exact landing on final position
                    legal_moves.append((piece, new_position, 'finish'))
                # else: would go beyond final position - not legal
            else:
                # Piece is on main track (1-68)
                new_position = self.calculate_new_position(player, current_pos, effective_roll)

                if new_position is not None:
                    # Destination room (max 2 pieces -- color-blind on the
                    # main track, same-color only if this move enters the
                    # home column; see _occupancy_count_for_move)
                    at_dest = positions.get(new_position, ())
                    if new_position >= Board.HOME_COLUMN_START:
                        n_at_dest = 0
                        for p in at_dest:
                            if p.color == color:
                                n_at_dest += 1
                    else:
                        n_at_dest = len(at_dest)
                    if n_at_dest < Board.MAX_PIECES_PER_SQUARE:
                        # Check if path crosses any blockade
                        if not self.path_crosses_blockade(player, current_pos, effective_roll, all_blockades):
                            legal_moves.append((piece, new_position, 'move'))

        # Special rule: If rolled a 6 and have blockades, must open one if possible
        if dice_roll == BONUS_TURN_ROLL and all_blockades:
            # Only moves whose piece is part of a blockade
            blockade_opening_moves = [move for move in legal_moves if move[0].position in all_blockades]
            # If there are moves that open blockades, only allow those
            if blockade_opening_moves:
                legal_moves = blockade_opening_moves

        return legal_moves

    def capture_and_finish_summary(self, player, dice_roll):
        """
        What get_legal_moves(player, dice_roll) would capture and whether it
        could finish, without materialising the move list: returns
        (captured, has_finish) where `captured` is the set of every piece
        (any colour) that some legal move would capture -- exactly
        {p for m in get_legal_moves(...) for p in would_capture(m)} -- and
        `has_finish` is whether any legal move has move_type 'finish'.

        The encoder's capture-threat feature asks this question 6-8 times
        per opponent per encode (docs/V2_PLAN.md C6(c)); the legality logic
        below mirrors get_legal_moves step for step (verified identical on
        100k random states by tests/test_rules_fast.py), with would_capture
        inlined.
        """
        board = self.board
        positions = board.positions
        color = player.color
        pieces = player.pieces
        has_pieces_in_base = any(p.in_base for p in pieces)

        effective_roll = dice_roll
        if dice_roll == BONUS_TURN_ROLL and not has_pieces_in_base:
            effective_roll = ALL_OUT_BONUS_ROLL

        if dice_roll == ENTRY_ROLL and has_pieces_in_base:
            pieces_at_start = positions.get(player.starting_position, ())
            own_pieces = [p for p in pieces_at_start if p.color == color]
            opponent_pieces = [p for p in pieces_at_start if p.color != color]
            n_own, n_opponent = len(own_pieces), len(opponent_pieces)
            can_enter = False
            if len(pieces_at_start) == 0:
                can_enter = True
            elif n_own == 1 and n_opponent == 0:
                can_enter = True
            elif n_own == 0 and n_opponent == 1:
                can_enter = True
            elif n_own == Board.MAX_PIECES_PER_SQUARE:
                can_enter = False
            elif n_own == 1 and n_opponent == 1:
                can_enter = True
            elif n_opponent == Board.MAX_PIECES_PER_SQUARE:
                can_enter = True
            if can_enter:
                # Entry moves only; would_capture('enter') depends on the
                # square alone, so every entering piece captures the same.
                if n_own == 1 and n_opponent == 1:
                    return {opponent_pieces[0]}, False
                if n_opponent == Board.MAX_PIECES_PER_SQUARE:
                    return {max(opponent_pieces, key=lambda p: p.move_order)}, False
                return set(), False

        all_blockades = board.blockades
        # (opens_blockade, captured_list_or_None, is_finish) per legal move;
        # the per-move record is only needed for the six-with-blockades
        # filter, so it is skipped otherwise.
        filter_on_six = dice_roll == BONUS_TURN_ROLL and bool(all_blockades)
        records = []
        captured = set()
        has_finish = False
        for piece in pieces:
            if piece.in_base or piece.finished:
                continue
            current_pos = piece.position
            if current_pos >= Board.HOME_COLUMN_START:
                new_position = current_pos + effective_roll
                if new_position < Board.FINAL_POSITION:
                    n_same = 0
                    for p in positions.get(new_position, ()):
                        if p.color == color:
                            n_same += 1
                    if n_same < Board.MAX_PIECES_PER_SQUARE:
                        if filter_on_six:
                            records.append((current_pos in all_blockades, None, False))
                elif new_position == Board.FINAL_POSITION:
                    if filter_on_six:
                        records.append((current_pos in all_blockades, None, True))
                    else:
                        has_finish = True
            else:
                new_position = self.calculate_new_position(player, current_pos, effective_roll)
                if new_position is None:
                    continue
                at_dest = positions.get(new_position, ())
                if new_position >= Board.HOME_COLUMN_START:
                    n_at_dest = 0
                    for p in at_dest:
                        if p.color == color:
                            n_at_dest += 1
                else:
                    n_at_dest = len(at_dest)
                if n_at_dest >= Board.MAX_PIECES_PER_SQUARE:
                    continue
                if self.path_crosses_blockade(player, current_pos, effective_roll, all_blockades):
                    continue
                # would_capture for 'move': non-safe main-track destinations
                # only. A main-track piece reaching FINAL_POSITION is typed
                # 'move' by get_legal_moves (only home-column pieces get
                # 'finish'), so it never counts toward has_finish.
                if new_position >= Board.HOME_COLUMN_START or new_position in Board.SAFE_SQUARES:
                    caps = None
                else:
                    caps = [p for p in at_dest if p.color != color]
                if filter_on_six:
                    records.append((current_pos in all_blockades, caps, False))
                elif caps:
                    captured.update(caps)

        if filter_on_six:
            opening = [r for r in records if r[0]]
            for _opens, caps, is_finish in (opening or records):
                if caps:
                    captured.update(caps)
                if is_finish:
                    has_finish = True
        return captured, has_finish

    def would_capture(self, move):
        """
        Non-mutating capture check for a single legal move, as returned by
        get_legal_moves. Mirrors Board.enter_piece / Board.move_piece's
        capture conditions exactly without mutating board state, so it's
        safe to call speculatively over many candidate moves (e.g. when
        scoring capture threats/opportunities for the observation space).

        Deliberately does NOT consult is_safe_square for 'enter' moves --
        a starting square is generally safe, but entry captures are
        occupancy-based (docs/RULES.md "Entering the Board" rules 5-6), not
        safety-based, so a piece can be captured via an opponent's entry
        even while sitting on a "safe" square.

        Args:
            move: tuple (piece, new_position, move_type) as returned by
                get_legal_moves.

        Returns:
            list: pieces that would be captured if this move were executed
            right now against the current board state (usually 0 or 1;
            'enter' captures at most 1, mirroring enter_piece's own
            most-recent-of-two rule for the two-opponents case).
        """
        piece, new_position, move_type = move

        if move_type == 'enter':
            pieces_at_start = self.board.get_pieces_at(new_position)
            own = [p for p in pieces_at_start if p.color == piece.color]
            opponents = [p for p in pieces_at_start if p.color != piece.color]
            if len(own) == 1 and len(opponents) == 1:
                return [opponents[0]]
            if len(opponents) == Board.MAX_PIECES_PER_SQUARE:
                return [max(opponents, key=lambda p: p.move_order)]
            return []

        # 'move' or 'finish': captures only on a non-safe main-track
        # square, never in the home column (mirrors Board.move_piece's own
        # guard exactly).
        if self.board.is_safe_square(new_position) or new_position >= Board.HOME_COLUMN_START:
            return []
        return [p for p in self.board.get_pieces_at(new_position) if p.color != piece.color]

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
        if current_pos >= Board.HOME_COLUMN_START:
            # Not a case get_legal_moves produces; path-based answer verbatim.
            new_position = self.compute_path(player, current_pos, dice_roll)[-1]
        else:
            # Closed form of compute_path(...)[-1] (C6(c)): `to_entry` main-
            # track steps reach the home entry point (0 if already on it);
            # the next step is the first home-column square.
            to_entry = (player.home_entry_point - current_pos) % Board.MAIN_TRACK_SIZE
            if dice_roll <= to_entry:
                new_position = (current_pos - 1 + dice_roll) % Board.MAIN_TRACK_SIZE + 1
            else:
                new_position = Board.HOME_COLUMN_START + dice_roll - to_entry - 1

        # Check if we would overshoot the finish position
        if new_position > Board.FINAL_POSITION:
            return None

        return new_position
