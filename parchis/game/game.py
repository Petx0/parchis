"""
Game engine module for Parchís.

Game owns turn/state orchestration (dice rolls, bonus chains, the
three-sixes rule, win detection) and delegates move-legality questions to a
RuleEngine (parchis.game.rules) and move execution/capture logic to Board
(parchis.game.board). Human-readable rendering lives in
parchis.game.formatting, kept separate so the engine itself has no
presentation concerns.

Game keeps thin wrapper methods for the rule-engine queries
(get_legal_moves, get_blockades, compute_path, path_crosses_blockade,
_calculate_new_position) so existing callers (tests, the RL environment)
don't need to reach into game.rules directly.
"""

from parchis.game.board import Board
from parchis.game.player import Player
from parchis.game.dice import Dice
from parchis.game.rules import RuleEngine
from parchis.game.records import MoveInfo, RollEntry, TurnInfo
from parchis.game.constants import (
    BONUS_TURN_ROLL,
    THREE_SIXES_LIMIT,
    CAPTURE_BONUS_SQUARES,
    FINISH_BONUS_SQUARES,
)
from parchis.game import formatting


class Game:
    """Main game engine that orchestrates the Parchís game."""

    def __init__(self, num_players=4, logger=None):
        """
        Initialize a new game.

        Args:
            num_players: Number of players (2-4)
            logger: Optional GameLogger instance for recording the game
        """
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.num_players = num_players
        self.board = Board()
        self.dice = Dice()
        self.logger = logger
        self.rules = RuleEngine(self.board)

        # Create players with appropriate colors
        # For 2-player games, use opposite colors (Red vs Yellow or Blue vs Green)
        if num_players == 2:
            # Randomly choose between the two opposite color pairs
            import random
            if random.choice([True, False]):
                colors = ["RED", "YELLOW"]  # Opposite on board
            else:
                colors = ["BLUE", "GREEN"]  # Opposite on board
        else:
            colors = Player.COLORS[:num_players]

        self.players = [Player(i, color) for i, color in enumerate(colors)]

        # Initialize board with starting pieces
        for player in self.players:
            for piece in player.pieces:
                if not piece.in_base:
                    self.board.add_piece(piece, piece.position)

        # Determine starting player through initial dice rolls
        self.starting_player, self.initial_dice_rolls = self.determine_starting_player()

        # Rotate players list so starting player is first
        starting_idx = self.players.index(self.starting_player)
        self.players = self.players[starting_idx:] + self.players[:starting_idx]

        self.current_player_idx = 0
        self.turn_number = 0
        self.game_over = False
        self.winner = None

    def determine_starting_player(self):
        """
        Determine which player starts through initial dice rolls.
        All players roll. Highest roll wins. If tied, those players roll again.

        Returns:
            tuple: (winning_player, list of roll rounds)
                   roll rounds is a list of dicts with format:
                   {'player': Player, 'roll': int}
        """
        all_roll_rounds = []
        candidates = self.players.copy()

        while len(candidates) > 1:
            # Roll for all candidates
            round_rolls = []
            for player in candidates:
                roll = self.dice.roll()
                round_rolls.append({'player': player, 'roll': roll})

            all_roll_rounds.append(round_rolls)

            # Find highest roll
            max_roll = max(r['roll'] for r in round_rolls)

            # Get players who tied with highest roll
            candidates = [r['player'] for r in round_rolls if r['roll'] == max_roll]

        return candidates[0], all_roll_rounds

    def get_current_player(self):
        """Get the current player."""
        return self.players[self.current_player_idx]

    def next_player(self):
        """Move to the next player's turn."""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    # --- Rule-engine delegation ---
    # These wrap self.rules so existing callers can keep calling them on
    # Game directly; the actual legality logic lives in RuleEngine.

    def get_blockades(self):
        """See RuleEngine.get_blockades."""
        return self.rules.get_blockades()

    def compute_path(self, player, start_pos, num_squares):
        """See RuleEngine.compute_path."""
        return self.rules.compute_path(player, start_pos, num_squares)

    def path_crosses_blockade(self, player, start_pos, num_squares, blockades):
        """See RuleEngine.path_crosses_blockade."""
        return self.rules.path_crosses_blockade(player, start_pos, num_squares, blockades)

    def get_legal_moves(self, player, dice_roll):
        """See RuleEngine.get_legal_moves."""
        return self.rules.get_legal_moves(player, dice_roll)

    def _calculate_new_position(self, player, current_pos, dice_roll):
        """See RuleEngine.calculate_new_position."""
        return self.rules.calculate_new_position(player, current_pos, dice_roll)

    # --- Move execution ---

    def execute_move(self, piece, new_position, move_type):
        """
        Execute a move.

        Args:
            piece: The piece to move
            new_position: The target position
            move_type: Type of move ('enter', 'move', 'finish')

        Returns:
            MoveInfo: Information about the move (captured pieces, etc.)
        """
        old_position = piece.position

        # Special handling for entry moves: the starting square is always
        # safe, so it follows its own capture rules (see Board.enter_piece).
        if move_type == 'enter':
            captured = self.board.enter_piece(piece, new_position)
        else:
            captured = self.board.move_piece(piece, new_position)

        return MoveInfo(
            piece=piece,
            old_position=old_position,
            new_position=new_position,
            move_type=move_type,
            captured=captured,
        )

    def handle_bonus_moves(self, player, initial_move_info, turn_info):
        """
        Handle bonus moves after a move is executed.
        Bonuses can chain: capture → 20 squares → capture → 20 squares → finish → 10 squares, etc.

        Args:
            player: The player who made the move
            initial_move_info: MoveInfo for the move that triggered the bonus
            turn_info: TurnInfo to append bonus moves to

        Returns:
            None (modifies turn_info in place)
        """
        move_info = initial_move_info

        while True:
            bonus_move_executed = False

            # Check for capture bonus
            if len(move_info.captured) > 0:
                bonus_info = self._execute_bonus_move(player, CAPTURE_BONUS_SQUARES, 'capture_bonus', turn_info)
                if bonus_info:
                    move_info = bonus_info
                    bonus_move_executed = True
                    continue

            # Check for finish bonus
            if move_info.new_position == Board.FINAL_POSITION:
                bonus_info = self._execute_bonus_move(player, FINISH_BONUS_SQUARES, 'finish_bonus', turn_info)
                if bonus_info:
                    move_info = bonus_info
                    bonus_move_executed = True
                    continue

            # No more bonuses to process
            if not bonus_move_executed:
                break

    def _execute_bonus_move(self, player, bonus_squares, bonus_type, turn_info):
        """
        Execute a single bonus move.

        Args:
            player: The player receiving the bonus
            bonus_squares: Number of squares for the bonus (10 or 20)
            bonus_type: Type of bonus ('capture_bonus' or 'finish_bonus')
            turn_info: TurnInfo to append to

        Returns:
            MoveInfo: Move info if bonus was executed, None otherwise
        """
        legal_moves = self.get_legal_moves(player, bonus_squares)

        if not legal_moves:
            turn_info.rolls.append(RollEntry(
                bonus_type=bonus_type,
                bonus_squares=bonus_squares,
                legal_moves_count=0,
            ))
            return None

        # Player must choose a bonus move
        chosen_move = player.choose_move(legal_moves)

        if not chosen_move:
            # Should not happen if legal_moves is not empty, but handle it
            turn_info.rolls.append(RollEntry(
                bonus_type=bonus_type,
                bonus_squares=bonus_squares,
                legal_moves_count=len(legal_moves),
            ))
            return None

        piece, new_position, move_type = chosen_move
        move_info = self.execute_move(piece, new_position, move_type)

        turn_info.rolls.append(RollEntry(
            bonus_type=bonus_type,
            bonus_squares=bonus_squares,
            legal_moves_count=len(legal_moves),
            chosen_move=chosen_move,
            move_info=move_info,
        ))

        return move_info

    @staticmethod
    def apply_three_sixes_penalty(board, second_six_piece, second_six_entered_home):
        """
        Apply the three-consecutive-sixes penalty: capture (send to base)
        the piece moved with the *second* of three consecutive 6s, unless
        it entered the home column on that exact move (docs/RULES.md
        Exception 2) or no piece was moved on the second six specifically
        (docs/RULES.md: "If no piece was moved with the second 6 ... no
        piece is captured"). The single implementation of this rule, used
        by both play_turn() and parchis.rl.env.ParchisEnv.

        Args:
            board: The Board to remove the piece from.
            second_six_piece: The piece moved on the second consecutive 6,
                or None if no piece was moved on that roll.
            second_six_entered_home: Whether that move transitioned the
                piece from the main track into its home column.

        Returns:
            tuple(bool, bool): (penalty_applied, protected_by_home_entry)
        """
        if second_six_piece is None:
            return False, False
        if second_six_entered_home:
            return False, True
        if second_six_piece.finished:
            return False, False
        board.remove_piece(second_six_piece)
        second_six_piece.send_to_base()
        return True, False

    def play_turn(self):
        """
        Play a single turn for the current player.
        Handles bonus turns for rolling 6 and the three-6s rule.

        Returns:
            TurnInfo: Information about the turn (includes all rolls if bonus turns occurred)
        """
        player = self.get_current_player()
        self.turn_number += 1

        turn_info = TurnInfo(turn_number=self.turn_number, player=player)

        consecutive_sixes = 0
        second_six_piece = None
        second_six_entered_home = False
        keep_rolling = True

        while keep_rolling and not self.game_over:
            # Roll dice
            dice_roll = self.dice.roll()

            # Track consecutive sixes
            if dice_roll == BONUS_TURN_ROLL:
                consecutive_sixes += 1
            else:
                consecutive_sixes = 0

            if consecutive_sixes == THREE_SIXES_LIMIT:
                # docs/RULES.md: "they do not get to use the third 6" - the
                # third six is never offered as a move at all. Resolve the
                # penalty immediately, against whichever piece was moved on
                # the second six (second_six_piece, scoped precisely to that
                # roll below - not just "whatever moved most recently",
                # which was the prior behavior and could wrongly penalize a
                # piece moved on this third roll, or a stale piece from an
                # earlier unrelated roll if the second six had no legal move).
                penalty_applied, protected = Game.apply_three_sixes_penalty(
                    self.board, second_six_piece, second_six_entered_home
                )
                turn_info.three_sixes_penalty = penalty_applied
                turn_info.home_entry_protection = protected
                if penalty_applied:
                    turn_info.penalty_piece = second_six_piece
                turn_info.rolls.append(RollEntry(dice_roll=dice_roll, legal_moves_count=0))
                keep_rolling = False
                continue

            # Get legal moves
            legal_moves = self.get_legal_moves(player, dice_roll)

            # Player chooses a move
            chosen_move = player.choose_move(legal_moves)

            roll_info = RollEntry(
                dice_roll=dice_roll,
                legal_moves_count=len(legal_moves),
                chosen_move=chosen_move,
            )

            # Execute move if one was chosen
            if chosen_move:
                piece, new_position, move_type = chosen_move
                old_position = piece.position  # Track old position before move
                move_info = self.execute_move(piece, new_position, move_type)
                roll_info.move_info = move_info

                # Scope second_six_piece precisely to "the roll where the
                # streak is exactly 2", not "whatever moved most recently".
                if consecutive_sixes == 2:
                    second_six_piece = piece
                    second_six_entered_home = (
                        old_position is not None
                        and old_position < Board.HOME_COLUMN_START
                        and new_position >= Board.HOME_COLUMN_START
                    )

                # Add the roll info before processing bonuses
                turn_info.rolls.append(roll_info)

                # Handle bonus moves (capture bonus, finish bonus, chaining)
                self.handle_bonus_moves(player, move_info, turn_info)

                # Check if player won (after all bonuses are processed).
                # Skip the sixes check entirely once the game is over - a
                # win landing exactly on a third consecutive six must not
                # spuriously penalize an unrelated piece.
                if player.has_won():
                    self.game_over = True
                    self.winner = player
                    turn_info.game_over = True
                    turn_info.winner = player
                    keep_rolling = False
                    continue
            else:
                # No move was chosen, just add the roll info. If this was
                # the second 6 specifically, there is no piece to penalize
                # later (docs/RULES.md: "no piece was moved with the second
                # 6 ... no piece is captured").
                if consecutive_sixes == 2:
                    second_six_piece = None
                    second_six_entered_home = False
                turn_info.rolls.append(roll_info)

            keep_rolling = (dice_roll == BONUS_TURN_ROLL)

        # Log the turn
        if self.logger:
            self.logger.log_turn(turn_info)

        # Move to next player
        self.next_player()

        return turn_info

    def play_game(self):
        """
        Play a complete game until there's a winner.

        Returns:
            Player: The winning player
        """
        if self.logger:
            self.logger.log_game_start(self.players)

        while not self.game_over:
            self.play_turn()

        if self.logger:
            self.logger.log_game_end(self.winner, self.turn_number)

        return self.winner

    def get_game_state_string(self):
        """See parchis.game.formatting.format_game_state."""
        return formatting.format_game_state(self)

    def format_turn_info(self, turn_info):
        """See parchis.game.formatting.format_turn_info."""
        return formatting.format_turn_info(turn_info)
