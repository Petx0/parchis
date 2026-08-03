"""
Gymnasium environment for Parchís game compatible with stable-baselines3.
"""

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parchis.game.game import Game
from parchis.game.board import Board
from parchis.game.constants import BONUS_TURN_ROLL, THREE_SIXES_LIMIT
from parchis.rl import rewards


def _random_opponent_policy(player, legal_moves):
    """Default opponent policy: uniform-random legal move."""
    return player.choose_move(legal_moves)


class ParchisEnv(gym.Env):
    """
    Custom Gymnasium Environment for Parchís game.

    Observation Space (dynamic size based on num_players, all normalized to [0, 1]):
        Total size = 79 * num_players + 36 (e.g., 194 for 2 players, 352 for 4 players)

        Board State (num_players × 76 positions):
            - One channel per player (current player first, then opponents by turn order)
            - Each channel has 76 positions (1-76: main track 1-68, home column 69-76)
            - Values: 0, 0.5, or 1.0 (representing 0, 1, or 2 pieces at that position)

        Global State (3 * num_players + 8 + 24 + 4 features):
            - Piece counts (2 × num_players values):
                * For each player: pieces_in_base / 4.0, pieces_finished / 4.0

            - Progress scores (num_players values):
                * For each player: average completion across 4 pieces, where a
                  piece contributes 0.0 in base, position/76.0 on board, or
                  1.0 finished (see _calculate_normalized_progress)
                * Higher = closer to winning, range [0.0, 1.0]

            - Dice roll (7 values, one-hot encoded):
                * is_dice_1, is_dice_2, is_dice_3, is_dice_4, is_dice_5
                * is_dice_6_normal (has pieces in base, effective_roll=6)
                * is_dice_6_no_base (no pieces in base, effective_roll=7)

            - Bonus indicator (1 value):
                * bonus_squares / 20.0 (0.0 for none, 0.5 for 10, 1.0 for 20)

            - Own-piece features (24 values = 4 pieces × 6 features, fixed
              slot by piece_id so they line up with the Discrete(4) action
              space - unlike the board-state block, this block is never
              reordered by turn):
                * in_base, finished, normalized_position, on_safe_square,
                  capture_threatened, capture_opportunity (see
                  _get_observation for exact definitions)

            - Blockade indicator (2 values):
                * own_blockades / 12.0, opponent_blockades / 12.0 (clipped
                  to 1.0; 12 = total number of safe squares, the hard cap
                  on simultaneous blockades)

            - Six-streak (1 value):
                * consecutive_sixes / THREE_SIXES_LIMIT

            - Bonus chain count (1 value):
                * bonus_chain_count / 4.0 (clipped to 1.0)

    Action Space:
        - Discrete(4): Choose which piece (0-3) to move
        - Invalid actions are masked using action_masks in info dict

    Bonus Moves:
        - Agent controls bonus moves triggered by captures (20 squares) or finishes (10 squares)
        - When a bonus is triggered, the same player continues with another step() call
        - Bonuses can chain: capture → 20 squares → capture → 20 squares → finish → 10 squares
        - The observation indicates when a bonus move is active via is_bonus_move flag

    Six-Again / Three-Sixes:
        - Rolling a 6 grants a reroll: the same player gets another step()
          call for a fresh roll instead of the turn ending (docs/RULES.md).
        - Three consecutive 6s in one turn: the third six is never used to
          move a piece: the piece moved on the *second* six is captured
          (sent back to base) and the turn ends immediately, unless that
          piece transitioned into its home column on that exact move (home
          entry protection) or no piece was moved on the second six.
        - This applies symmetrically to the agent (tracked via
          self.consecutive_sixes across step() calls) and to every
          opponent (auto-played synchronously within one step() call, see
          _play_full_opponent_turn), sharing the same rule implementation
          (Game.apply_three_sixes_penalty).

    Reward Structure (Turn-Cycle with Opponent Penalty):
        The reward is calculated at the end of each turn cycle (after all opponents
        have played), capturing both the agent's progress and opponent progress:

        - Progress Calculation (per piece):
            * In base: 0.0
            * On board: position / 76.0 (where 76 = 68 main track + 8 home column)
            * Finished: 1.0
            * Total progress = average across 4 pieces (range: 0.0 to 1.0)

        - Turn-Cycle Reward:
            * reward = my_Δ - α * combined_opponent_Δ
            * my_Δ = my progress at next turn - my progress at this turn
            * combined_opponent_Δ = opponents' progress changes combined via
              opponent_weighting ("mean" by default: average of all
              opponents' deltas; "leader": only the delta of whoever had
              the highest progress at the start of the cycle -- see
              parchis/rl/rewards.py, experiment-only, not the default for
              any training script)
            * α = opponent_weight (default 0.5)
            * A turn cycle can now span 1-3 dice rolls for whoever's turn
              it is (due to six-again rerolls) plus any bonus chain; reward
              is still computed exactly once per full cycle, only once the
              turn has genuinely ended.

        - Key Properties:
            * Rewards only at turn boundaries (0 during bonus moves and
              six-streak rerolls)
            * Captures consequences of positioning (getting captured hurts)
            * Encourages blocking/defensive play when α > 0
            * Can be negative if opponents advance more than you

        This approach provides strategic feedback that considers the full turn
        cycle, encouraging the agent to think about opponent responses. See
        parchis/rl/rewards.py for the reward formulas, named constants, and
        opponent-weighting schemes.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 1}

    VALID_REWARD_TYPES = rewards.VALID_REWARD_TYPES
    VALID_OPPONENT_WEIGHTING_SCHEMES = rewards.VALID_OPPONENT_WEIGHTING_SCHEMES

    # Own-piece feature block: 4 pieces (fixed slot by piece_id) x 6 features.
    PIECE_FEATURES_PER_PIECE = 6
    OWN_PIECE_FEATURES_SIZE = 4 * PIECE_FEATURES_PER_PIECE  # 24
    # 2 blockade (own/opponent) + 1 six-streak + 1 bonus-chain-count.
    STRATEGIC_FEATURES_SIZE = 4
    # Hard cap on simultaneous blockades: one per safe square.
    MAX_BLOCKADES = len(Board.SAFE_SQUARES)
    # Soft, saturating cap used to normalize bonus_chain_count.
    BONUS_CHAIN_NORMALIZATION_CAP = 4.0

    def __init__(self, num_players=4, render_mode=None, reward_type="progress_delta",
                 opponent_policy_fn=None, opponent_weight=rewards.DEFAULT_OPPONENT_WEIGHT,
                 opponent_weighting=rewards.DEFAULT_OPPONENT_WEIGHTING):
        """
        Initialize the Parchís environment.

        Args:
            num_players: Number of players (2-4)
            render_mode: Rendering mode (currently only 'human' supported)
            reward_type: Reward structure to use. One of:
                - "progress_delta": Turn-cycle delta (my_Δ - α * opp_Δ). Dense, small.
                - "win_loss": +1.0 on win, -1.0 on loss, 0.0 otherwise. Sparse.
                - "win_loss_shaped": Terminal +/-1.0, plus 0.1 * progress_delta mid-game.
            opponent_policy_fn: Callable(player, legal_moves) -> chosen_move used
                to decide every move for every seat except the one randomly
                assigned to the learning agent each reset() (see
                agent_player_idx), including chained bonus moves and
                six-again rerolls. Defaults to uniform-random. This is the
                single place opponent turns are resolved -- ParchisSelfPlayEnv
                plugs a model-backed policy in here instead of
                re-implementing its own opponent loop.
            opponent_weight: alpha, the weight applied to the combined
                opponent-progress term in the progress_delta/win_loss_shaped
                reward (default 0.5). Remains a plain mutable attribute
                after construction (env.opponent_weight = 0.3 still works).
            opponent_weighting: How multiple opponents' deltas are combined
                into that single term -- one of VALID_OPPONENT_WEIGHTING_SCHEMES
                (default "mean"). See parchis/rl/rewards.py.
        """
        super().__init__()

        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        if reward_type not in self.VALID_REWARD_TYPES:
            raise ValueError(
                f"Invalid reward_type '{reward_type}'. "
                f"Must be one of: {self.VALID_REWARD_TYPES}"
            )

        if opponent_weighting not in self.VALID_OPPONENT_WEIGHTING_SCHEMES:
            raise ValueError(
                f"Invalid opponent_weighting '{opponent_weighting}'. "
                f"Must be one of: {self.VALID_OPPONENT_WEIGHTING_SCHEMES}"
            )

        self.num_players = num_players
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.opponent_policy_fn = opponent_policy_fn or _random_opponent_policy

        # Action space: Choose which piece (0-3) to move
        self.action_space = spaces.Discrete(4)

        # Observation space (dynamic based on num_players):
        # - Board state: num_players × 76 positions
        # - Global state:
        #   * Piece counts: 2 × num_players (in_base, finished for each)
        #   * Progress scores: num_players
        #   * Dice roll: 7 (one-hot)
        #   * Bonus indicator: 1
        #   * Own-piece features: 24 (4 pieces × 6, fixed slot by piece_id)
        #   * Strategic features: 4 (blockade ×2, six-streak, bonus-chain-count)
        # Total: 76*N + 3*N + 8 + 24 + 4 = 79 * num_players + 36
        self.board_state_size = num_players * Board.FINAL_POSITION
        self.global_state_size = (
            3 * num_players + 8 + self.OWN_PIECE_FEATURES_SIZE + self.STRATEGIC_FEATURES_SIZE
        )
        obs_size = self.board_state_size + self.global_state_size
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,  # All values normalized to [0, 1]
            shape=(obs_size,),
            dtype=np.float32
        )

        # Initialize game state
        self.game = None
        self.agent_player_idx = 0  # reassigned each reset() to a random seat
        self.current_dice_roll = None
        self.last_reward = 0
        self.episode_length = 0
        self.max_episode_length = 1000  # Prevent infinite games

        # Bonus move tracking
        self.pending_bonus = None  # {'type': 'capture_bonus'/'finish_bonus', 'squares': 10/20}
        self.bonus_chain_count = 0  # Track bonus chain length for rewards

        # Six-again / three-sixes streak tracking for whoever's turn it
        # currently is (agent or opponent - see _play_full_opponent_turn).
        # Reset at the start of every turn by _start_new_turn_for_next_player.
        self.consecutive_sixes = 0
        self.second_six_piece = None
        self.second_six_entered_home = False

        # Per-turn-cycle KPI signals (docs/RL_DESIGN_REVIEW.md Phase 4):
        # captures and the three-sixes penalty were already computed
        # locally at their respective call sites but discarded before this
        # -- surfaced via _get_info() once the cycle completes, alongside
        # final_progress etc. Reset every turn cycle, same lifecycle as
        # turn_start_progress below.
        self.captures_by_agent_this_cycle = 0
        self.captures_against_agent_this_cycle = 0
        self.three_sixes_penalty_this_cycle = False

        # Turn-cycle reward configuration
        self.opponent_weight = opponent_weight  # α parameter: weight for opponent progress penalty
        self.opponent_weighting = opponent_weighting  # how multiple opponents' deltas combine (parchis/rl/rewards.py)
        self.turn_start_progress = {}  # {player_idx: progress} at start of turn cycle

        # Episode statistics for monitoring
        self.pieces_finished_count = 0
        self.pieces_out_of_base_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Additional information including action_masks
        """
        super().reset(seed=seed)

        if seed is not None:
            # Route the seed to the actual sources of game randomness:
            # Dice.roll() and Player.choose_move() both draw from Python's
            # global `random` module, not self.np_random, so seeding only
            # self.np_random (as super().reset() just did) left `seed`
            # cosmetic -- it controlled nothing an agent's training/eval
            # results depended on. self.np_random is still what drives
            # agent_player_idx below.
            random.seed(seed)

        # Create new game
        self.game = Game(num_players=self.num_players)
        self.episode_length = 0
        self.last_reward = 0
        self.pending_bonus = None
        self.bonus_chain_count = 0

        # Randomize which internal player index the learning agent occupies
        # this episode. Game.__init__ always rotates its player list so the
        # dice-determined starting player is index 0 (a real game rule), so
        # without this, an agent hardcoded to index 0 would always move
        # first -- a real, unaccounted-for advantage baked into every
        # trained model.
        self.agent_player_idx = int(self.np_random.integers(self.num_players))

        # Roll dice for the first turn (a fresh streak can never
        # immediately trip the three-sixes penalty).
        self._reset_six_streak_state()
        self._roll_and_check_sixes()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        # Initialize turn-cycle progress tracking for all players
        self.turn_start_progress = {
            i: self._calculate_normalized_progress(self.game.players[i])
            for i in range(self.num_players)
        }
        self.captures_by_agent_this_cycle = 0
        self.captures_against_agent_this_cycle = 0
        self.three_sixes_penalty_this_cycle = False
        self.pieces_finished_count = 0
        self.pieces_out_of_base_count = 1  # Start with 1 piece out

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Action to take (0-3, representing piece to move)

        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode is done (someone won)
            truncated: Whether episode was truncated (max length reached)
            info: Additional information including action_masks
        """
        if self.game is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        current_player = self.game.get_current_player()
        resolving_bonus = self.pending_bonus is not None

        # Determine if we're handling a bonus move or regular move
        if resolving_bonus:
            bonus_squares = self.pending_bonus['squares']
            legal_moves = self.game.get_legal_moves(current_player, bonus_squares)
        else:
            legal_moves = self.game.get_legal_moves(current_player, self.current_dice_roll)

        # Map action to legal move
        reward = 0.0
        chosen_move = None
        move_executed = False

        if len(legal_moves) > 0:
            # Filter legal moves by piece
            piece_moves = {}
            for move in legal_moves:
                piece, new_pos, move_type = move
                if piece.piece_id not in piece_moves:
                    piece_moves[piece.piece_id] = move

            # Check if action corresponds to a legal move
            if action in piece_moves:
                chosen_move = piece_moves[action]
                piece, new_position, move_type = chosen_move
                old_position = piece.position

                # Execute the move
                move_info = self.game.execute_move(piece, new_position, move_type)
                move_executed = True

                # Check if this move triggered a bonus
                if resolving_bonus:
                    # We just executed a bonus move, clear it
                    self.pending_bonus = None

                # Six-streak tracking only applies to the direct dice-roll
                # move, not to bonus-chain follow-up moves (docs/RULES.md:
                # "the piece that was moved with the second 6", i.e. the
                # roll itself, not any capture/finish bonus it triggers).
                # Scoped precisely to the roll where the streak is exactly
                # 2 -- see Game.apply_three_sixes_penalty.
                if not resolving_bonus and self.consecutive_sixes == 2:
                    self.second_six_piece = piece
                    self.second_six_entered_home = (
                        old_position is not None
                        and old_position < Board.HOME_COLUMN_START
                        and new_position >= Board.HOME_COLUMN_START
                    )

                # Check for new bonus triggers
                new_bonus = None
                if len(move_info.captured) > 0:
                    # Capture triggers 20-square bonus
                    new_bonus = {'type': 'capture_bonus', 'squares': 20}
                    self.bonus_chain_count += 1
                    self.captures_by_agent_this_cycle += len(move_info.captured)
                elif move_info.new_position == Board.FINAL_POSITION:
                    # Finish triggers 10-square bonus
                    new_bonus = {'type': 'finish_bonus', 'squares': 10}
                    self.bonus_chain_count += 1

                # Set pending bonus if triggered AND if there are legal moves with it
                if new_bonus is not None:
                    # Check if there are legal moves with this bonus
                    legal_moves_with_bonus = self.game.get_legal_moves(current_player, new_bonus['squares'])
                    if len(legal_moves_with_bonus) > 0:
                        self.pending_bonus = new_bonus
                        # Don't advance to next player, same player gets bonus move
                    else:
                        # Bonus triggered but no legal moves - skip bonus
                        self.pending_bonus = None
                        self.bonus_chain_count = 0
                        # Turn is over, continue to next player
                else:
                    # No bonus, reset chain count
                    self.bonus_chain_count = 0
            else:
                # Invalid action chosen (should not happen with action masking)
                reward = 0.0
                move_executed = False
        else:
            # No legal moves at all this roll. If this was the second six
            # specifically, there is no piece to penalize later.
            if not resolving_bonus and self.consecutive_sixes == 2:
                self.second_six_piece = None
                self.second_six_entered_home = False

        # Determine if this roll is resolved (no pending bonus and move was
        # executed or no legal moves). This does NOT by itself mean the
        # whole turn -- and therefore the reward -- is over: a six-again
        # reroll can still keep the same player's turn open, exactly like a
        # pending bonus chain does. turn_cycle_complete (below) is the
        # actual reward-computation gate.
        turn_over = self.pending_bonus is None and (move_executed or len(legal_moves) == 0)

        # Check if current player won
        terminated = False
        truncated = False
        turn_cycle_complete = False

        if current_player.has_won():
            terminated = True
            turn_cycle_complete = True
            # No additional terminal reward - progress already accounts for finishing
        elif turn_over:
            if self.current_dice_roll == BONUS_TURN_ROLL:
                # Six-again: reroll for the same player. This may itself
                # immediately trip the three-sixes penalty (in which case
                # the turn is over now) or produce a fresh actionable roll
                # (in which case control returns to whoever's turn this is
                # below, with no player change, and the cycle -- and reward
                # -- stays open, exactly like mid-bonus-chain).
                actionable = self._roll_and_check_sixes()
                if not actionable:
                    terminated, truncated = self._end_agent_turn_and_autoplay_opponents()
                    turn_cycle_complete = True
                # else: same player continues with a fresh self.current_dice_roll;
                # turn_cycle_complete stays False, reward stays 0.0 this step.
            else:
                terminated, truncated = self._end_agent_turn_and_autoplay_opponents()
                turn_cycle_complete = True

        # Calculate reward when the turn cycle is complete or game ends
        cycle_captures_by_agent = None
        cycle_captures_against_agent = None
        cycle_three_sixes_penalty = None
        if turn_cycle_complete or terminated:
            my_progress_now = self._calculate_normalized_progress(self.game.players[self.agent_player_idx])
            my_delta = my_progress_now - self.turn_start_progress[self.agent_player_idx]

            opponent_deltas = {}
            opponent_start_progress = {}
            for i in range(self.num_players):
                if i == self.agent_player_idx:
                    continue
                opp_progress_now = self._calculate_normalized_progress(self.game.players[i])
                opponent_deltas[i] = opp_progress_now - self.turn_start_progress[i]
                opponent_start_progress[i] = self.turn_start_progress[i]

            combined_opponent_delta = rewards.combine_opponent_deltas(
                opponent_deltas, opponent_start_progress, weighting=self.opponent_weighting
            )
            reward, progress_delta = rewards.compute_reward(
                self.reward_type, my_delta, combined_opponent_delta, self.opponent_weight,
                terminated, self.game.players[self.agent_player_idx].has_won(),
            )

            # Snapshot this cycle's KPI signals (docs/RL_DESIGN_REVIEW.md
            # Phase 4) before resetting them for the next cycle, same
            # lifecycle as turn_start_progress below.
            cycle_captures_by_agent = self.captures_by_agent_this_cycle
            cycle_captures_against_agent = self.captures_against_agent_this_cycle
            cycle_three_sixes_penalty = self.three_sixes_penalty_this_cycle

            # Reset turn start progress for next cycle
            self.turn_start_progress = {
                i: self._calculate_normalized_progress(self.game.players[i])
                for i in range(self.num_players)
            }
            self.captures_by_agent_this_cycle = 0
            self.captures_against_agent_this_cycle = 0
            self.three_sixes_penalty_this_cycle = False

        observation = self._get_observation()
        info = self._get_info()

        # Surface this cycle's KPI signals (docs/RL_DESIGN_REVIEW.md Phase
        # 4: capture rate, three-sixes-penalty rate), gated on the same
        # condition reward is computed under -- these are only meaningful
        # once a full turn cycle has resolved.
        if turn_cycle_complete or terminated:
            info['captures_by_agent'] = cycle_captures_by_agent
            info['captures_against_agent'] = cycle_captures_against_agent
            info['three_sixes_penalty'] = cycle_three_sixes_penalty

        # Add final progress and other metrics to info for monitoring
        if terminated or truncated:
            learning_player = self.game.players[self.agent_player_idx]
            info['final_progress'] = self._calculate_normalized_progress(learning_player)
            info['pieces_finished'] = len(learning_player.get_finished_pieces())
            info['pieces_out_of_base'] = len(learning_player.get_pieces_on_board()) + len(learning_player.get_finished_pieces())
            info['won'] = learning_player.has_won()

        return observation, reward, terminated, truncated, info

    def _reset_six_streak_state(self):
        """Reset the consecutive-6 streak state for a fresh turn."""
        self.consecutive_sixes = 0
        self.second_six_piece = None
        self.second_six_entered_home = False

    def _roll_and_check_sixes(self):
        """
        Roll one die for whoever self.game.current_player_idx currently is
        and update the consecutive-6 streak. If this produces the third
        consecutive 6, it is never used to move a piece (docs/RULES.md:
        "the player's turn ends immediately ... they do not get to use the
        third 6") -- apply the penalty to whoever was moved on the second
        six (self.second_six_piece / self.second_six_entered_home) and
        report that this roll is not actionable. Otherwise
        self.current_dice_roll is now a live decision (a regular move, or
        another reroll if it's a 6).

        Returns:
            bool: True if self.current_dice_roll is now actionable for the
                current player; False if the three-sixes penalty just fired
                and the turn is already over.
        """
        self.current_dice_roll = self.game.dice.roll()

        if self.current_dice_roll == BONUS_TURN_ROLL:
            self.consecutive_sixes += 1
        else:
            self.consecutive_sixes = 0

        if self.consecutive_sixes == THREE_SIXES_LIMIT:
            penalty_applied, _protected = Game.apply_three_sixes_penalty(
                self.game.board, self.second_six_piece, self.second_six_entered_home
            )
            if penalty_applied:
                self.three_sixes_penalty_this_cycle = True
            return False

        return True

    def _start_new_turn_for_next_player(self):
        """
        A turn has genuinely ended: advance to the next player, reset their
        six-streak state, and roll their first die. A freshly-reset streak
        can never immediately trip the three-sixes penalty (that requires 3
        prior rolls this turn), so the resulting roll is always actionable.
        """
        self.game.next_player()
        self._reset_six_streak_state()
        self._roll_and_check_sixes()

    def _end_agent_turn_and_autoplay_opponents(self):
        """
        The agent's turn (including any six-streak reroll chain) has
        genuinely ended. Advance to the next player, then auto-play every
        opponent's full turn (including their own six-streaks and bonus
        chains, via self.opponent_policy_fn) until control returns to the
        agent, the game ends, or the episode truncates.

        Returns:
            tuple(bool, bool): (terminated, truncated)
        """
        terminated = False

        self._start_new_turn_for_next_player()
        self.episode_length += 1
        truncated = self.episode_length >= self.max_episode_length

        while not truncated and self.game.current_player_idx != self.agent_player_idx and not terminated:
            opponent = self.game.get_current_player()
            if self._play_full_opponent_turn(opponent):
                terminated = True
                break
            self._start_new_turn_for_next_player()
            self.episode_length += 1
            if self.episode_length >= self.max_episode_length:
                truncated = True
                break

        return terminated, truncated

    def _play_full_opponent_turn(self, player):
        """
        Fully resolve one non-agent player's entire turn synchronously,
        including six-again rerolls, the three-sixes penalty, and bonus
        chains -- mirroring Game.play_turn() but routing every move
        through self.opponent_policy_fn instead of player.choose_move.

        Reuses the same self.consecutive_sixes/self.second_six_piece/
        self.second_six_entered_home instance state the agent's own turn
        uses (safe: turns are never interleaved, this fully resolves
        before control passes to anyone else) so _get_observation()'s
        six-streak feature stays accurate even if queried mid-opponent-turn
        (e.g. by ParchisSelfPlayEnv's model-backed opponent policy).

        The caller (_start_new_turn_for_next_player) has already rolled
        this player's first die and reset the streak before invoking this.

        Args:
            player: The opponent whose turn to resolve.

        Returns:
            bool: True if this player won during their turn.
        """
        while True:
            legal_moves = self.game.get_legal_moves(player, self.current_dice_roll)
            chosen_move = self.opponent_policy_fn(player, legal_moves)

            if chosen_move:
                piece, new_position, move_type = chosen_move
                old_position = piece.position
                move_info = self.game.execute_move(piece, new_position, move_type)

                if self.consecutive_sixes == 2:
                    self.second_six_piece = piece
                    self.second_six_entered_home = (
                        old_position is not None
                        and old_position < Board.HOME_COLUMN_START
                        and new_position >= Board.HOME_COLUMN_START
                    )

                # Handle bonus moves for non-learning players
                if len(move_info.captured) > 0:
                    # Capture bonus: 20 squares
                    self._record_opponent_capture(move_info.captured)
                    self._auto_play_bonus(player, 20)
                elif move_info.new_position == Board.FINAL_POSITION:
                    # Finish bonus: 10 squares
                    self._auto_play_bonus(player, 10)

                if player.has_won():
                    return True
            else:
                if self.consecutive_sixes == 2:
                    self.second_six_piece = None
                    self.second_six_entered_home = False

            if self.current_dice_roll != BONUS_TURN_ROLL:
                return False

            if not self._roll_and_check_sixes():
                return False  # third-six penalty fired; turn over

    def _record_opponent_capture(self, captured_pieces):
        """
        Tally the capture_rate KPI (docs/RL_DESIGN_REVIEW.md Phase 4) for
        pieces captured by an opponent's move during their auto-played
        turn. Only pieces of the agent's own color count toward
        captures_against_agent_this_cycle -- in a 3-4 player game an
        opponent's move can capture a *different* opponent's piece, which
        isn't a capture "against" the learning agent.
        """
        agent_color = self.game.players[self.agent_player_idx].color
        for piece in captured_pieces:
            if piece.color == agent_color:
                self.captures_against_agent_this_cycle += 1

    def _calculate_normalized_progress(self, player=None):
        """
        Calculate a player's normalized progress (0.0 to 1.0). See
        parchis.rl.rewards.calculate_normalized_progress for the formula.

        Args:
            player: Player to calculate progress for (if None, uses current player)

        Returns:
            float: Average progress across 4 pieces (range: 0.0 to 1.0)
        """
        if player is None:
            player = self.game.get_current_player()

        return rewards.calculate_normalized_progress(player)

    def _auto_play_bonus(self, player, bonus_squares):
        """
        Automatically play bonus moves for non-learning players, via
        self.opponent_policy_fn (same hook used for their regular moves).

        Args:
            player: The player receiving the bonus
            bonus_squares: Number of squares for the bonus (10 or 20)
        """
        legal_moves = self.game.get_legal_moves(player, bonus_squares)
        chosen_move = self.opponent_policy_fn(player, legal_moves)

        if chosen_move:
            piece, new_position, move_type = chosen_move
            move_info = self.game.execute_move(piece, new_position, move_type)

            # Recursively handle chained bonuses
            if len(move_info.captured) > 0:
                self._record_opponent_capture(move_info.captured)
                self._auto_play_bonus(player, 20)
            elif move_info.new_position == Board.FINAL_POSITION:
                self._auto_play_bonus(player, 10)

    def _get_observation(self):
        """
        Construct the observation array (dynamic size based on num_players).

        Returns:
            numpy array containing:
            - Board state: num_players × 76 positions
            - Global state: piece counts, progress scores, dice, bonus,
              own-piece features, blockade indicator, six-streak,
              bonus-chain count
        """
        obs = np.zeros(self.board_state_size + self.global_state_size, dtype=np.float32)

        # Get players ordered by turn (current player first)
        current_idx = self.game.current_player_idx
        ordered_players = (
            self.game.players[current_idx:] +
            self.game.players[:current_idx]
        )

        # ===== BOARD STATE: num_players × 76 positions =====
        for player_idx, player in enumerate(ordered_players):
            offset = player_idx * Board.FINAL_POSITION

            # Fill positions 1-76 with piece counts (normalized). Finished
            # pieces are excluded: Board never tracks them as occupants of
            # position 76 either (Board.move_piece calls piece.mark_finished()
            # instead of add_piece() when a piece reaches FINAL_POSITION), so
            # up to 4 finished pieces sharing position 76 would otherwise
            # accumulate past the declared [0, 1] bound at that index. Whether
            # a player has finished pieces is already captured separately by
            # the piece-counts block below (pieces_finished / 4.0).
            for piece in player.pieces:
                if not piece.in_base and not piece.finished and piece.position is not None:
                    pos = piece.position
                    if 1 <= pos <= Board.FINAL_POSITION:
                        # Normalize: 0 pieces = 0.0, 1 piece = 0.5, 2 pieces = 1.0
                        obs[offset + pos - 1] += 0.5

        # ===== GLOBAL STATE =====
        global_offset = self.board_state_size

        # --- Piece counts: 2 values per player ---
        for i, player in enumerate(ordered_players):
            obs[global_offset + i * 2] = len(player.get_pieces_in_base()) / 4.0
            obs[global_offset + i * 2 + 1] = len(player.get_finished_pieces()) / 4.0

        # --- Progress scores: 1 value per player ---
        progress_offset = global_offset + 2 * self.num_players
        for player_idx, player in enumerate(ordered_players):
            obs[progress_offset + player_idx] = self._calculate_normalized_progress(player)

        # --- Dice roll: 7 values (one-hot) ---
        dice_offset = progress_offset + self.num_players
        if self.current_dice_roll is not None:
            dice = self.current_dice_roll
            if dice < 6:
                # Dice 1-5: simple one-hot
                obs[dice_offset + dice - 1] = 1.0
            elif dice == 6:
                # Dice 6: check if pieces in base
                if len(ordered_players[0].get_pieces_in_base()) > 0:
                    obs[dice_offset + 5] = 1.0  # is_dice_6_normal
                else:
                    obs[dice_offset + 6] = 1.0  # is_dice_6_no_base (effective_roll=7)

        # --- Bonus indicator: 1 value ---
        bonus_offset = dice_offset + 7
        if self.pending_bonus is not None:
            obs[bonus_offset] = self.pending_bonus['squares'] / 20.0
        # else: already 0.0 from np.zeros

        # ===== OWN-PIECE FEATURES: 4 pieces × 6, fixed slot by piece_id =====
        # Indexed strictly by piece.piece_id (matching how _get_info()'s
        # action_masks[piece.piece_id] already works) -- never reordered by
        # turn order, unlike the board-state block above. This is what lets
        # the network distinguish the consequence of choosing action=0 vs
        # action=3 when multiple of the agent's own pieces have
        # simultaneously legal moves.
        own_piece_offset = bonus_offset + 1
        agent_player = self.game.players[self.agent_player_idx]
        for piece in agent_player.pieces:
            base = own_piece_offset + piece.piece_id * self.PIECE_FEATURES_PER_PIECE
            in_base = piece.in_base
            finished = piece.finished
            pos = piece.position

            obs[base + 0] = 1.0 if in_base else 0.0
            obs[base + 1] = 1.0 if finished else 0.0

            if finished:
                obs[base + 2] = 1.0
            elif in_base:
                obs[base + 2] = 0.0
            else:
                obs[base + 2] = pos / Board.FINAL_POSITION

            on_safe = (
                not in_base and not finished
                and (pos in Board.SAFE_SQUARES or pos >= Board.HOME_COLUMN_START)
            )
            obs[base + 3] = 1.0 if on_safe else 0.0

            # capture_threatened: any opponent piece 1-6 squares behind this
            # piece, on a square it could actually be captured from. A
            # piece already on a safe square (main-track safe square or
            # home column) can never be threatened.
            threatened = 0.0
            if not in_base and not finished and not on_safe:
                for d in range(1, 7):
                    behind = ((pos - d - 1) % Board.MAIN_TRACK_SIZE) + 1
                    if any(p.color != agent_player.color for p in self.game.board.get_pieces_at(behind)):
                        threatened = 1.0
                        break
            obs[base + 4] = threatened

            # capture_opportunity: any opponent piece 1-6 squares ahead of
            # this piece on a non-safe square. Own safety doesn't matter
            # here -- a piece on a safe square can still threaten others.
            opportunity = 0.0
            if not in_base and not finished and pos < Board.HOME_COLUMN_START:
                for d in range(1, 7):
                    ahead = ((pos + d - 1) % Board.MAIN_TRACK_SIZE) + 1
                    if ahead in Board.SAFE_SQUARES:
                        continue
                    if any(p.color != agent_player.color for p in self.game.board.get_pieces_at(ahead)):
                        opportunity = 1.0
                        break
            obs[base + 5] = opportunity

        # ===== BLOCKADE INDICATOR: 2 values =====
        # Both directions matter: blockades restrict *everyone's* movement
        # (docs/RULES.md), including the agent's own pieces being blocked
        # by an opponent's blockade.
        blockade_offset = own_piece_offset + self.OWN_PIECE_FEATURES_SIZE
        all_blockades = self.game.get_blockades()
        own_blockades = sum(
            1 for pos in all_blockades
            if self.game.board.get_pieces_at(pos)[0].color == agent_player.color
        )
        opponent_blockades = len(all_blockades) - own_blockades
        obs[blockade_offset] = min(own_blockades / self.MAX_BLOCKADES, 1.0)
        obs[blockade_offset + 1] = min(opponent_blockades / self.MAX_BLOCKADES, 1.0)

        # --- Six-streak: 1 value ---
        six_streak_offset = blockade_offset + 2
        obs[six_streak_offset] = self.consecutive_sixes / THREE_SIXES_LIMIT

        # --- Bonus chain count: 1 value ---
        bonus_chain_offset = six_streak_offset + 1
        obs[bonus_chain_offset] = min(self.bonus_chain_count / self.BONUS_CHAIN_NORMALIZATION_CAP, 1.0)

        return obs

    def _get_info(self):
        """
        Get additional information including action masks.

        Returns:
            dict with action_masks and other info
        """
        current_player = self.game.get_current_player()

        # Get legal moves based on whether we have a pending bonus
        if self.pending_bonus is not None:
            bonus_squares = self.pending_bonus['squares']
            legal_moves = self.game.get_legal_moves(current_player, bonus_squares)
            effective_roll = bonus_squares
        else:
            legal_moves = self.game.get_legal_moves(current_player, self.current_dice_roll)
            effective_roll = self.current_dice_roll

        # Create action mask (1 = valid action, 0 = invalid)
        action_masks = np.zeros(4, dtype=np.int8)

        for move in legal_moves:
            piece, new_pos, move_type = move
            action_masks[piece.piece_id] = 1

        # If no legal moves, allow any action (it will just skip the turn)
        if len(legal_moves) == 0:
            action_masks = np.ones(4, dtype=np.int8)

        info = {
            'action_masks': action_masks,
            'agent_player_idx': self.agent_player_idx,
            'legal_moves_count': len(legal_moves),
            'current_player': current_player.color,
            'dice_roll': self.current_dice_roll,
            'episode_length': self.episode_length,
            'is_bonus_move': self.pending_bonus is not None,
            'bonus_type': self.pending_bonus['type'] if self.pending_bonus else None,
            'bonus_squares': self.pending_bonus['squares'] if self.pending_bonus else None,
            'bonus_chain_count': self.bonus_chain_count,
            'consecutive_sixes': self.consecutive_sixes,
            'effective_roll': effective_roll
        }

        return info

    def render(self):
        """
        Render the environment.
        Currently not implemented - placeholder for future implementation.
        """
        if self.render_mode == 'human':
            # TODO: Implement rendering
            pass

    def close(self):
        """
        Clean up resources.
        """
        self.game = None
