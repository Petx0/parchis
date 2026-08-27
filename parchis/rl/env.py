"""
Gymnasium environment for Parchís game compatible with stable-baselines3.
"""

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from parchis.game.game import Game
from parchis.game.board import Board
from parchis.game.constants import (
    BONUS_TURN_ROLL, THREE_SIXES_LIMIT, CAPTURE_BONUS_SQUARES, FINISH_BONUS_SQUARES,
)
from parchis.rl import rewards


def _random_opponent_policy(player, legal_moves):
    """Default opponent policy: uniform-random legal move."""
    return player.choose_move(legal_moves)


class ParchisEnv(gym.Env):
    """
    Custom Gymnasium Environment for Parchís game.

    Observation Space (dynamic size based on num_players, all normalized to [0, 1]):
        Total size = 79 * num_players + 31 (e.g., 189 for 2 players, 347 for 4 players)

        Board State (num_players × 76 positions):
            - One channel per player (current player first, then opponents by turn order)
            - Each channel has 76 positions (1-76: main track 1-68, home column 69-76)
            - Values: 0, 0.5, or 1.0 (representing 0, 1, or 2 pieces at that position)

        Global State (3 * num_players + 7 + 2 + 21 + 1 features):
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

            - Bonus indicator (2 values, mutually exclusive binary flags):
                * has_finish_bonus, has_capture_bonus

            - Own-piece features (21 values = 4 pieces × 5 per-piece features,
              fixed slot by piece_id so they line up with the Discrete(4)
              action space - unlike the board-state block, this block is
              never reordered by turn - plus 1 shared value):
                * Per piece (stride 5): in_base, finished, normalized_position,
                  on_safe_square, capture_threat_score (roll-based [0,1] score:
                  fraction of the 6 dice faces that would let some opponent
                  capture this piece this turn, directly or via a bonus
                  chain - see _capture_threat_scores)
                * Shared (1 value, not per-piece): capture_opportunity (roll-
                  based [0,1] score: fraction of the 6 dice faces that would
                  let the agent capture something with any of its own pieces
                  this turn, single-roll only - see _capture_opportunity_score)

            - Six-streak (1 value):
                * consecutive_sixes / THREE_SIXES_LIMIT

        bonus_chain_count is still tracked on the instance and exposed via
        _get_info() (for KPI logging in parchis/training/common.py and
        parchis/evaluation/evaluate.py) but is no longer part of the
        observation array, and the blockade indicator (own/opponent
        blockade counts) has been removed entirely - see
        docs/observation_space_changes.md for the rationale.

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

    # Own-piece feature block: 4 pieces (fixed slot by piece_id) x 5
    # per-piece features, plus 1 shared capture_opportunity slot covering
    # all 4 pieces at once (not indexed by piece_id).
    PIECE_FEATURES_PER_PIECE = 5
    OWN_PIECE_FEATURES_SIZE = 4 * PIECE_FEATURES_PER_PIECE + 1  # 21
    # 2 mutually-exclusive bonus flags (has_finish_bonus/has_capture_bonus).
    BONUS_FEATURES_SIZE = 2
    # Six-streak only. Blockade indicator and bonus-chain count were cut
    # from the observation entirely (docs/observation_space_changes.md).
    STRATEGIC_FEATURES_SIZE = 1

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
        #   * Bonus indicator: 2 (has_finish_bonus, has_capture_bonus)
        #   * Own-piece features: 21 (4 pieces × 5, fixed slot by piece_id,
        #     + 1 shared capture_opportunity score)
        #   * Strategic features: 1 (six-streak)
        # Total: 76*N + 3*N + 7 + 2 + 21 + 1 = 79 * num_players + 31
        self.board_state_size = num_players * Board.FINAL_POSITION
        self.global_state_size = (
            3 * num_players + 7 + self.BONUS_FEATURES_SIZE
            + self.OWN_PIECE_FEATURES_SIZE + self.STRATEGIC_FEATURES_SIZE
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

    def _capture_threat_scores(self, agent_player):
        """
        Roll-based capture_threat_score per own piece. For every opponent
        and every face value 1-6, count a "hit" against a given own piece
        if rolling that value this turn would let the opponent capture it,
        either directly or via the bonus move (10/20 squares) that roll
        unlocks. Legality and capture outcomes come entirely from
        Game.get_legal_moves / Game.would_capture -- never hand-rolled
        distance math -- so every rule edge case (mandatory-5-entry, the
        6-with-0-base 7-move, the 6-with-blockade must-open restriction,
        entry captures on an opponent's "safe" starting square) is
        automatically correct.

        Known, deferred perf note: this (and _capture_opportunity_score)
        run O(num_players * 6) extra get_legal_moves calls, and
        ParchisSelfPlayEnv._choose_opponent_move now calls
        _get_observation() for every opponent decision (including bonus-
        chain moves), not just once per agent step -- so this is on a
        hotter path than when it was first written. Not fixed here
        (correctness-only pass); a real fix would cache/reuse legal-move
        queries across the per-roll loop instead of recomputing them.

        Known, accepted limitation: the bonus-chain check queries
        get_legal_moves(opponent, 10/20) against the current, unmutated
        board. This correctly finds a bonus-chain threat delivered by any
        OTHER already-on-board piece of that opponent. It does not find
        the rarer case where the SAME piece that captures/finishes
        continues, via its own bonus move, from its new post-move
        position -- that would require speculatively mutating and
        reverting board state for every (opponent, roll) pair, which is
        not done here.

        Args:
            agent_player: the learning agent's Player.

        Returns:
            dict[int, int]: piece_id -> raw hit count, summed across all
            opponents (0..6 per opponent). Not yet divided by 6 or
            clipped -- double threats from different opponents are not
            deduplicated ("double threat = double risk").
        """
        hit_counts = {piece.piece_id: 0 for piece in agent_player.pieces}

        for opponent in self.game.players:
            if opponent is agent_player:
                continue

            moves_by_roll = {v: self.game.get_legal_moves(opponent, v) for v in range(1, 7)}
            direct_targets_by_roll = {
                v: {p for m in moves for p in self.game.would_capture(m)}
                for v, moves in moves_by_roll.items()
            }
            capture_rolls = {v for v, targets in direct_targets_by_roll.items() if targets}
            finish_rolls = {
                v for v, moves in moves_by_roll.items()
                if any(move_type == 'finish' for (_p, _pos, move_type) in moves)
            }

            # Bonus-move lists don't depend on which face value triggered
            # them or which own piece is being scored -- compute each at
            # most once per opponent per _get_observation() call.
            bonus20_targets = set()
            if capture_rolls:
                bonus20_moves = self.game.get_legal_moves(opponent, CAPTURE_BONUS_SQUARES)
                bonus20_targets = {p for m in bonus20_moves for p in self.game.would_capture(m)}
            bonus10_targets = set()
            if finish_rolls:
                bonus10_moves = self.game.get_legal_moves(opponent, FINISH_BONUS_SQUARES)
                bonus10_targets = {p for m in bonus10_moves for p in self.game.would_capture(m)}

            for piece in agent_player.pieces:
                for v in range(1, 7):
                    hit = (
                        piece in direct_targets_by_roll[v]
                        or (v in capture_rolls and piece in bonus20_targets)
                        or (v in finish_rolls and piece in bonus10_targets)
                    )
                    if hit:
                        hit_counts[piece.piece_id] += 1

        return hit_counts

    def _capture_opportunity_score(self, agent_player):
        """
        Shared, single-roll-only capture_opportunity score covering all 4
        of the agent's own pieces at once. For each face value 1-6: does
        the agent have >=1 legal move this turn, via ANY of its 4 pieces,
        that is a capture? OR'd across pieces -- a face value counts once
        even if multiple own pieces could capture with it. Explicitly
        scoped to single-roll captures only -- unlike
        _capture_threat_scores, this does NOT extend through bonus chains
        (asymmetric on purpose).

        Args:
            agent_player: the learning agent's Player.

        Returns:
            float: (# of the 6 face values that produce >=1 capturing
            move) / 6.
        """
        hits = 0
        for v in range(1, 7):
            moves = self.game.get_legal_moves(agent_player, v)
            if any(self.game.would_capture(m) for m in moves):
                hits += 1
        return hits / 6.0

    def _get_observation(self, perspective_seat=None):
        """
        Construct the observation array (dynamic size based on num_players).

        Args:
            perspective_seat: index into self.game.players whose own-piece
                features (and capture_opportunity, which is derived from
                them) this observation should reflect. Defaults to
                self.agent_player_idx, preserving the observation the
                learning agent has always received via reset()/step().
                Pass the ACTING player's seat when building an observation
                for someone other than the learning agent (e.g. an
                opponent-model policy) -- see
                docs/AGENT_REBUILD_PLAN.md §1.3: without this, the own-piece
                block and capture_opportunity silently described the
                learning agent's pieces no matter whose decision the
                observation was built for, while the board-state/piece-
                count/progress/dice blocks below already correctly rotate
                by self.game.current_player_idx, independent of this
                parameter.

        Returns:
            numpy array containing:
            - Board state: num_players × 76 positions
            - Global state: piece counts, progress scores, dice, bonus,
              own-piece features, six-streak
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

        # --- Bonus indicator: 2 mutually exclusive binary flags ---
        bonus_offset = dice_offset + 7
        if self.pending_bonus is not None:
            if self.pending_bonus['type'] == 'finish_bonus':
                obs[bonus_offset] = 1.0      # has_finish_bonus
            elif self.pending_bonus['type'] == 'capture_bonus':
                obs[bonus_offset + 1] = 1.0  # has_capture_bonus
        # else: both already 0.0 from np.zeros

        # ===== OWN-PIECE FEATURES: 4 pieces × 5, fixed slot by piece_id,
        # + 1 shared capture_opportunity slot =====
        # Per-piece features indexed strictly by piece.piece_id (matching
        # how _get_info()'s action_masks[piece.piece_id] already works) --
        # never reordered by turn order, unlike the board-state block
        # above. This is what lets the network distinguish the consequence
        # of choosing action=0 vs action=3 when multiple of the agent's
        # own pieces have simultaneously legal moves.
        own_piece_offset = bonus_offset + self.BONUS_FEATURES_SIZE
        seat = self.agent_player_idx if perspective_seat is None else perspective_seat
        agent_player = self.game.players[seat]

        threat_hit_counts = self._capture_threat_scores(agent_player)

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

            # capture_threat_score: roll-based, see _capture_threat_scores.
            # Deliberately NOT gated on `on_safe` -- a piece on an
            # opponent's starting square can still be captured via that
            # opponent's `enter` move (docs/RULES.md "Entering the Board"
            # rules 5-6), even though the square is "safe" in the general
            # sense.
            obs[base + 4] = min(threat_hit_counts[piece.piece_id] / 6.0, 1.0)

        # capture_opportunity: single shared slot, not per-piece. Derived
        # from PIECE_FEATURES_PER_PIECE (not hardcoded) so this stays in
        # sync if that constant ever changes -- matches how six_streak_offset
        # derives from OWN_PIECE_FEATURES_SIZE two lines below.
        obs[own_piece_offset + 4 * self.PIECE_FEATURES_PER_PIECE] = (
            self._capture_opportunity_score(agent_player)
        )

        # --- Six-streak: 1 value (final block) ---
        six_streak_offset = own_piece_offset + self.OWN_PIECE_FEATURES_SIZE
        obs[six_streak_offset] = self.consecutive_sixes / THREE_SIXES_LIMIT

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
