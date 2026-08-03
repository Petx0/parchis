"""
Self-play environment wrapper for Parchís.

The learning agent occupies a randomly-assigned seat each episode (see
ParchisEnv.agent_player_idx, reassigned every reset()). Every other seat is
resolved by ParchisEnv's opponent_policy_fn hook (see parchis.rl.env), which
this wrapper points at a frozen policy model instead of the default random
policy. ParchisEnv itself owns the one and only "resolve opponent turns"
loop, including chained bonus moves -- this wrapper never re-implements it.
"""

import random

import gymnasium as gym

from parchis.rl.env import ParchisEnv
from parchis.rl import opponent_pool


class ParchisSelfPlayEnv(gym.Env):
    """
    Self-play wrapper for Parchís environment.

    The learning agent (a randomly-assigned seat each episode) plays against
    opponent(s) using a policy model. The opponent model is frozen and
    periodically updated during training. Supports 2-4 players.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, opponent_model=None, num_players=2, opponent_weight=0.5,
                 reward_type="progress_delta", render_mode=None, pool_seed=None):
        """
        Initialize the self-play environment.

        Args:
            opponent_model: MaskablePPO model to use for opponents (if None, uses random).
                Equivalent to update_opponent_model(opponent_model) at construction time --
                a pool of exactly one member.
            num_players: Number of players (2-4)
            opponent_weight: α value for turn-cycle reward (default 0.5)
            reward_type: Reward structure to use, see ParchisEnv.VALID_REWARD_TYPES
                (default "progress_delta")
            render_mode: Render mode
            pool_seed: Seed for the dedicated opponent-pool-sampling RNG. Deliberately
                separate from the game's own dice-roll RNG (the global `random` module,
                seeded via ParchisEnv.reset(seed=...)) -- pool selection must never draw
                from that shared stream, or it would perturb seeded reproducibility of
                the dice rolls themselves.
        """
        super().__init__()

        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")

        self.opponent_model = opponent_model
        self.opponent_move_count = 0

        # Opponent pool: sampled once per episode (at reset()), not per
        # opponent move -- see reset() below. A single-model construction
        # (the common case: update_opponent_model()) is just a pool of 1.
        self.opponent_pool = [opponent_model] if opponent_model is not None else []
        self.opponent_pool_weights = [1.0] if opponent_model is not None else []
        self.opponent_selection_counts = {}
        self._pool_rng = random.Random(pool_seed)

        # ParchisEnv calls self._choose_opponent_move for every opponent
        # move (regular and bonus); we don't intercept or duplicate that
        # loop here.
        self.base_env = ParchisEnv(
            num_players=num_players,
            render_mode=render_mode,
            reward_type=reward_type,
            opponent_policy_fn=self._choose_opponent_move,
            opponent_weight=opponent_weight,
        )

        # Copy spaces from base environment
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

    def update_opponent_pool(self, models, weights=None):
        """
        Replace the opponent pool with a new set of models.

        Args:
            models: Non-empty list of MaskablePPO-like models, oldest first /
                newest last (matches opponent_pool.compute_recency_weights'
                ordering convention).
            weights: Optional list, same length as models. None means uniform.

        One member is sampled (see reset()) at the start of each episode and
        used for every opponent seat that whole episode.
        """
        if not models:
            raise ValueError("models must not be empty")
        self.opponent_pool = list(models)
        self.opponent_pool_weights = list(weights) if weights is not None else [1.0] * len(models)
        self.opponent_selection_counts = {}

    def update_opponent_model(self, new_model):
        """
        Update the opponent model with a new version (back-compat wrapper:
        a pool of exactly one model, selected deterministically every episode).

        Args:
            new_model: New MaskablePPO model to use for opponents
        """
        self.update_opponent_pool([new_model])
        print(f"✓ Opponent model updated (opponents have made {self.opponent_move_count:,} moves so far)")

    def reset(self, seed=None, options=None):
        """Reset the environment.

        If the opponent pool is non-empty, samples one member for the whole
        upcoming episode (every opponent seat shares that identity, matching
        the existing single-opponent-per-episode structure) before delegating
        to the base env's reset.
        """
        if self.opponent_pool:
            idx = opponent_pool.sample_pool_index(self.opponent_pool_weights, self._pool_rng)
            self.opponent_model = self.opponent_pool[idx]
            self.opponent_selection_counts[idx] = self.opponent_selection_counts.get(idx, 0) + 1
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action):
        """Execute one step for the learning agent (whichever seat was
        randomly assigned this episode, see ParchisEnv.agent_player_idx)."""
        return self.base_env.step(action)

    def _choose_opponent_move(self, player, legal_moves):
        """
        Opponent-move policy passed into ParchisEnv as opponent_policy_fn:
        use the frozen opponent model when available, falling back to
        random selection. Called for every opponent move, including
        chained bonus moves.

        Args:
            player: The opponent whose turn it is
            legal_moves: List of legal moves for that player

        Returns:
            Chosen move tuple (piece, new_position, move_type) or None
        """
        if not legal_moves:
            return None

        if self.opponent_model is None:
            return player.choose_move(legal_moves)

        # Observation/action-mask are computed from self.base_env's live
        # game state; this is called synchronously from inside
        # self.base_env.step(), so current_player_idx already reflects
        # this opponent's turn.
        obs = self.base_env._get_observation()
        action_masks = self.base_env._get_info()['action_masks']

        try:
            action, _ = self.opponent_model.predict(
                obs,
                action_masks=action_masks,
                deterministic=False,  # Use stochastic policy for diversity
            )
            action = int(action)

            piece_moves = {}
            for move in legal_moves:
                piece, new_pos, move_type = move
                piece_moves[piece.piece_id] = (piece, new_pos, move_type)

            if action in piece_moves:
                self.opponent_move_count += 1
                return piece_moves[action]

            # Model chose an action with no matching legal move (shouldn't
            # happen given action_masks, but guard against it anyway).
            return player.choose_move(legal_moves)

        except Exception as e:
            # Fallback to random on any error
            print(f"Warning: Opponent model prediction failed ({e}), using random move")
            return player.choose_move(legal_moves)

    def render(self):
        """Render the environment."""
        return self.base_env.render()

    def close(self):
        """Clean up resources."""
        self.base_env.close()
