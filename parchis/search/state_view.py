"""
Reuse ParchisEnv's validated observation/mask construction for game states
explored during search, without going through its Gym-API step() loop.

_get_observation() (parchis/rl/env.py) only reads instance attributes
(self.game, self.agent_player_idx, self.current_dice_roll, self.pending_bonus,
self.consecutive_sixes) -- it never mutates game state itself. So a real
(throwaway) ParchisEnv can be constructed once and have those attributes
"puppeted" to mirror whatever state MCTS is currently exploring, then its
own unmodified _get_observation()/_get_info() called directly. This avoids
duplicating the observation formula (piece-indexed slots, capture threat/
opportunity, six-streak, bonus flags, ...) a second time, at the cost of
this slightly unusual construction. Zero changes to parchis/rl/env.py.
"""

import numpy as np

from parchis.rl.env import ParchisEnv


class ObservationAdapter:
    """One throwaway ParchisEnv per num_players, reused across many calls
    (constructing a fresh ParchisEnv per query would be wasteful -- its
    __init__ does real work, e.g. building the observation_space)."""

    def __init__(self, num_players):
        self.num_players = num_players
        self._env = ParchisEnv(num_players=num_players)

    def _sync(self, game, agent_player_idx, current_dice_roll, pending_bonus, consecutive_sixes):
        env = self._env
        env.game = game
        env.agent_player_idx = agent_player_idx
        env.current_dice_roll = current_dice_roll
        env.pending_bonus = pending_bonus
        env.consecutive_sixes = consecutive_sixes

    def observation(self, game, agent_player_idx, current_dice_roll=None,
                     pending_bonus=None, consecutive_sixes=0):
        """The observation ParchisEnv would produce for `agent_player_idx`'s
        perspective at this exact game state. `current_dice_roll`/`pending_bonus`
        matter to the observation's dice-onehot/bonus-flag blocks -- pass
        current_dice_roll for a fresh (non-bonus) decision, or pending_bonus
        (e.g. {'type': 'capture_bonus', 'squares': 20}) for a bonus-chain one."""
        self._sync(game, agent_player_idx, current_dice_roll, pending_bonus, consecutive_sixes)
        return self._env._get_observation()

    def action_mask(self, legal_moves):
        """Same construction _get_info() uses (parchis/rl/env.py) -- pulled
        out standalone since MCTS always already has `legal_moves` on hand
        and doesn't need _get_info()'s other pending_bonus/current_dice_roll
        bookkeeping just to get the mask."""
        mask = np.zeros(4, dtype=np.int8)
        for piece, _new_pos, _move_type in legal_moves:
            mask[piece.piece_id] = 1
        if len(legal_moves) == 0:
            mask[:] = 1
        return mask
