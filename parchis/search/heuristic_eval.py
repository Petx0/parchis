"""
Phase A placeholder evaluate_fn: no trained network yet. Uniform priors
over legal actions, and a cheap progress-based heuristic for leaf values --
validates the tree mechanics (PUCT selection, expansion, backprop) fully
independently of any trained network. Phase B (network_eval.py) replaces
this with a real policy/value network's outputs.
"""

import math

from parchis.rl.rewards import calculate_normalized_progress


def make_heuristic_evaluate_fn(scale=4.0):
    """Returns evaluate_fn(game, agent_seat, legal_moves, dice_roll) -> (priors, value).

    priors: uniform over legal_moves' action ids.
    value: tanh(scale * (agent_progress - mean_opponent_progress)), bounded
        to [-1, 1] to match the terminal win/loss value's own scale --
        positive when the agent is further along than its opponents on
        average. A cheap stand-in for a trained value network, nothing more.
    """
    def evaluate_fn(game, agent_seat, legal_moves, dice_roll):
        n = len(legal_moves)
        priors = {piece.piece_id: 1.0 / n for piece, _new_pos, _move_type in legal_moves} if n > 0 else {}

        agent_player = game.players[agent_seat]
        agent_progress = calculate_normalized_progress(agent_player)
        # Identity check, not player.player_id -- Game.__init__ rotates
        # self.players so the dice-determined starting player becomes list
        # index 0, but Player.player_id is assigned before that rotation
        # and never updated, so it can diverge from the player's own list
        # index. agent_seat is always a list index (see mcts.py's
        # _install_fixed_policy docstring for the full explanation).
        opponents = [p for p in game.players if p is not agent_player]
        opp_progress = (
            sum(calculate_normalized_progress(p) for p in opponents) / len(opponents)
            if opponents else 0.0
        )

        value = math.tanh(scale * (agent_progress - opp_progress))
        return priors, value

    return evaluate_fn
