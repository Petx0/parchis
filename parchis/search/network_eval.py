"""
Phase B evaluate_fn: real trained-network priors + value, replacing Phase
A's uniform-priors/progress-heuristic placeholder (heuristic_eval.py).

Deliberately simplified relative to the plan's original "network's own
priors, greedily" note for simulated OPPONENT moves too: this module only
uses the network for the TREE's own priors/leaf-value (both always from
the searching agent's own seat -- no ambiguity there, see docstring
below). Opponents during simulation still use the fixed random policy,
same as Phase A -- avoids replicating env_selfplay.py's own opponent-
observation quirk (self-play always builds the observation from the
original training agent's seat, even for opponent moves) inside a second,
untested code path. A real, deliberate scope decision, not an oversight --
worth revisiting once this simpler version is validated (matches this
whole project's "screen cheap first" pattern).
"""

import numpy as np

from parchis.search.state_view import ObservationAdapter


def make_network_evaluate_fn(model, num_players):
    """Returns evaluate_fn(game, agent_seat, legal_moves, dice_roll) -> (priors, value),
    backed by `model` (a loaded MaskablePPO). Always builds the observation
    from `agent_seat`'s own perspective -- this evaluate_fn is only ever
    called to evaluate the SEARCHING AGENT's own tree nodes (see mcts.py's
    module docstring: only the agent's own decisions are ever nodes), so
    there's no self-play-style "observation for a seat that isn't the one
    actually deciding" ambiguity to resolve here.

    `dice_roll`/`pending_bonus` feed the observation's dice-onehot/bonus-flag
    blocks; this module only ever sees fresh (non-bonus) rolls, since that's
    the only kind of decision mcts.py ever calls evaluate_fn for (see its
    "Search scope" note) -- pending_bonus is always None here.
    """
    adapter = ObservationAdapter(num_players=num_players)

    def evaluate_fn(game, agent_seat, legal_moves, dice_roll):
        obs = adapter.observation(game, agent_seat, current_dice_roll=dice_roll,
                                   pending_bonus=None, consecutive_sixes=0)
        mask = adapter.action_mask(legal_moves)

        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(obs_tensor, action_masks=mask[None, :])
        probs = distribution.distribution.probs.detach().cpu().numpy()[0]
        value_tensor = model.policy.predict_values(obs_tensor)
        value = float(value_tensor.item())

        priors = {piece.piece_id: float(probs[piece.piece_id]) for piece, _np, _mt in legal_moves}
        total = sum(priors.values())
        if total <= 0:
            # Degenerate (shouldn't happen given a legal action_masks
            # matches legal_moves exactly) -- fall back to uniform rather
            # than dividing by zero.
            n = len(priors)
            priors = {a: 1.0 / n for a in priors}
        else:
            priors = {a: p / total for a, p in priors.items()}

        return priors, value

    return evaluate_fn
