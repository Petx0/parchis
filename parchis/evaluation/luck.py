"""
Luck-adjusted scoring for dice games (docs/V2_PLAN.md §4, C8; the "luck-adjusted result" of
GNU Backgammon's evaluator-based variance reduction).

Every dice roll moves the value of the game. For the roll r made in pre-roll state S define, for
every seat,

    luck_r[seat] = V(S, r)[seat] - (1/6) * sum over faces f of V(S, f)[seat]

where V(S, f) is a FIXED evaluator's per-seat win-probability vector for the post-roll state
(S, f). The die is fair, so E_r[ luck_r | S ] = 0 exactly, for any evaluator whatsoever. Hence

    adjusted[seat] = outcome[seat] - sum over the game's rolls of luck_r[seat]

has the same expectation as the raw 0/1 outcome and a smaller variance whenever V is correlated
with what a roll is really worth. This is a control variate with a known (zero) mean: the
evaluator's quality decides how much variance is removed, never whether the estimate is biased.
The adjustment composes with duplicate (seat-swapped, same-seed) pairing, which removes only the
luck shared by the two games of a pair; this removes the rest, roll by roll.

LuckTracker wraps game.dice.roll. It is installed after Game() so the colour/starting-player
rolls of Game.__init__ are excluded, like arena's roll recorder. At roll time
game.current_player_idx is the roller, no bonus is pending (bonus moves are resolved inside the
turn before the next roll) and the roller's six-streak equals the number of rolls already made
this turn (every earlier roll of a turn was a 6, or the turn would have ended), which is what
parchis.az.turn_context hands the acting agent for the same roll. One batched forward of six
rows per roll; the random stream is untouched, so dice sequences, moves and raw results are
identical with and without a tracker.
"""

import numpy as np

from parchis.az import encoding, search

NUM_FACES = 6


def face_values(evaluator, game, mover, consecutive_sixes):
    """(6, num_players) array, absolute seat order: the evaluator's value of the post-roll state for
    each face 1..6. An evaluator with face_values(game, mover, consecutive_sixes) is called directly
    (NetLuckEvaluator, SearchLuckEvaluator); anything else is duck-typed on
    encode(game, seat, roll=, consecutive_sixes=) + evaluate_batch(rows, seats), six rows,
    e.g. a bare parchis.az.agent.NetEvaluator."""
    direct = getattr(evaluator, "face_values", None)
    if direct is not None:
        return np.asarray(direct(game, mover, consecutive_sixes), dtype=np.float64)
    rows = np.stack([evaluator.encode(game, mover, roll=face, consecutive_sixes=consecutive_sixes)
                     for face in range(1, NUM_FACES + 1)])
    return np.asarray(evaluator.evaluate_batch(rows, [mover] * NUM_FACES), dtype=np.float64)


class NetLuckEvaluator:
    """The net's own post-roll value, one encode per roll (encoding.encode_all_rolls) and one
    forward of six rows -- identical values to the duck-typed path over a NetEvaluator."""

    def __init__(self, net_evaluator):
        self.net_evaluator = net_evaluator

    def face_values(self, game, mover, consecutive_sixes):
        rows = encoding.encode_all_rolls(game, mover, consecutive_sixes=consecutive_sixes)
        return self.net_evaluator.evaluate_batch(rows, [mover] * NUM_FACES)


class SearchLuckEvaluator:
    """A sharper evaluator: the root value of a depth-`depth` search for each face (what the roll
    lets the mover do, not just what the net reads off the position); falls back to the net's
    post-roll value when the face leaves no legal move (search returns a draw vector there).
    Roughly depth-1-agent cost per face, so about six times NetLuckEvaluator."""

    def __init__(self, net_evaluator, depth=1):
        self.net_evaluator = net_evaluator
        self.depth = depth

    def face_values(self, game, mover, consecutive_sixes):
        out = np.empty((NUM_FACES, game.num_players), dtype=np.float64)
        for face in range(1, NUM_FACES + 1):
            move, _values, root = search.search(game, roll=face, consecutive_sixes=consecutive_sixes,
                                                depth=self.depth, evaluator=self.net_evaluator)
            if move is None:
                root = self.net_evaluator(game, mover, roll=face, consecutive_sixes=consecutive_sixes)
            out[face - 1] = root
        return out


class LuckTracker:
    """Per-seat cumulative luck of one game, in the evaluator's win-probability units.

    evaluator: see face_values() -- NetLuckEvaluator / SearchLuckEvaluator, or a bare
    parchis.az.agent.NetEvaluator (duck-typed, six encodes per roll)."""

    def __init__(self, game, evaluator):
        self.game = game
        self.evaluator = evaluator
        self.luck = np.zeros(game.num_players, dtype=np.float64)
        self.n_rolls = 0
        self._turn_number = None
        self._rolls_this_turn = 0
        self._original_roll = game.dice.roll
        game.dice.roll = self._roll

    def _roll(self):
        game = self.game
        if game.turn_number != self._turn_number:
            self._turn_number = game.turn_number
            self._rolls_this_turn = 0
        mover = game.current_player_idx
        values = face_values(self.evaluator, game, mover, self._rolls_this_turn)
        roll = self._original_roll()
        self.luck += values[roll - 1] - values.mean(axis=0)
        self.n_rolls += 1
        self._rolls_this_turn += 1
        return roll
