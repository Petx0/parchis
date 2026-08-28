"""
Wires encoding + net + search.py into a Player.choose_move-compatible /
arena-factory-compatible agent (docs/AGENT_REBUILD_PLAN.md Part 3 item 9):
factory(game, seat, roll_box) -> choose_move_fn, matching
parchis/search/agents.py's existing convention so this plugs straight into
parchis/evaluation/arena.py (and, through it, parchis/evaluation/duplicate.py).

depth>=1 only: routes through parchis.az.search.search() at that depth. A
"depth=0, no search" agent is just whatever raw policy the evaluator is
built from, used directly -- e.g. parchis.agents.heuristic's own
choose_move_with_weights for the hand-built evaluator, or a net's policy
head for a learned one. Not this module's concern: Part 3 item 10's
"heuristic + depth 0" baseline is exactly
parchis.agents.heuristic.make_heuristic_agent_factory(...) used as-is, no
wrapping needed.

Correctly distinguishes a fresh dice roll from a bonus-chain continuation
purely by observation via parchis.az.turn_context.TurnContextTracker -- see
that module's docstring for why (§1.4's mcts.py bug, restated).
"""

import numpy as np

from parchis.agents.decision_recorder import DecisionRecord
from parchis.az import encoding, search
from parchis.az.net import value_probs
from parchis.az.turn_context import TurnContextTracker
from parchis.game.board import Board


class NetEvaluator:
    """search.py evaluator wrapping a trained parchis.az.net.NumpyAZNet:
    encodes from `observer_seat`'s perspective, runs the net, and remaps
    the value head's RELATIVE-to-observer channel order back to the
    ABSOLUTE seat order search.py's evaluator contract requires (the
    encoding/net only ever see/produce relative channel order -- see
    parchis/az/search.py's module docstring).

    Also exposes encode()/evaluate_batch() -- search.py's _Collector
    duck-types on this pair to switch from evaluating every leaf eagerly
    (one Python call, one net.forward() each) to collecting every leaf's
    row across a WHOLE search and running exactly one batched
    net.forward() at the end (see search.py's BATCHED LEAF EVALUATION
    module docstring section). __call__ above is unchanged and still used
    directly wherever a single one-off evaluation is wanted (e.g.
    parchis.evaluation.puzzles.runner); it's now just a batch-of-one call
    through the same two methods, not a separate code path -- see
    test_search.py::test_net_evaluator_batched_matches_eager_call_path for
    the cross-check that the two stay in agreement."""

    def __init__(self, numpy_net):
        self.numpy_net = numpy_net

    def __call__(self, game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        row = self.encode(game, observer_seat, roll=roll, pending_bonus=pending_bonus,
                           consecutive_sixes=consecutive_sixes)
        return self.evaluate_batch(row[None, :], [observer_seat])[0]

    def encode(self, game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        """The cheap, pure part of evaluation: a fixed-size row, safe to
        compute now and batch for later even though `game` itself will be
        mutated/restored by search.py before the batch is ever run."""
        return encoding.encode(game, observer_seat, roll=roll, pending_bonus=pending_bonus,
                                consecutive_sixes=consecutive_sixes)

    def evaluate_batch(self, rows, observer_seats):
        """rows: (batch, input_size) array of encode() outputs.
        observer_seats: the observer_seat each row was encoded from
        (needed per-row since np.roll's shift differs by observer).
        Returns an (batch, num_players) array, one absolute-seat-order
        value vector per row -- ONE net.forward() call for the whole
        batch, which is the entire point of this method existing."""
        _policy_logits, value_logits = self.numpy_net.forward(rows)
        relative_probs = value_probs(value_logits)
        return np.stack([
            np.roll(relative_probs[i], observer_seats[i])
            for i in range(len(observer_seats))
        ])


def heuristic_position_evaluator(game, observer_seat=None, roll=None, pending_bonus=None,
                                  consecutive_sixes=0, scale=4.0):
    """A hand-built (non-learned) search.py evaluator -- Part 3 item 10's
    gate needs one ("if search doesn't help a hand-built evaluator, it
    will not help a learned one"). Mirrors parchis/search/heuristic_eval.py's
    existing Phase A placeholder shape (tanh of a scaled progress
    differential), generalized from a single scalar to a full per-seat
    vector (max^n needs one for every seat, not just the searching agent's
    own), and using encoding.py's colour-invariant relative progress
    rather than parchis.rl.rewards.calculate_normalized_progress's
    raw-absolute-position one -- see encoding._relative_piece_progress's
    docstring for why that distinction matters here."""
    progresses = np.array([
        sum(encoding._relative_piece_progress(p, Board.STARTING_POSITIONS[player.color])
            for p in player.pieces) / 4.0
        for player in game.players
    ])
    n = len(progresses)
    values = np.zeros(n)
    for seat in range(n):
        other_mean = (progresses.sum() - progresses[seat]) / (n - 1) if n > 1 else 0.0
        values[seat] = np.tanh(scale * (progresses[seat] - other_mean))
    return values


def make_search_agent_factory(evaluator, depth):
    """factory(game, seat, roll_box) -> choose_move_fn, for search.py at a
    fixed `depth` (>= 1) using `evaluator` (see parchis.az.search's
    evaluator contract -- NetEvaluator/heuristic_position_evaluator above
    both satisfy it)."""
    if depth < 1:
        raise ValueError(f"make_search_agent_factory requires depth >= 1, got {depth}")

    def factory(game, seat, roll_box):
        tracker = TurnContextTracker()

        def choose_move(legal_moves):
            if not legal_moves:
                tracker.record_move(game, None)
                return None

            roll, pending_bonus, consecutive_sixes = tracker.context_for(roll_box)
            move, _move_values, _root_value = search.search(
                game, roll=roll, pending_bonus=pending_bonus,
                consecutive_sixes=consecutive_sixes, depth=depth, evaluator=evaluator,
            )
            tracker.record_move(game, move)
            return move

        return choose_move

    return factory


def make_recording_search_agent_factory(evaluator, depth, recorder):
    """Visualization-only sibling of make_search_agent_factory: IDENTICAL
    decision logic (same evaluator/depth/search.search() call, same
    TurnContextTracker use, same early-return-on-no-legal-moves shape), but
    also appends a DecisionRecord of the move_values/root_value
    make_search_agent_factory itself discards -- see
    parchis/agents/decision_recorder.py's module docstring for why this is
    a sibling rather than an optional parameter on the existing factory
    (that one is live production code, used every round.py promotion match;
    this one never runs on that path). A regression test
    (test_agent.py::test_recording_factory_matches_plain_factory) asserts
    both factories choose the identical move for the same seed, so the two
    implementations can't silently drift apart."""
    if depth < 1:
        raise ValueError(f"make_recording_search_agent_factory requires depth >= 1, got {depth}")

    def factory(game, seat, roll_box):
        tracker = TurnContextTracker()

        def choose_move(legal_moves):
            if not legal_moves:
                tracker.record_move(game, None)
                return None

            roll, pending_bonus, consecutive_sixes = tracker.context_for(roll_box)
            move, move_values, root_value = search.search(
                game, roll=roll, pending_bonus=pending_bonus,
                consecutive_sixes=consecutive_sixes, depth=depth, evaluator=evaluator,
            )
            recorder.records.append(DecisionRecord(
                seat=seat, turn_number=game.turn_number,
                decision_index_in_turn=recorder.next_index(game.turn_number),
                kind="search", root_value=root_value, move_values=move_values,
                chosen_piece_id=move[0].piece_id if move is not None else None,
            ))
            tracker.record_move(game, move)
            return move

        return choose_move

    return factory
