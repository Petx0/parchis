"""
Full-width expectimax search over decision/chance nodes
(docs/AGENT_REBUILD_PLAN.md §2.3) -- replaces parchis/search/mcts.py's
PUCT-guided sampling for this package. Branching factor here is ~2.76 with
an exactly-6-outcome chance node (§1.1), the regime where exhaustive
expectimax dominates sampled MCTS; MCTS earns its keep only when the
branching factor makes exhaustive expansion impossible.

    value(state, depth) -> np.ndarray[num_players]      # win-probability vector
      terminal            -> one-hot on the winner
      depth == 0          -> evaluator(state)
      DECISION node       -> for each legal move: apply -> value(child, depth-1)
                             return the child vector maximising v[player_to_move]  # max^n
      CHANCE node         -> (1/6) * sum over the 6 faces of value(child, depth-1)

A DECISION node is one choose_move: a turn-start roll, a six-again reroll,
OR a bonus-chain move -- uniform treatment, fixing §1.4's real-vs-simulated
inconsistency (parchis/search/mcts.py resolved bonus/six-again via a fixed
random policy during simulation) by construction. A CHANCE node is one
Dice.roll(), enumerated exactly over its 6 faces, never sampled (§1.4's
other fix -- mcts.py sampled one roll per node and fixed it for every
visit). max^n on the win-probability vector reduces to minimax at 2
players and is the correct generalization at 3-4 (§2.2) -- no separate
per-player-count search logic.

Depth counts DECISION LAYERS to fully expand before falling back to a
direct evaluator call, not individual game.play_turn() turns: depth=1
expands only the root's own ~2.9 legal moves (§1.1) and evaluates each
result immediately (no chance/bonus resolution attempted -- see
_evaluate_immediately) for ~3 leaves total; depth=2 additionally resolves
the intervening chance node exactly (6 faces) before expanding a second
decision layer, for ~2.9*6*2.9=~54 leaves; depth=3 is ~54*6*2.9=~940
leaves (both matching §1.1/Part 3 item 8's own measured/targeted figures).
A bonus-chain decision reached WITHIN the depth budget is its own decision
layer too (uniform treatment), just without the *6 chance factor (bonus
squares are a fixed rule constant, never a dice roll).

State transitions use Game.snapshot()/restore() exclusively, never
copy.deepcopy -- see docs/AGENT_REBUILD_PLAN.md Part 3 item 1 / §1.1 for
why (~20x cheaper per round trip on this class, measured).

`evaluator` contract: a callable
    evaluator(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0)
        -> np.ndarray[num_players]
returning a win-probability-like vector indexed by ABSOLUTE seat number
(0..num_players-1) -- NOT the canonical relative-to-observer channel order
parchis.az.encoding uses internally. A real net-backed evaluator (see
parchis/az/agent.py's NetEvaluator) is responsible for that remapping; a
test oracle (e.g. "value = each seat's own progress") can just return
absolute-order values directly, with no knowledge of the encoding's
relative convention needed.
"""

import numpy as np

from parchis.game.board import Board
from parchis.game.constants import (
    BONUS_TURN_ROLL, THREE_SIXES_LIMIT, CAPTURE_BONUS_SQUARES, FINISH_BONUS_SQUARES,
)
from parchis.game.game import Game

DEFAULT_DEPTH = 2


def _one_hot(seat, num_players):
    v = np.zeros(num_players, dtype=np.float64)
    v[seat] = 1.0
    return v


def _draw_vector(num_players):
    return np.full(num_players, 1.0 / num_players, dtype=np.float64)


def _evaluate_immediately(game, evaluator, move_info, roll, pending_bonus, consecutive_sixes):
    """Depth has run out right after a move: hand the evaluator the
    IMMEDIATE post-move state without resolving whatever would normally
    come next (a chance node's 6-way roll, or a bonus decision) -- this is
    what keeps depth=1 to ~3 leaves rather than ~3*6 (see module docstring).

    Three cases, in order of how much certainty is available:
      - The move captured/finished: the resulting bonus is a fixed rule
        consequence (not a chance outcome), so this IS a fully well-defined
        pre-decision state (§2.2) -- encoded exactly, same mover, roll=None.
      - The move cleanly ends the turn (roll wasn't 6, or this was already
        a bonus move): also fully determined -- it's unambiguously about to
        be the NEXT player's turn, roll not yet drawn. Advances
        game.next_player() (undone by the caller's snapshot/restore, not
        here) so the evaluator's canonical observer is the player actually
        about to decide, matching §2.3's "player_to_move" convention.
      - The move continues via six-again (roll == 6, no bonus): genuinely
        ambiguous at this depth -- rerolling stays with the SAME player,
        but that isn't distinguishable in the encoding from "next player,
        roll unknown" without extending it further. Accepted, documented
        simplification: evaluated from the current mover's own
        perspective; the six-streak feature (consecutive_sixes) at least
        signals a reroll is more likely here than the base rate.
    """
    if move_info.captured:
        return evaluator(game, game.current_player_idx, roll=None,
                          pending_bonus={'type': 'capture_bonus', 'squares': CAPTURE_BONUS_SQUARES},
                          consecutive_sixes=consecutive_sixes)
    if move_info.new_position == Board.FINAL_POSITION:
        return evaluator(game, game.current_player_idx, roll=None,
                          pending_bonus={'type': 'finish_bonus', 'squares': FINISH_BONUS_SQUARES},
                          consecutive_sixes=consecutive_sixes)

    six_again = (pending_bonus is None and roll == BONUS_TURN_ROLL)
    if not six_again:
        game.next_player()
        return evaluator(game, game.current_player_idx, roll=None, pending_bonus=None,
                          consecutive_sixes=0)
    return evaluator(game, game.current_player_idx, roll=None, pending_bonus=None,
                      consecutive_sixes=consecutive_sixes)


def _resolve_turn_continuation(game, roll, pending_bonus, consecutive_sixes,
                                second_six_piece, second_six_entered_home,
                                depth, evaluator, num_players):
    """A roll's decision (and any bonus chain it triggered) has fully
    bottomed out with depth still remaining: resolve whatever comes next
    -- the SAME player rerolls (six-again, only when the ORIGINAL roll of
    this whole cycle was a 6 and no bonus is pending) or the turn passes to
    the next player. Both need a fresh roll, always exactly enumerated
    (chance nodes are never depth-limited -- see module docstring). `depth`
    here is already the depth for the resulting next decision layer (the
    caller decremented it)."""
    six_again = (pending_bonus is None and roll == BONUS_TURN_ROLL)
    if six_again:
        return _chance_node(game, consecutive_sixes, second_six_piece,
                             second_six_entered_home, depth, evaluator, num_players)

    game.next_player()
    return _chance_node(game, 0, None, False, depth, evaluator, num_players)


def _chance_node(game, consecutive_sixes, second_six_piece, second_six_entered_home,
                  depth, evaluator, num_players):
    """Exactly enumerate game.get_current_player()'s next roll over all 6
    faces, weighted 1/6 each (§1.4/§2.3: fixes mcts.py's open-loop chance
    sampling -- with only 6 outcomes, enumerating them exactly is cheaper
    than sampling them badly). One face may trigger the three-sixes
    penalty (Game.apply_three_sixes_penalty) instead of a decision -- the
    third six is never offered as a move, and does NOT itself grant a
    further reroll (docs/RULES.md: "they do not get to use the third 6")."""
    total = np.zeros(num_players, dtype=np.float64)
    for face in range(1, 7):
        new_streak = consecutive_sixes + 1 if face == BONUS_TURN_ROLL else 0
        if new_streak == THREE_SIXES_LIMIT:
            # Advancing to the next player here costs a depth unit too,
            # for the exact same reason _decision_value's "no legal move"
            # branch does (see its comment): exact expectimax must also
            # explore the branch where three-sixes penalties keep firing
            # indefinitely, which real single-game simulations never
            # encounter but exhaustive enumeration can recurse into
            # without bound if this transition were free.
            snap = game.snapshot()
            Game.apply_three_sixes_penalty(game.board, second_six_piece, second_six_entered_home)
            game.next_player()
            if depth <= 0:
                total += evaluator(game, game.current_player_idx, roll=None,
                                    pending_bonus=None, consecutive_sixes=0)
            else:
                total += _chance_node(game, 0, None, False, depth - 1, evaluator, num_players)
            game.restore(snap)
        else:
            total += _decision_value(game, face, None, new_streak, depth, evaluator, num_players)
    return total / 6.0


def _expand_decision(game, roll, pending_bonus, consecutive_sixes, depth, evaluator, num_players):
    """Shared core for both search() (the real root, which also wants the
    full per-move breakdown) and _decision_value (recursive nodes, which
    only need the aggregated max^n value). Assumes legal_moves is
    non-empty and depth > 0 -- callers handle both edge cases first.

    Returns:
        dict {piece_id: np.ndarray[num_players]} -- one entry per legal
        move at THIS decision, each the value of choosing it.
    """
    player = game.get_current_player()
    mover_seat = game.current_player_idx
    effective_value = roll if pending_bonus is None else pending_bonus['squares']
    legal_moves = game.get_legal_moves(player, effective_value)

    move_values = {}
    for piece, new_position, move_type in legal_moves:
        old_position = piece.position
        snap = game.snapshot()
        move_info = game.execute_move(piece, new_position, move_type)

        if player.has_won():
            child_value = _one_hot(mover_seat, num_players)
        elif depth - 1 <= 0:
            child_value = _evaluate_immediately(
                game, evaluator, move_info, roll, pending_bonus, consecutive_sixes,
            )
        elif move_info.captured:
            child_value = _decision_value(
                game, None, {'type': 'capture_bonus', 'squares': CAPTURE_BONUS_SQUARES},
                consecutive_sixes, depth - 1, evaluator, num_players,
            )
        elif move_info.new_position == Board.FINAL_POSITION:
            child_value = _decision_value(
                game, None, {'type': 'finish_bonus', 'squares': FINISH_BONUS_SQUARES},
                consecutive_sixes, depth - 1, evaluator, num_players,
            )
        else:
            this_second_six_piece = None
            this_second_six_entered_home = False
            if pending_bonus is None and consecutive_sixes == 2:
                this_second_six_piece = piece
                this_second_six_entered_home = (
                    old_position is not None and old_position < Board.HOME_COLUMN_START
                    and new_position >= Board.HOME_COLUMN_START
                )
            child_value = _resolve_turn_continuation(
                game, roll, pending_bonus, consecutive_sixes,
                this_second_six_piece, this_second_six_entered_home,
                depth - 1, evaluator, num_players,
            )

        game.restore(snap)
        move_values[piece.piece_id] = child_value

    return move_values


def _decision_value(game, roll, pending_bonus, consecutive_sixes, depth, evaluator, num_players):
    """Value at a decision node for game.get_current_player() (recursive
    case -- discards the per-move breakdown _expand_decision computes,
    keeping only this node's own max^n aggregate for its parent)."""
    if game.game_over:
        winner_seat = game.players.index(game.winner) if game.winner is not None else None
        return _one_hot(winner_seat, num_players) if winner_seat is not None else _draw_vector(num_players)

    player = game.get_current_player()
    mover_seat = game.current_player_idx
    effective_value = roll if pending_bonus is None else pending_bonus['squares']
    legal_moves = game.get_legal_moves(player, effective_value)

    if not legal_moves:
        # A decision with no choice still costs a depth unit, exactly like
        # one with choices (_expand_decision below) -- otherwise a chain
        # of "no legal move" decisions (e.g. exhaustively exploring the
        # branch where a player keeps NOT rolling the 5 they need to
        # enter) would never consume the search's depth budget and could
        # recurse without bound: a single REAL game always eventually
        # rolls something else, but exact expectimax must also explore the
        # branch where it doesn't, for arbitrarily long. Confirmed by a
        # RecursionError before this fix (mover's own state can't help
        # here since it doesn't change: only depth stops the recursion).
        if depth <= 0:
            return evaluator(game, mover_seat, roll=roll, pending_bonus=pending_bonus,
                              consecutive_sixes=consecutive_sixes)
        # No piece moved this roll -> no piece to penalize even if this
        # streak later hits 3 (docs/RULES.md: "no piece was moved with the
        # second 6 ... no piece is captured").
        return _resolve_turn_continuation(
            game, roll, pending_bonus, consecutive_sixes,
            None, False, depth - 1, evaluator, num_players,
        )

    if depth <= 0:
        return evaluator(game, mover_seat, roll=roll, pending_bonus=pending_bonus,
                          consecutive_sixes=consecutive_sixes)

    move_values = _expand_decision(game, roll, pending_bonus, consecutive_sixes,
                                    depth, evaluator, num_players)
    best_piece_id = max(move_values, key=lambda pid: move_values[pid][mover_seat])
    return move_values[best_piece_id]


def search(game, roll=None, pending_bonus=None, consecutive_sixes=0,
           depth=DEFAULT_DEPTH, evaluator=None):
    """
    Top-level entry point: game.get_current_player() must choose among
    game.get_legal_moves(current_player, effective_value), where
    effective_value is `roll` (a fresh dice roll, 1-6) or
    pending_bonus['squares'] (20 or 10) -- exactly one of the two should be
    given, matching parchis.az.encoding.encode()'s own contract.

    `game` is NEVER mutated: every move explored is applied via
    Game.execute_move and undone via Game.restore(snapshot) before
    returning (verified by parchis/tests/test_search.py via a snapshot
    hash before/after).

    Returns:
        tuple(move_or_None, move_values, root_value):
          move: the (piece, new_position, move_type) maximizing the
              mover's own win probability (max^n) -- None if no legal move.
          move_values: {piece_id: np.ndarray[num_players]} for EVERY legal
              move at the root -- §2.3's policy training target is a
              softmax over these at temperature tau, a better target than
              MCTS visit counts at this branching factor. Empty if no
              legal move.
          root_value: move_values[move's piece_id] -- this position's
              overall value vector; a draw vector if there was no legal
              move (mirrors the "no legal moves -> turn passes" rule: the
              position itself isn't decided by that, so there is no
              single well-defined win/loss vector to report without
              recursing further, which the root deliberately does not do
              on the caller's behalf).
    """
    if evaluator is None:
        raise ValueError("search() requires an evaluator")
    num_players = game.num_players
    player = game.get_current_player()
    mover_seat = game.current_player_idx
    effective_value = roll if pending_bonus is None else pending_bonus['squares']
    legal_moves = game.get_legal_moves(player, effective_value)

    if not legal_moves:
        return None, {}, _draw_vector(num_players)

    move_values = _expand_decision(game, roll, pending_bonus, consecutive_sixes,
                                    depth, evaluator, num_players)
    best_piece_id = max(move_values, key=lambda pid: move_values[pid][mover_seat])
    best_move = next(m for m in legal_moves if m[0].piece_id == best_piece_id)
    return best_move, move_values, move_values[best_piece_id]
