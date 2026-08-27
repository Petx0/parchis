"""
Handcrafted linear-score agent (docs/AGENT_REBUILD_PLAN.md §2.4).

Not optional: this is the absolute strength anchor the project otherwise
lacks (before this, "56% vs. our own checkpoint" had no way to be read as
"is this agent actually good at Parchís"), the bootstrap opponent for
generating the first training data, and a pool member that stops a
self-play lineage from collapsing into a single narrow strategy.

Scores each candidate move with a linear combination of ~10 features,
computed with the real rule engine (never hand-rolled distance math) via
Game.snapshot()/restore() to look at the position the move actually
produces without mutating the caller's game. See FEATURE_NAMES for the
feature list and _score_move for their definitions.
"""

import random

import numpy as np

from parchis.agents.decision_recorder import DecisionRecord
from parchis.game.board import Board
from parchis.rl.rewards import calculate_normalized_progress

FEATURE_NAMES = (
    "capture_value",           # sum of captured pieces' own pre-move progress
    "enters_from_base",        # 1.0 if this move brings a piece out of base
    "progress_gained",         # this piece's own progress delta from the move
    "lands_safe",              # 1.0 if the destination is safe (or finish/home)
    "lands_in_threat",         # post-move capture_threat_score at the destination
    "forms_blocking_blockade", # 1.0 if this creates a blockade crossing an
                                # opponent piece's very next roll
    "exact_finish",            # 1.0 if this move reaches square 76
    "home_column_advance",     # 1.0 if the destination is in the home column
    "leading_opponent_suppression",  # progress of the leader, IF a leader
                                      # piece is captured by this move
    "develops_most_behind_own_piece",  # 1.0 if this is our own least-advanced
                                        # on-board piece
)
NUM_FEATURES = len(FEATURE_NAMES)

# A reasonable hand-picked starting point, signs only: reward everything
# except walking into a threat. Magnitudes are not tuned -- see
# TUNED_WEIGHTS (fit by cem_tune_weights, see module bottom for how/when).
DEFAULT_WEIGHTS = np.array([
    1.0,   # capture_value
    0.5,   # enters_from_base
    1.0,   # progress_gained
    0.3,   # lands_safe
    -1.0,  # lands_in_threat
    0.5,   # forms_blocking_blockade
    0.8,   # exact_finish
    0.3,   # home_column_advance
    0.5,   # leading_opponent_suppression
    0.2,   # develops_most_behind_own_piece
], dtype=np.float64)

# Fit by cem_tune_weights(opponent_factories=[DEFAULT_WEIGHTS-heuristic, random],
# num_players=2, population_size=20, games_per_candidate=60, generations=10,
# seed=20260825) -- the final generation's population MEAN (not the single
# best-ever-sampled individual, which scored comparably but has more
# per-candidate evaluation noise baked in at only 60 games/candidate; see
# docs/AZ_DESIGN.md Phase 1 entry for both numbers and the held-out
# validation that picked between them). Held-out result (300 fresh games,
# seeds not used during tuning): 88.3% [84.2%, 91.5%] vs random, 62-66% vs
# DEFAULT_WEIGHTS across two independent seeds -- "clearly above an
# untuned one and far above random", as the plan expects.
TUNED_WEIGHTS = np.array([
    2.37, 0.30, 1.06, 0.68, -1.62, -0.40, 1.80, -0.63, 1.22, -0.16,
], dtype=np.float64)


def _own_progress_delta(piece, new_position, move_type):
    old_progress = 0.0 if piece.in_base else piece.position / Board.FINAL_POSITION
    new_progress = 1.0 if move_type == 'finish' else new_position / Board.FINAL_POSITION
    return new_progress - old_progress


def _lands_safe(new_position, move_type):
    if move_type == 'finish' or new_position >= Board.HOME_COLUMN_START:
        return 1.0
    return 1.0 if new_position in Board.SAFE_SQUARES else 0.0


def _threat_score_at(game, mover, position):
    """Fraction (clipped to [0, 1]) of the 6 dice faces, summed across every
    opponent (double threat = double risk, not deduplicated -- mirrors
    parchis/rl/env.py::_capture_threat_scores' own convention), on which
    that opponent has a legal move directly landing on `position` and
    capturing there right now. Single-roll only, no bonus-chain extension
    (unlike _capture_threat_scores) -- kept cheap since this runs once per
    candidate move, not once per observation."""
    if position is None or position >= Board.HOME_COLUMN_START:
        return 0.0
    hits = 0
    for opponent in game.players:
        if opponent is mover:
            continue
        for v in range(1, 7):
            for move in game.get_legal_moves(opponent, v):
                _piece, new_pos, _mt = move
                if new_pos == position and game.would_capture(move):
                    hits += 1
                    break  # at most one hit per (opponent, face value)
    return min(hits / 6.0, 1.0)


def _forms_blocking_blockade(game, mover, position):
    """1.0 if `position` is now a blockade (2 own pieces on a safe square)
    that lies within some opponent's on-board piece's path for at least one
    of the 6 faces -- i.e. it blocks that opponent's very next move, not
    just blockades in the abstract. Deliberately only checks one roll of
    lookahead, not deeper blocking value."""
    if position not in game.get_blockades():
        return 0.0
    for opponent in game.players:
        if opponent is mover:
            continue
        for piece in opponent.get_pieces_on_board():
            if piece.position >= Board.HOME_COLUMN_START:
                continue
            for v in range(1, 7):
                if position in game.compute_path(opponent, piece.position, v):
                    return 1.0
    return 0.0


def _leading_opponent(game, mover):
    opponents = [p for p in game.players if p is not mover]
    if not opponents:
        return None
    return max(opponents, key=calculate_normalized_progress)


def _score_move(game, mover, move, weights):
    """Score one candidate move for `mover` in `game`'s CURRENT state
    (game is not mutated: any move execution used to compute post-move
    features is undone via snapshot()/restore() before returning)."""
    piece, new_position, move_type = move
    features = np.zeros(NUM_FEATURES, dtype=np.float64)

    # --- Pre-move features (don't need the move actually applied) ---
    captured = game.would_capture(move)
    features[0] = sum(p.position / Board.FINAL_POSITION for p in captured)  # capture_value
    features[1] = 1.0 if move_type == 'enter' else 0.0  # enters_from_base
    features[2] = _own_progress_delta(piece, new_position, move_type)  # progress_gained
    features[3] = _lands_safe(new_position, move_type)  # lands_safe
    features[6] = 1.0 if move_type == 'finish' else 0.0  # exact_finish
    features[7] = 1.0 if new_position >= Board.HOME_COLUMN_START else 0.0  # home_column_advance

    leader = _leading_opponent(game, mover)
    if leader is not None and any(p.color == leader.color for p in captured):
        features[8] = calculate_normalized_progress(leader)  # leading_opponent_suppression

    on_board = mover.get_pieces_on_board()
    if on_board and piece in on_board:
        most_behind = min(on_board, key=lambda p: p.position)
        features[9] = 1.0 if piece is most_behind else 0.0  # develops_most_behind_own_piece

    # --- Post-move features: apply, query, undo ---
    snap = game.snapshot()
    game.execute_move(piece, new_position, move_type)
    features[4] = _threat_score_at(game, mover, new_position if move_type != 'finish' else None)
    features[5] = _forms_blocking_blockade(game, mover, new_position)
    game.restore(snap)

    return float(np.dot(weights, features))


def choose_move_with_weights(game, player, legal_moves, weights, rng=None):
    """Player.choose_move-shaped decision, but needs `game` (see
    make_heuristic_agent_factory for the arena-compatible wrapper that
    supplies it via closure). Ties broken randomly, matching
    Player.choose_move's own random-tiebreak convention elsewhere in this
    codebase."""
    if not legal_moves:
        return None
    rng = rng or random
    scored = [(_score_move(game, player, move, weights), move) for move in legal_moves]
    best_score = max(s for s, _m in scored)
    best_moves = [m for s, m in scored if s == best_score]
    return rng.choice(best_moves)


def make_heuristic_agent_factory(weights=None):
    """factory(game, seat, roll_box) -> choose_move_fn, matching
    parchis/search/agents.py's convention so this plugs straight into
    parchis/evaluation/arena.py (and, through it, duplicate.py/the ladder).
    `roll_box` is accepted but unused -- the heuristic only ever looks at
    `legal_moves` and live game state, never the raw dice roll itself."""
    w = DEFAULT_WEIGHTS if weights is None else weights

    def factory(game, seat, roll_box):
        player = game.players[seat]

        def choose_move(legal_moves):
            return choose_move_with_weights(game, player, legal_moves, w)

        return choose_move

    return factory


def make_recording_heuristic_agent_factory(weights=None, recorder=None):
    """Visualization-only sibling of make_heuristic_agent_factory: same
    choose_move_with_weights-driven decision, but also scores every legal
    move itself (rather than letting choose_move_with_weights do it
    internally) so it can append a DecisionRecord(kind="heuristic",
    move_scores=...) to `recorder` -- see
    parchis/agents/decision_recorder.py's module docstring for why
    move_scores is a plain {piece_id: float} dict, not a per-seat value
    vector like the search agent's DecisionRecord. Ties are broken the
    same way choose_move_with_weights does (uniformly among the top
    score); `recorder` is required (unlike the plain factory, this one has
    no reason to exist without one)."""
    w = DEFAULT_WEIGHTS if weights is None else weights

    def factory(game, seat, roll_box):
        player = game.players[seat]

        def choose_move(legal_moves):
            if not legal_moves:
                return None
            scored = [(_score_move(game, player, move, w), move) for move in legal_moves]
            best_score = max(s for s, _m in scored)
            best_moves = [m for s, m in scored if s == best_score]
            chosen = random.choice(best_moves)
            recorder.records.append(DecisionRecord(
                seat=seat, turn_number=game.turn_number,
                decision_index_in_turn=recorder.next_index(game.turn_number),
                kind="heuristic",
                move_scores={move[0].piece_id: score for score, move in scored},
                chosen_piece_id=chosen[0].piece_id,
            ))
            return chosen

        return choose_move

    return factory


DEFAULT_EPSILON = 0.15  # not specified by the plan; chosen as a moderate
                         # exploration rate -- frequent enough to diversify
                         # self-play data generation (Part 3 item 11's
                         # "ε-noisy heuristic" pool member) without
                         # swamping the heuristic's own signal.


def make_epsilon_noisy_heuristic_agent_factory(weights=None, epsilon=DEFAULT_EPSILON, seed=None):
    """factory(game, seat, roll_box) -> choose_move_fn: with probability
    `epsilon` picks a uniformly random legal move instead of the
    heuristic's own choice, otherwise defers to choose_move_with_weights
    exactly like make_heuristic_agent_factory. Adds exploration diversity
    to self-play data generation (docs/AGENT_REBUILD_PLAN.md Part 3 item
    11's "ε-noisy heuristic" pool member) -- a pool of only {tuned
    heuristic, random} would under-sample positions a competent-but-not-
    perfect player reaches, which is exactly the distribution the value
    net needs to learn from.

    `seed` seeds a PRIVATE random.Random instance (both the epsilon coin
    flip and the fallback uniform choice, and choose_move_with_weights's
    own tie-breaks) -- deliberately never the shared global `random`
    module, so sampling many noisy agents in a pool doesn't perturb each
    other's reproducibility or the game's own dice sequence.
    """
    w = DEFAULT_WEIGHTS if weights is None else weights
    rng = random.Random(seed)

    def factory(game, seat, roll_box):
        player = game.players[seat]

        def choose_move(legal_moves):
            if not legal_moves:
                return None
            if rng.random() < epsilon:
                return rng.choice(legal_moves)
            return choose_move_with_weights(game, player, legal_moves, w, rng=rng)

        return choose_move

    return factory


# --- CEM weight tuning ---

def cem_tune_weights(opponent_factories, num_players=2, population_size=16,
                      games_per_candidate=40, generations=8, elite_frac=0.25,
                      init_mean=None, init_std=1.0, min_std=0.05, seed=None,
                      max_turns=400):
    """
    Cross-entropy method over the 10 feature weights: each generation,
    sample `population_size` candidates ~ N(mean, diag(std^2)), evaluate
    each by win rate over `games_per_candidate` arena.play_match games
    against a fixed, round-robin mix of `opponent_factories`, refit
    (mean, std) from the top `elite_frac` fraction, repeat.

    Args:
        opponent_factories: non-empty list of arena-style
            factory(game, seat, roll_box) -> choose_move_fn callables the
            candidate plays against (split evenly across
            games_per_candidate; a candidate's own score is its overall
            win rate pooled across all of them).
        num_players: passed to arena.play_match (2-4).
        seed: seeds both the candidate sampling and the games played, for
            a reproducible tuning run.

    Returns:
        tuple(np.ndarray, list[dict]): (best mean weight vector found,
        per-generation history: [{'mean': array, 'std': array,
        'best_score': float, 'mean_score': float}, ...]).
    """
    from parchis.evaluation import arena

    if not opponent_factories:
        raise ValueError("opponent_factories must not be empty")

    rng = np.random.default_rng(seed)
    py_seed_stream = random.Random(seed)

    mean = np.array(DEFAULT_WEIGHTS if init_mean is None else init_mean, dtype=np.float64)
    std = np.full(NUM_FEATURES, init_std, dtype=np.float64)
    n_elite = max(1, int(round(population_size * elite_frac)))

    history = []
    best_weights = mean.copy()
    best_score_overall = -1.0

    games_per_opponent = max(1, games_per_candidate // len(opponent_factories))

    for gen in range(generations):
        candidates = rng.normal(loc=mean, scale=std, size=(population_size, NUM_FEATURES))
        scores = np.zeros(population_size)

        for i, weights in enumerate(candidates):
            candidate_factory = make_heuristic_agent_factory(weights)
            wins, games = 0, 0
            for opp_factory in opponent_factories:
                result = arena.play_match(
                    candidate_factory, opp_factory, n_games=games_per_opponent,
                    num_players=num_players, max_turns=max_turns,
                    seed=py_seed_stream.randrange(2**31),
                )
                wins += result["wins_a"]
                games += result["n_games"]
            scores[i] = wins / games if games else 0.0

        elite_idx = np.argsort(scores)[-n_elite:]
        elites = candidates[elite_idx]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), min_std)

        gen_best = float(scores.max())
        if gen_best > best_score_overall:
            best_score_overall = gen_best
            best_weights = candidates[int(np.argmax(scores))].copy()

        history.append({
            "mean": mean.copy(), "std": std.copy(),
            "best_score": gen_best, "mean_score": float(scores.mean()),
        })
        print(f"CEM gen {gen}: best={gen_best:.3f} mean={scores.mean():.3f} "
              f"weights={np.round(mean, 2).tolist()}")

    return best_weights, history
