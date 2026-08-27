"""
Phase 3 self-play target construction (docs/AGENT_REBUILD_PLAN.md Part 3
Phase 3 / §1.6): turns one search.search() call's outputs (move_values,
root_value) plus a game's eventual outcome into the two SOFT training
targets a round actually trains on -- neither is the hard "which move/seat
won" label Phase 2's bootstrap used.

    z_value  = (1 - lambda) * outcome + lambda * root_value      (§1.6: a
        single game's outcome is dominated by dice, not the move being
        labelled -- blending in the search's OWN root value is the
        variance-reduction fix modern practice (KataGo, Stochastic MuZero)
        uses instead of regressing onto raw rollout outcomes.)
    z_policy = masked_softmax({search-assessed value of each legal move},
        temperature=tau_target)                                  (§2.3: a
        softmax over root move values, "a far better target than MCTS
        visit counts at this branching factor")

Both `outcome` and `root_value` (and therefore `z_value`) must already be
in the SAME mover-relative channel order parchis.az.encoding uses (index 0
= this decision's own mover) -- exactly Phase 2's own hard-won convention
(docs/AZ_DESIGN.md's Phase 2 entry). search.py's own contract returns
move_values/root_value in ABSOLUTE seat order (see its module docstring),
so callers here are responsible for the same np.roll(-mover_seat) fix
parchis.az.agent.NetEvaluator and parchis.az.selfplay.generate_games both
already apply -- see selfplay.generate_round_games, which does this
before calling blend_value_target.

The move-SELECTION policy used to actually play out a self-play game
(anneal_temperature + dirichlet_mixed_probs, applied to the exact same
per-move values) is a DIFFERENT, separate distribution from z_policy: the
acting policy needs to explore (high temperature early, Dirichlet noise at
the root), while the training target should reflect the search's genuine,
un-perturbed assessment of move quality. Conflating the two would train
the net to imitate its own exploration noise instead of its own judgment.
"""

import numpy as np

from parchis.az.net import NUM_ACTIONS

DEFAULT_LAMBDA = 0.5  # docs/AGENT_REBUILD_PLAN.md Part 3/4: "lambda ~ 0.5, tuned once"

# Not given a specific number by the plan (only tau's OWN anneal schedule
# is specified numerically) -- chosen to match the "settled" end of that
# same schedule (see anneal_temperature), on the theory that a training
# target should reflect the same sharpness the self-play process itself
# converges to once exploration has cooled down, rather than an
# independently-guessed number.
DEFAULT_TAU_TARGET = 0.25

DEFAULT_TAU_START = 1.0
DEFAULT_TAU_END = 0.25
DEFAULT_ANNEAL_PLIES = 15

# AlphaZero's own published root-noise mixing weight (epsilon=0.25).
# AlphaZero's alpha is picked roughly proportional to 1/(average legal
# moves) (chess: alpha=0.3 at ~35 moves; go: alpha=0.03 at ~250); this
# game's ~2.76 legal moves per decision (Part 1 §1.1) is far smaller than
# either, so alpha=1.0 (a near-uniform Dirichlet over ~3 actions) is the
# proportionally-scaled choice, not a re-used chess/go constant.
DEFAULT_DIRICHLET_ALPHA = 1.0
DEFAULT_DIRICHLET_EPSILON = 0.25


def blend_value_target(outcome, root_value, lam=DEFAULT_LAMBDA):
    """z_value = (1 - lam) * outcome + lam * root_value (§1.6). Both
    arguments must be same-length, mover-relative vectors (see module
    docstring) -- this function is agnostic to that convention, it just
    blends two vectors elementwise; correctness of what's passed in is the
    caller's responsibility (see selfplay.generate_round_games)."""
    outcome = np.asarray(outcome, dtype=np.float64)
    root_value = np.asarray(root_value, dtype=np.float64)
    if outcome.shape != root_value.shape:
        raise ValueError(f"shape mismatch: outcome={outcome.shape} root_value={root_value.shape}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be within [0, 1], got {lam}")
    return (1.0 - lam) * outcome + lam * root_value


def policy_target_from_move_values(move_values, mover_seat, tau_target=DEFAULT_TAU_TARGET,
                                    num_actions=NUM_ACTIONS):
    """z_policy (§2.3): masked softmax, over exactly the legal piece_ids in
    `move_values`, of each move's value TO ITS OWN MOVER
    (move_values[pid][mover_seat] -- move_values itself is in ABSOLUTE seat
    order per search.py's own contract, so indexing by mover_seat is what
    extracts "this move's value to whoever is actually choosing it"), at
    temperature `tau_target`. Illegal piece_ids (not a key of move_values)
    get exactly 0.0, never a tiny nonzero leak -- matches
    parchis.az.net.masked_policy_probs' own convention.

    move_values: {piece_id: np.ndarray[num_players]}, search.search()'s own
    per-move breakdown -- must be non-empty (a decision with no legal move
    has no policy target to build).

    Returns: (num_actions,) float32 array summing to 1.0.
    """
    if not move_values:
        raise ValueError("policy_target_from_move_values requires at least one legal move")
    if tau_target <= 0.0:
        raise ValueError(f"tau_target must be > 0, got {tau_target}")

    values = np.full(num_actions, -np.inf, dtype=np.float64)
    for piece_id, vec in move_values.items():
        values[piece_id] = vec[mover_seat]

    scaled = values / tau_target
    scaled = scaled - np.max(scaled)  # stable softmax; -inf slots stay -inf
    exp = np.where(np.isfinite(scaled), np.exp(scaled), 0.0)
    total = exp.sum()
    return (exp / total).astype(np.float32)


def anneal_temperature(ply, tau_start=DEFAULT_TAU_START, tau_end=DEFAULT_TAU_END,
                        anneal_plies=DEFAULT_ANNEAL_PLIES):
    """Linear anneal from tau_start (ply 0) to tau_end (ply >= anneal_plies)
    -- Part 3's "temperature tau annealed 1.0 -> 0.25 over the first ~15
    plies" for the self-play ACTING policy (see module docstring for why
    this is distinct from tau_target)."""
    if anneal_plies <= 0:
        return tau_end
    frac = min(max(ply, 0) / anneal_plies, 1.0)
    return tau_start + frac * (tau_end - tau_start)


def dirichlet_mixed_probs(move_values, mover_seat, tau, rng, num_actions=NUM_ACTIONS,
                           alpha=DEFAULT_DIRICHLET_ALPHA, epsilon=DEFAULT_DIRICHLET_EPSILON):
    """The self-play ACTING distribution at one root: softmax(move values /
    tau) over the legal piece_ids, mixed with Dirichlet(alpha) noise at
    weight epsilon (AlphaZero's own root-exploration recipe):
    (1-epsilon)*softmax + epsilon*dirichlet_sample, renormalized over the
    SAME legal support (an illegal slot never receives noise mass).

    rng: a numpy.random.Generator (e.g. np.random.default_rng(seed)) --
    threaded through by the caller for reproducibility, not created here.

    Returns:
        tuple(np.ndarray, list[int]): (probs, legal_piece_ids). probs is a
        dense length-num_actions float array (0.0 on illegal slots);
        legal_piece_ids is the sorted list of keys of move_values, for
        sample_move_piece_id to draw from consistently with probs' own
        nonzero support.
    """
    if not move_values:
        raise ValueError("dirichlet_mixed_probs requires at least one legal move")
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    legal_piece_ids = sorted(move_values.keys())
    n_legal = len(legal_piece_ids)

    values = np.array([move_values[pid][mover_seat] for pid in legal_piece_ids], dtype=np.float64)
    scaled = values / tau
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    base_probs = exp / exp.sum()

    noise = rng.dirichlet(np.full(n_legal, alpha)) if epsilon > 0.0 else np.zeros(n_legal)
    mixed = (1.0 - epsilon) * base_probs + epsilon * noise
    mixed = mixed / mixed.sum()  # renormalize away any float drift

    probs = np.zeros(num_actions, dtype=np.float64)
    for pid, p in zip(legal_piece_ids, mixed):
        probs[pid] = p
    return probs, legal_piece_ids


def sample_move_piece_id(probs, legal_piece_ids, rng):
    """Sample one piece_id from `legal_piece_ids`, weighted by `probs`
    (dense, e.g. dirichlet_mixed_probs' output) -- a thin, separately-
    testable wrapper around rng.choice so callers don't have to
    re-normalize probs[legal_piece_ids] inline at every call site.

    rng: a numpy.random.Generator (see dirichlet_mixed_probs)."""
    weights = np.array([probs[pid] for pid in legal_piece_ids], dtype=np.float64)
    weights = weights / weights.sum()
    return int(rng.choice(legal_piece_ids, p=weights))
