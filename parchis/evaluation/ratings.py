#!/usr/bin/env python3
"""
Bradley-Terry maximum-likelihood ratings over
parchis.evaluation.ladder's accumulated `runs/pairings.jsonl`
(docs/AGENT_REBUILD_PLAN.md §5.3).

Replaces parchis.evaluation.elo.update_ratings' order-dependent sequential
K-factor updates with one MLE fit over the WHOLE pairing history at once:
"one number per agent comparable across the whole project history," rather
than a running score whose current value depends on what order pairings
happened to be played in.

Model: P(i beats j) = sigmoid(rating_i - rating_j), one log-strength
`rating` per participant, fit by minimizing the negative log-likelihood of
the observed win counts (scipy.optimize.minimize, L-BFGS-B -- scipy is
already a project dependency, see parchis/evaluation/stats.py).
Bradley-Terry ratings are only identifiable up to an additive constant, so
one participant is ANCHORED at rating 0.0: "random" if present among the
participants (the natural zero point -- "how much better than chance"),
else the alphabetically-first participant (deterministic, not arbitrary).

Usage:
    python -m parchis.evaluation.ratings [--pairings-path runs/pairings.jsonl] [--bootstrap 200]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from parchis.evaluation.ladder import DEFAULT_PAIRINGS_PATH

ANCHOR_PARTICIPANT = "random"
DEFAULT_BOOTSTRAP_REPS = 200
DEFAULT_CONFIDENCE = 0.95


def load_pairings(pairings_path):
    """Returns a list of pairing-record dicts, one per line of
    `pairings_path` (as written by parchis.evaluation.ladder.run_ladder).
    Returns [] if the file doesn't exist yet -- the graceful "no data to
    rate" case, not an error."""
    path = Path(pairings_path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _participants(pairings):
    names = set()
    for p in pairings:
        names.add(p["participant_a"])
        names.add(p["participant_b"])
    return sorted(names)


def _default_anchor(participants):
    return ANCHOR_PARTICIPANT if ANCHOR_PARTICIPANT in participants else participants[0]


def _negative_log_likelihood(free_ratings, anchor_idx, pairings, index_of):
    ratings = np.insert(free_ratings, anchor_idx, 0.0)
    nll = 0.0
    for p in pairings:
        i, j = index_of[p["participant_a"]], index_of[p["participant_b"]]
        wins_a, n = p["wins_a"], p["n_games"]
        diff = ratings[i] - ratings[j]
        # log P(a beats b) = -log(1 + exp(-diff)) = -logaddexp(0, -diff);
        # the numerically stable form (avoids overflow for large |diff|),
        # not the textbook 1/(1+exp(-diff)) formula directly.
        log_p_a = -np.logaddexp(0.0, -diff)
        log_p_b = -np.logaddexp(0.0, diff)
        nll -= wins_a * log_p_a + (n - wins_a) * log_p_b
    return nll


def fit_ratings(pairings, anchor=None):
    """Fits one Bradley-Terry log-strength rating per participant over
    `pairings` (dicts with at least "participant_a", "participant_b",
    "wins_a", "n_games" -- the shape parchis.evaluation.ladder.run_ladder
    appends to pairings.jsonl).

    Args:
        pairings: list of pairing-record dicts.
        anchor: participant name to fix at rating 0.0. None (default)
            picks _default_anchor's choice ("random" if present, else the
            alphabetically-first participant). Passing this explicitly
            (rather than always recomputing it) is what lets
            bootstrap_rating_cis compare every resample against the SAME
            reference point -- see its docstring.

    Returns:
        dict {name: rating}. Raises ValueError if fewer than 2
        participants, or if `anchor` is given but absent from `pairings`.
    """
    participants = _participants(pairings)
    if len(participants) < 2:
        raise ValueError(f"Need at least 2 participants to fit ratings, got {participants}")
    if anchor is None:
        anchor = _default_anchor(participants)
    elif anchor not in participants:
        raise ValueError(f"anchor {anchor!r} not among participants {participants}")

    index_of = {name: i for i, name in enumerate(participants)}
    anchor_idx = index_of[anchor]
    free_x0 = np.zeros(len(participants) - 1)

    result = minimize(
        _negative_log_likelihood, free_x0,
        args=(anchor_idx, pairings, index_of),
        method="L-BFGS-B",
    )
    ratings = np.insert(result.x, anchor_idx, 0.0)
    return {name: float(ratings[i]) for name, i in index_of.items()}


def bootstrap_rating_cis(pairings, n_reps=DEFAULT_BOOTSTRAP_REPS, seed=0,
                          confidence=DEFAULT_CONFIDENCE):
    """Percentile bootstrap CIs for fit_ratings' output: resample
    `pairings` with replacement `n_reps` times, refit each time, take
    percentiles per participant across replicates.

    The anchor is fixed ONCE, from the full (unresampled) `pairings`, and
    passed explicitly into every replicate's fit_ratings call -- letting
    fit_ratings pick its own anchor per-replicate would let it silently
    drift to a different participant whenever a resample happens to drop
    the usual anchor, which would compare ratings against DIFFERENT zero
    points across replicates and contaminate the resulting CIs. A replicate
    that drops the anchor entirely (or collapses to under 2 participants)
    is skipped rather than force-fit against a reference point it doesn't
    actually contain.

    Returns:
        dict {name: (lower, upper)} -- (nan, nan) for any participant with
        fewer than 2 surviving replicates to take percentiles over.
    """
    rng = np.random.default_rng(seed)
    all_participants = _participants(pairings)
    anchor = _default_anchor(all_participants)
    samples = {name: [] for name in all_participants}

    n = len(pairings)
    for _ in range(n_reps):
        resampled = [pairings[i] for i in rng.integers(0, n, size=n)]
        resampled_participants = _participants(resampled)
        if anchor not in resampled_participants or len(resampled_participants) < 2:
            continue
        fitted = fit_ratings(resampled, anchor=anchor)
        for name, rating in fitted.items():
            samples[name].append(rating)

    alpha = (1.0 - confidence) / 2.0
    cis = {}
    for name, values in samples.items():
        if len(values) < 2:
            cis[name] = (float('nan'), float('nan'))
            continue
        cis[name] = (
            float(np.percentile(values, 100 * alpha)),
            float(np.percentile(values, 100 * (1 - alpha))),
        )
    return cis


def main():
    parser = argparse.ArgumentParser(
        description="Fit Bradley-Terry ratings over a ladder's accumulated pairings.jsonl",
    )
    parser.add_argument('--pairings-path', default=DEFAULT_PAIRINGS_PATH)
    parser.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP_REPS,
                         help="Bootstrap replicates for CIs, 0 to skip (default: %(default)s)")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    pairings = load_pairings(args.pairings_path)
    if not pairings:
        print(f"No pairings found at {args.pairings_path} -- run parchis.evaluation.ladder first.")
        return

    ratings = fit_ratings(pairings)
    cis = (bootstrap_rating_cis(pairings, n_reps=args.bootstrap, seed=args.seed)
           if args.bootstrap > 0 else {})

    print("=" * 60)
    print("BRADLEY-TERRY RATINGS")
    print("=" * 60)
    for name, rating in sorted(ratings.items(), key=lambda kv: kv[1], reverse=True):
        if name in cis:
            lower, upper = cis[name]
            print(f"  {name:<30} {rating:8.3f}  [{lower:7.3f}, {upper:7.3f}]")
        else:
            print(f"  {name:<30} {rating:8.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
