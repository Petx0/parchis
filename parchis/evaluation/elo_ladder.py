#!/usr/bin/env python3
"""
Elo ladder: round-robin evaluate a set of saved checkpoints (+ an optional
random-opponent baseline) against each other, so "is checkpoint N+1 actually
stronger" is answerable directly instead of inferred from a noisy,
moving-target self-play win rate (see docs/RL_DESIGN_REVIEW.md Phase 4).

2-player matches only: ParchisSelfPlayEnv puts the *same* opponent model in
every non-agent seat for num_players > 2 (parchis/rl/env_selfplay.py), so a
3-4 player match between checkpoints A and B is actually "A vs 3xB," not a
clean pairwise comparison -- there's no valid Elo interpretation of that
without a much larger multiplayer-Elo redesign, which this ladder's
"lightweight" scope (docs/RL_DESIGN_REVIEW.md) doesn't call for.

Each pairing reuses evaluate_agent() (parchis/evaluation/evaluate.py) for
its games -- this gets the hang-safety timeout (docs/EVALUATION_FIX.md) for
free, and means a pairing's whole games_per_pairing block is treated as one
Elo-update observation (see parchis/evaluation/elo.py's update_ratings
docstring for why that's an acceptable simplification here). Because
ParchisEnv.agent_player_idx is already randomized per-episode
(docs/CODE_REVIEW.md), a pairing's games already split roughly evenly
across both seats regardless of which checkpoint is passed as
evaluate_agent's agent vs. opponent -- no need to manually alternate
"who's home" for fairness.

Checkpoints are given explicitly via --checkpoints rather than
auto-discovered from a directory: checkpoint naming has no single
convention across this codebase's training scripts (checkpoint_<N>_steps,
opponent_checkpoint_<n>_<steps>steps, bare final_model/interrupted_model,
alpha_<v>_<weighting>_<timestamp>, <arch>_<reward_type> with no shared
prefix at all) -- auto-detecting "the right" checkpoints across that would
be real, error-prone new complexity for no clear benefit.

Usage:
    python -m parchis.evaluation.elo_ladder \\
        --checkpoints ./models/run_a/checkpoint_50000_steps ./models/run_a/checkpoint_100000_steps \\
        --games-per-pairing 40
"""

import os
import json
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from parchis.evaluation.evaluate import evaluate_agent
from parchis.evaluation import elo
from parchis.evaluation import stats as eval_stats

RANDOM_PARTICIPANT = "random"

# Scope decision (see module docstring): matches are always 2-player.
LADDER_NUM_PLAYERS = 2


@dataclass
class PairingResult:
    """One pairing's outcome and the Elo update it produced."""
    participant_a: str
    participant_b: str
    games: int
    wins_a: int
    win_rate_a: float
    win_rate_a_ci: Tuple[float, float]
    rating_a_before: float
    rating_b_before: float
    rating_a_after: float
    rating_b_after: float


def _play_pairing(name_a, name_b, checkpoint_paths, n_games, verbose):
    """
    Play n_games games between two participants (either may be
    RANDOM_PARTICIPANT, never both -- round_robin_pairings never pairs a
    participant with itself) and return (wins_a, n_games, win_rate_a) from
    participant A's perspective.

    evaluate_agent() always needs a real checkpoint as its agent_model_path
    -- it has no "the agent is random" option, only "the opponent is
    random" (opponent_model_path=None). If A is the random baseline, the
    pairing is run with B as evaluate_agent's agent instead and the result
    is inverted.
    """
    if name_a != RANDOM_PARTICIPANT:
        opponent_path = None if name_b == RANDOM_PARTICIPANT else checkpoint_paths[name_b]
        stats = evaluate_agent(
            agent_model_path=checkpoint_paths[name_a],
            opponent_model_path=opponent_path,
            n_games=n_games,
            num_players=LADDER_NUM_PLAYERS,
            verbose=0,
        )
        return stats['wins'], n_games, stats['win_rate']

    # name_a is the random baseline; name_b must be a real checkpoint.
    stats = evaluate_agent(
        agent_model_path=checkpoint_paths[name_b],
        opponent_model_path=None,
        n_games=n_games,
        num_players=LADDER_NUM_PLAYERS,
        verbose=0,
    )
    wins_a = n_games - stats['wins']
    return wins_a, n_games, wins_a / n_games


def _save_results_json(pairing_results, ratings, save_path):
    """Save results to a JSON file (called after each pairing for crash recovery)."""
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "ratings": ratings,
            "pairings": [asdict(p) for p in pairing_results],
        }, f, indent=2)


def _print_ratings_table(ratings, participants):
    """Print a formatted, rank-sorted ratings table."""
    print("\n" + "=" * 60)
    print("ELO LADDER RESULTS")
    print("=" * 60)
    ranked = sorted(participants, key=lambda name: ratings[name], reverse=True)
    for rank, name in enumerate(ranked, start=1):
        print(f"  {rank}. {name:<40} {ratings[name]:8.1f}")
    print("=" * 60)


def run_ladder(
    checkpoint_paths: Dict[str, str],
    include_random_baseline: bool = True,
    games_per_pairing: int = 40,
    k_factor: float = elo.DEFAULT_K_FACTOR,
    initial_rating: float = elo.DEFAULT_INITIAL_RATING,
    seed: int = 42,
    save_path: Optional[str] = None,
    verbose: int = 1,
) -> Tuple[Dict[str, float], List[PairingResult]]:
    """
    Run a full round-robin Elo ladder over checkpoint_paths (+ an optional
    random-opponent baseline participant).

    Args:
        checkpoint_paths: {friendly_name: path} for every real checkpoint
            to compare (no .zip extension, matching MaskablePPO.load()'s
            convention elsewhere in this codebase).
        include_random_baseline: Add a RANDOM_PARTICIPANT pseudo-participant.
        games_per_pairing: Games played per unordered pair.
        k_factor: Elo K-factor (see parchis/evaluation/elo.py).
        initial_rating: Starting rating for every participant.
        seed: Seeds the pairing-order RNG (random.Random(seed), never the
            bare `random` module -- see elo.round_robin_pairings).
        save_path: Directory to write results.json into (skipped if None).
        verbose: 0 = silent, 1 = per-pairing lines + final ratings table.

    Returns:
        tuple(dict {name: final_rating}, list[PairingResult])
    """
    participants = list(checkpoint_paths.keys())
    if include_random_baseline:
        participants.append(RANDOM_PARTICIPANT)
    if len(participants) < 2:
        raise ValueError(
            "Need at least 2 participants (checkpoints, optionally plus the "
            "random baseline) to run a ladder"
        )

    ratings = {name: initial_rating for name in participants}
    pairings = elo.round_robin_pairings(participants, random.Random(seed))
    pairing_results = []

    if verbose > 0:
        print("=" * 60)
        print("ELO LADDER")
        print("=" * 60)
        print(f"  Participants: {participants}")
        print(f"  Games per pairing: {games_per_pairing}")
        print(f"  Total pairings: {len(pairings)}")
        print("=" * 60)

    for name_a, name_b in pairings:
        wins_a, n_games, win_rate_a = _play_pairing(
            name_a, name_b, checkpoint_paths, games_per_pairing, verbose
        )
        ci = eval_stats.wilson_score_interval(wins_a, n_games)

        rating_a_before, rating_b_before = ratings[name_a], ratings[name_b]
        rating_a_after, rating_b_after = elo.update_ratings(
            rating_a_before, rating_b_before, win_rate_a, k_factor=k_factor
        )
        ratings[name_a], ratings[name_b] = rating_a_after, rating_b_after

        pairing_results.append(PairingResult(
            participant_a=name_a, participant_b=name_b, games=n_games,
            wins_a=wins_a, win_rate_a=win_rate_a, win_rate_a_ci=ci,
            rating_a_before=rating_a_before, rating_b_before=rating_b_before,
            rating_a_after=rating_a_after, rating_b_after=rating_b_after,
        ))

        if verbose > 0:
            print(f"  {name_a} vs {name_b}: {wins_a}/{n_games} ({win_rate_a:.1%}) "
                  f"CI[{ci[0]:.1%}, {ci[1]:.1%}]  ->  "
                  f"{name_a}={rating_a_after:.1f}  {name_b}={rating_b_after:.1f}")

        if save_path:
            _save_results_json(pairing_results, ratings, save_path)

    if verbose > 0:
        _print_ratings_table(ratings, participants)

    if save_path:
        _save_results_json(pairing_results, ratings, save_path)
        if verbose > 0:
            print(f"\nResults saved to: {os.path.join(save_path, 'results.json')}")

    return ratings, pairing_results


def main():
    parser = argparse.ArgumentParser(
        description="Round-robin Elo ladder for saved Parchis checkpoints (2-player matches only)"
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                         help="Paths to saved checkpoints to compare (without .zip extension)")
    parser.add_argument("--checkpoint-names", type=str, nargs="+", default=None,
                         help="Friendly names for --checkpoints, same order/length "
                              "(default: basenames of --checkpoints)")
    parser.add_argument("--no-random-baseline", dest="include_random_baseline",
                         action="store_false",
                         help="Exclude the random-opponent baseline participant "
                              "(default: included)")
    parser.add_argument("--games-per-pairing", type=int, default=40,
                         help="Games played per pairing (default: 40)")
    parser.add_argument("--k-factor", type=float, default=elo.DEFAULT_K_FACTOR,
                         help=f"Elo K-factor (default: {elo.DEFAULT_K_FACTOR})")
    parser.add_argument("--initial-rating", type=float, default=elo.DEFAULT_INITIAL_RATING,
                         help=f"Starting Elo rating for every participant "
                              f"(default: {elo.DEFAULT_INITIAL_RATING})")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for pairing order (default: 42)")
    parser.add_argument("--save-path", type=str, default=None,
                         help="Directory to save results.json "
                              "(default: ./logs/elo_ladder/<timestamp>/)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                         help="Verbosity level (default: 1)")

    args = parser.parse_args()

    if args.checkpoint_names is not None and len(args.checkpoint_names) != len(args.checkpoints):
        parser.error("--checkpoint-names must have the same number of entries as --checkpoints")

    names = args.checkpoint_names or [os.path.basename(p) for p in args.checkpoints]
    if len(set(names)) != len(names):
        parser.error("--checkpoint-names (or auto-derived basenames) must be unique")

    checkpoint_paths = dict(zip(names, args.checkpoints))

    save_path = args.save_path
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("./logs/elo_ladder", timestamp)

    run_ladder(
        checkpoint_paths=checkpoint_paths,
        include_random_baseline=args.include_random_baseline,
        games_per_pairing=args.games_per_pairing,
        k_factor=args.k_factor,
        initial_rating=args.initial_rating,
        seed=args.seed,
        save_path=save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
