#!/usr/bin/env python3
"""
Win-rate matrix for 3-4 player checkpoint comparisons, where Elo has no
valid interpretation (see parchis/evaluation/elo_ladder.py's own module
docstring: ParchisSelfPlayEnv puts the SAME model in every non-agent seat,
so a match between checkpoints A and B is "A vs (N-1)xB," not a clean
pairwise comparison a transitive rating could be built from).

This tool doesn't attempt a rating either -- it reports, for each pair of
checkpoints (A, B), two independent numbers: A's win rate as the lone
tracked agent against (N-1) clones of B, and B's win rate as the lone
tracked agent against (N-1) clones of A. These are NOT complementary the
way a 2-player "A vs B" match is (unlike elo_ladder.py's _play_pairing,
which can infer one side from the other) -- they're two different opponent
compositions, so both are actually played.

Random-baseline participant: only ONE direction is computable per pair --
"checkpoint alone vs (N-1)x random" (evaluate_agent's agent slot always
needs a real model; there's no stand-in random-policy model here to let the
TRACKED seat itself play randomly). "random alone vs (N-1)xcheckpoint" is
skipped for that pair, not silently approximated -- matches Gap 1 / the
hyperparameter search's own "win rate vs random" convention for what a
single checkpoint's random baseline means.

Usage:
    python -m parchis.evaluation.multiplayer_matrix \\
        --checkpoints ./models/a/final_model ./models/b/final_model \\
        --checkpoint-names a b \\
        --players 4 --games-per-pairing 40
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


@dataclass
class MatrixEntry:
    """One direction of one pair: `agent`'s win rate as the lone tracked
    seat against (num_players - 1) clones of `opponent`."""
    agent: str
    opponent: str
    num_players: int
    games: int
    wins: int
    win_rate: float
    win_rate_ci: Tuple[float, float]


def _play_direction(agent_name, opponent_name, checkpoint_paths, num_players, n_games):
    """Play n_games with agent_name as the lone tracked seat against
    (num_players - 1) clones of opponent_name (or random opponents, if
    opponent_name is RANDOM_PARTICIPANT). Returns (wins, win_rate)."""
    opponent_path = None if opponent_name == RANDOM_PARTICIPANT else checkpoint_paths[opponent_name]
    stats = evaluate_agent(
        agent_model_path=checkpoint_paths[agent_name],
        opponent_model_path=opponent_path,
        n_games=n_games,
        num_players=num_players,
        verbose=0,
    )
    return stats['wins'], stats['win_rate']


def _save_results_json(entries, save_path):
    """Save results to a JSON file (called after each entry for crash recovery)."""
    os.makedirs(save_path, exist_ok=True)
    results_file = os.path.join(save_path, "results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "entries": [asdict(e) for e in entries],
        }, f, indent=2)


def _print_matrix(entries):
    """Print a formatted matrix of all directions played."""
    print("\n" + "=" * 70)
    print("WIN-RATE MATRIX RESULTS")
    print("=" * 70)
    for e in entries:
        print(f"  {e.agent} alone vs {e.num_players - 1}x{e.opponent}: "
              f"{e.wins}/{e.games} ({e.win_rate:.1%}) "
              f"CI[{e.win_rate_ci[0]:.1%}, {e.win_rate_ci[1]:.1%}]")
    print("=" * 70)


def run_matrix(
    checkpoint_paths: Dict[str, str],
    num_players: int = 4,
    include_random_baseline: bool = True,
    games_per_pairing: int = 40,
    seed: int = 42,
    save_path: Optional[str] = None,
    verbose: int = 1,
) -> List[MatrixEntry]:
    """
    Round-robin pairwise win-rate matrix for num_players > 2, where Elo has
    no valid interpretation (see module docstring).

    Args:
        checkpoint_paths: {friendly_name: path} for every real checkpoint
            to compare (no .zip extension, matching MaskablePPO.load()'s
            convention elsewhere in this codebase).
        num_players: 3 or 4 -- for 2-player comparisons use elo_ladder.py
            instead, which has a valid Elo interpretation there.
        include_random_baseline: Add a RANDOM_PARTICIPANT pseudo-participant
            (checkpoint-vs-random direction only, see module docstring).
        games_per_pairing: Games played per direction (not per pair -- each
            pair of real checkpoints plays this many games in EACH direction).
        seed: Seeds the pairing-order RNG (random.Random(seed), never the
            bare `random` module -- see elo.round_robin_pairings).
        save_path: Directory to write results.json into (skipped if None).
        verbose: 0 = silent, 1 = per-direction lines + final matrix.

    Returns:
        List of MatrixEntry -- two per pair of real checkpoints (one per
        direction), one per (checkpoint, random) pair.
    """
    if num_players not in (3, 4):
        raise ValueError(
            f"num_players must be 3 or 4 -- use elo_ladder.py for 2-player "
            f"comparisons, which have a valid Elo interpretation there. Got {num_players}"
        )

    participants = list(checkpoint_paths.keys())
    if include_random_baseline:
        participants.append(RANDOM_PARTICIPANT)
    if len(participants) < 2:
        raise ValueError(
            "Need at least 2 participants (checkpoints, optionally plus the "
            "random baseline) to run a win-rate matrix"
        )

    pairings = elo.round_robin_pairings(participants, random.Random(seed))
    entries: List[MatrixEntry] = []

    if verbose > 0:
        print("=" * 70)
        print("WIN-RATE MATRIX")
        print("=" * 70)
        print(f"  Participants: {participants}")
        print(f"  Players per game: {num_players}")
        print(f"  Games per direction: {games_per_pairing}")
        print("=" * 70)

    for name_a, name_b in pairings:
        for agent_name, opponent_name in ((name_a, name_b), (name_b, name_a)):
            if agent_name == RANDOM_PARTICIPANT:
                # No stand-in random-policy model for evaluate_agent's agent
                # slot -- see module docstring. Skip, don't approximate.
                continue

            wins, win_rate = _play_direction(
                agent_name, opponent_name, checkpoint_paths, num_players, games_per_pairing,
            )
            ci = eval_stats.wilson_score_interval(wins, games_per_pairing)
            entry = MatrixEntry(
                agent=agent_name, opponent=opponent_name, num_players=num_players,
                games=games_per_pairing, wins=wins, win_rate=win_rate, win_rate_ci=ci,
            )
            entries.append(entry)

            if verbose > 0:
                print(f"  {agent_name} alone vs {num_players - 1}x{opponent_name}: "
                      f"{wins}/{games_per_pairing} ({win_rate:.1%}) "
                      f"CI[{ci[0]:.1%}, {ci[1]:.1%}]")

            if save_path:
                _save_results_json(entries, save_path)

    if verbose > 0:
        _print_matrix(entries)

    if save_path:
        _save_results_json(entries, save_path)
        if verbose > 0:
            print(f"\nResults saved to: {os.path.join(save_path, 'results.json')}")

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise win-rate matrix for saved Parchis checkpoints at 3-4 players "
                     "(candidate alone vs (N-1)x reference, both directions per pair of "
                     "real checkpoints)"
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                         help="Paths to saved checkpoints to compare (without .zip extension)")
    parser.add_argument("--checkpoint-names", type=str, nargs="+", default=None,
                         help="Friendly names for --checkpoints, same order/length "
                              "(default: basenames of --checkpoints)")
    parser.add_argument("--players", type=int, default=4, choices=[3, 4],
                         help="Number of players per game (default: 4)")
    parser.add_argument("--no-random-baseline", dest="include_random_baseline",
                         action="store_false",
                         help="Exclude the random-opponent baseline participant "
                              "(default: included)")
    parser.add_argument("--games-per-pairing", type=int, default=40,
                         help="Games played per direction (default: 40)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for pairing order (default: 42)")
    parser.add_argument("--save-path", type=str, default=None,
                         help="Directory to save results.json "
                              "(default: ./logs/multiplayer_matrix/<timestamp>/)")
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
        save_path = os.path.join("./logs/multiplayer_matrix", timestamp)

    run_matrix(
        checkpoint_paths=checkpoint_paths,
        num_players=args.players,
        include_random_baseline=args.include_random_baseline,
        games_per_pairing=args.games_per_pairing,
        seed=args.seed,
        save_path=save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
