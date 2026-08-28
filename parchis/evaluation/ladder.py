#!/usr/bin/env python3
"""
AlphaZero-agent-compatible fixed benchmark ladder
(docs/AGENT_REBUILD_PLAN.md §5.2) -- round-robin duplicate-match
comparisons across a fixed set of "rungs" (random / heuristic / frozen
az-r{N} checkpoints), appended append-only to `runs/pairings.jsonl` so
parchis.evaluation.ratings can fit Bradley-Terry ratings over the whole
project's history. This is the mechanism "2p clears the ladder"
(docs/AGENT_REBUILD_PLAN.md's Phase 4 gate) actually refers to.

NOT built on parchis.evaluation.elo_ladder / multiplayer_matrix: those wrap
evaluate_agent() (parchis/evaluation/evaluate.py), which is MaskablePPO-
checkpoint-only (calls .load() directly) -- a search-driven AZ agent needs
live search access at inference time, not just saved weights (the same
reason parchis/evaluation/arena.py exists as its own parallel tool, per
that module's docstring). Uses parchis.evaluation.duplicate.play_duplicate_match
instead, which already generalizes to num_players > 2 (rotating the tested
agent through every seat on one shared dice seed) and is already
CRN-variance-reduced.

Rungs are named "NAME=SPEC" strings, SPEC in the same grammar
parchis.visualization.play_instrumented_game's --agent flag uses (see
parchis.agents.agent_spec) -- "checkpoint:<run_dir>[:depth=N]" |
"heuristic:tuned|default" | "random" -- with an explicit friendly NAME so a
rung's identity in pairings.jsonl stays stable even if the underlying
checkpoint path changes later (e.g. a run directory gets renamed/archived).

pairings.jsonl is project-wide and append-only (matching §5.2's spec
exactly) -- NOT a per-run artifact like runs/<name>/metrics.jsonl. Running
the ladder again (with the same or different rungs) only ever adds more
lines; nothing already on disk is rewritten.

Usage:
    python -m parchis.evaluation.ladder \\
        --rung random=random \\
        --rung heuristic_default=heuristic:default \\
        --rung heuristic_tuned=heuristic:tuned \\
        --rung az_champion=checkpoint:runs/selfplay_2p_v1_champion:depth=1 \\
        --num-players 2 --pairs 100
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from parchis.agents import agent_spec
from parchis.evaluation import duplicate
from parchis.evaluation.elo import round_robin_pairings

DEFAULT_PAIRINGS_PATH = "runs/pairings.jsonl"
DEFAULT_NUM_PLAYERS = 2
DEFAULT_N_PAIRS = 100
DEFAULT_MAX_TURNS = 500


def _parse_rung_spec(rung_str):
    """'NAME=SPEC' -> (name, factory). SPEC parsing itself is
    agent_spec.parse_spec -- see that module for the grammar."""
    if '=' not in rung_str:
        raise ValueError(f"--rung must be NAME=SPEC, got {rung_str!r} (missing '=')")
    name, spec = rung_str.split('=', 1)
    kind, params, _label = agent_spec.parse_spec(spec)
    return name, agent_spec.build_factory(kind, params)


def run_ladder(rung_factories, num_players=DEFAULT_NUM_PLAYERS, n_pairs=DEFAULT_N_PAIRS,
               max_turns=DEFAULT_MAX_TURNS, seed=42, pairings_path=DEFAULT_PAIRINGS_PATH,
               verbose=1):
    """
    Round-robins every unordered pair of `rung_factories` via
    duplicate.play_duplicate_match, appending one JSON line per pairing to
    `pairings_path` (created, including parent directories, if missing).

    Args:
        rung_factories: {name: arena-style factory(game, seat, roll_box)
            -> choose_move_fn}, at least 2 entries.
        num_players: passed to play_duplicate_match (2-4).
        n_pairs: duplicate-pairs per pairing (n_pairs * num_players games).
        seed: seeds both the pairing-order RNG and each pairing's own seed
            stream (random.Random(seed), matching elo.round_robin_pairings'
            own "never the bare random module" convention).
        pairings_path: append target -- see module docstring for why this
            is a project-wide file, not a per-run one.
        verbose: 0 = silent, 1 = one line per pairing as it completes.

    Returns:
        list of pairing-result dicts, in the exact shape appended to
        pairings_path (so callers/tests can inspect what was just written
        without re-reading the file).
    """
    names = list(rung_factories)
    if len(names) < 2:
        raise ValueError(f"Need at least 2 rungs to run a ladder, got {names}")

    rng = random.Random(seed)
    pairings = round_robin_pairings(names, rng)
    results = []

    path = Path(pairings_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("LADDER")
        print("=" * 60)
        print(f"  Rungs: {names}")
        print(f"  Pairings: {len(pairings)}, {n_pairs} duplicate-pairs each "
              f"({n_pairs * num_players} games/pairing)")
        print("=" * 60)

    with open(path, "a") as f:
        for name_a, name_b in pairings:
            result = duplicate.play_duplicate_match(
                rung_factories[name_a], rung_factories[name_b], n_pairs=n_pairs,
                num_players=num_players, max_turns=max_turns,
                seed=rng.randrange(2**31),
            )
            lower, upper = result["win_rate_a_ci"]
            record = {
                "timestamp": datetime.now().isoformat(),
                "participant_a": name_a,
                "participant_b": name_b,
                "num_players": num_players,
                "wins_a": result["wins_a"],
                "n_games": result["n_games"],
                "win_rate_a": result["win_rate_a"],
                "win_rate_a_ci": [lower, upper],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            results.append(record)

            if verbose:
                print(f"  {name_a} vs {name_b}: {record['wins_a']}/{record['n_games']} "
                      f"({record['win_rate_a']:.1%}) CI[{lower:.1%}, {upper:.1%}]")

    if verbose:
        print(f"\nAppended {len(results)} pairings to {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Round-robin duplicate-match ladder for AlphaZero-style agents "
                     "(search checkpoints, heuristic, random).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--rung', action='append', default=[], metavar='NAME=SPEC', required=True,
                         help="Repeatable, at least 2. SPEC: checkpoint:<run_dir>[:depth=N] | "
                              "heuristic:tuned|default | random.")
    parser.add_argument('--num-players', type=int, default=DEFAULT_NUM_PLAYERS)
    parser.add_argument('--pairs', type=int, default=DEFAULT_N_PAIRS,
                         help="Duplicate-pairs per pairing (default: %(default)s)")
    parser.add_argument('--max-turns', type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pairings-path', default=DEFAULT_PAIRINGS_PATH,
                         help="Append target (default: %(default)s)")
    args = parser.parse_args()

    try:
        rung_factories = dict(_parse_rung_spec(r) for r in args.rung)
    except ValueError as e:
        parser.error(str(e))
        return

    if len(rung_factories) != len(args.rung):
        parser.error("--rung NAMEs must be unique")

    run_ladder(
        rung_factories, num_players=args.num_players, n_pairs=args.pairs,
        max_turns=args.max_turns, seed=args.seed, pairings_path=args.pairings_path,
    )


if __name__ == "__main__":
    main()
