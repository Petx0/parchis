#!/usr/bin/env python3
"""
Pool a completed elo_ladder.py run's cross-group pairings into one
group-A-vs-group-B win rate + Wilson CI.

Introduced for docs/RL_DESIGN_REVIEW.md Phase 5: a round-robin Elo ladder
among e.g. 3 "baseline" checkpoints and 3 "redesigned" checkpoints produces
individual ratings and per-pairing win rates, but doesn't itself answer
"is redesigned better than baseline" as one number -- that requires pooling
every baseline-vs-redesigned pairing's raw wins/games together. This is a
pure function (mirrors parchis/evaluation/stats.py / elo.py's style) plus a
thin CLI that reads elo_ladder.py's results.json.

Usage:
    python -m parchis.evaluation.group_comparison \\
        --results-json ./logs/phase5_elo_ladder/results.json \\
        --group-a baseline_42 baseline_43 baseline_44 \\
        --group-b redesigned_42 redesigned_43 redesigned_44
"""

import json
import argparse

from parchis.evaluation import stats as eval_stats


def aggregate_group_win_rate(pairings, group_a_names, group_b_names):
    """
    Pool every cross-group pairing's wins/games into a single
    group_a-vs-group_b win rate (from group_a's perspective) + Wilson CI.

    Args:
        pairings: list of dicts shaped like elo_ladder.PairingResult (or
            the loaded results.json's 'pairings' list) -- each needs
            'participant_a', 'participant_b', 'games', 'wins_a'.
        group_a_names: participant names forming group A.
        group_b_names: participant names forming group B.

    Only pairings with one participant in each group count -- within-group
    pairings (e.g. two baseline seeds against each other) and pairings
    involving a participant in neither group (e.g. the random baseline)
    are ignored.

    Returns:
        dict: {'wins': int, 'games': int, 'win_rate': float,
        'ci': (lower, upper), 'n_pairings': int} -- group_a's aggregate
        record against group_b.

    Raises:
        ValueError: if group_a_names/group_b_names overlap, or if no
        cross-group pairing was found at all (e.g. a name typo).
    """
    group_a_set = set(group_a_names)
    group_b_set = set(group_b_names)
    overlap = group_a_set & group_b_set
    if overlap:
        raise ValueError(f"group_a and group_b must not overlap, but both contain: {sorted(overlap)}")

    wins = 0
    games = 0
    n_pairings = 0

    for pairing in pairings:
        a, b = pairing['participant_a'], pairing['participant_b']
        if a in group_a_set and b in group_b_set:
            wins += pairing['wins_a']
            games += pairing['games']
            n_pairings += 1
        elif a in group_b_set and b in group_a_set:
            # Same cross-group pairing, roles swapped: group_a's wins are
            # whatever group_b's participant (here "a") did *not* win.
            wins += pairing['games'] - pairing['wins_a']
            games += pairing['games']
            n_pairings += 1
        # else: within-group, or involves a participant in neither group
        # (e.g. the random baseline) -- not a cross-group comparison.

    if n_pairings == 0:
        raise ValueError(
            "No cross-group pairing found between group_a and group_b -- "
            "check the participant names match elo_ladder's --checkpoint-names "
            "(or auto-derived basenames) exactly."
        )

    return {
        'wins': wins,
        'games': games,
        'win_rate': wins / games,
        'ci': eval_stats.wilson_score_interval(wins, games),
        'n_pairings': n_pairings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pool a completed elo_ladder.py run's cross-group pairings "
                     "into one group-A-vs-group-B win rate + Wilson CI"
    )
    parser.add_argument("--results-json", type=str, required=True,
                         help="Path to elo_ladder.py's results.json")
    parser.add_argument("--group-a", type=str, nargs="+", required=True,
                         help="Participant names forming group A (e.g. baseline checkpoints)")
    parser.add_argument("--group-b", type=str, nargs="+", required=True,
                         help="Participant names forming group B (e.g. redesigned checkpoints)")

    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    result = aggregate_group_win_rate(data['pairings'], args.group_a, args.group_b)

    print("=" * 60)
    print("GROUP COMPARISON")
    print("=" * 60)
    print(f"  Group A: {args.group_a}")
    print(f"  Group B: {args.group_b}")
    print(f"  Cross-group pairings pooled: {result['n_pairings']}")
    print(f"  Group A record: {result['wins']}/{result['games']} ({result['win_rate']:.1%})")
    print(f"  95% CI: [{result['ci'][0]:.1%}, {result['ci'][1]:.1%}]")
    print("=" * 60)

    ratings = data.get('ratings', {})
    if ratings:
        print("\nElo ratings (for context):")
        for name in sorted(ratings, key=ratings.get, reverse=True):
            group = "A" if name in args.group_a else ("B" if name in args.group_b else "-")
            print(f"  [{group}] {name:<30} {ratings[name]:8.1f}")

    return result


if __name__ == "__main__":
    main()
