#!/usr/bin/env python3
"""
Phase A benchmark (go/no-go gate, per the plan): how expensive is a real
MCTS decision at various simulation budgets? Reports deepcopy cost in
isolation, then wall-clock per search() call at a few realistic budgets,
so the real training-loop cost (Phase C, if it gets there) can be
estimated honestly instead of assumed.

Usage: python -m parchis.search.benchmark_mcts
"""

import copy
import time

from parchis.game.game import Game
from parchis.search import mcts
from parchis.search.heuristic_eval import make_heuristic_evaluate_fn


def benchmark_deepcopy(n=200):
    game = Game(num_players=2)
    start = time.perf_counter()
    for _ in range(n):
        copy.deepcopy(game)
    elapsed = time.perf_counter() - start
    per_copy_ms = (elapsed / n) * 1000
    print(f"deepcopy(Game): {per_copy_ms:.3f} ms/copy ({n} copies in {elapsed:.2f}s)")
    return per_copy_ms


def benchmark_search(n_simulations_list, n_decisions=5):
    game = Game(num_players=2)
    player = game.players[0]
    evaluate_fn = make_heuristic_evaluate_fn()

    for n_sims in n_simulations_list:
        times = []
        for i in range(n_decisions):
            legal_moves = game.get_legal_moves(player, 3)
            if not legal_moves:
                legal_moves = game.get_legal_moves(player, 5)
            if not legal_moves:
                continue
            start = time.perf_counter()
            mcts.search(game, agent_seat=0, legal_moves=legal_moves, dice_roll=3,
                        n_simulations=n_sims, evaluate_fn=evaluate_fn, rng_seed=i)
            times.append(time.perf_counter() - start)
        if times:
            avg_ms = (sum(times) / len(times)) * 1000
            sims_per_sec = n_sims / (avg_ms / 1000)
            print(f"n_simulations={n_sims:>4}: {avg_ms:8.1f} ms/decision "
                  f"({sims_per_sec:8.0f} sims/sec)")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase A benchmark: MCTS cost")
    print("=" * 60)
    benchmark_deepcopy()
    print()
    benchmark_search([10, 25, 50, 100, 200])
    print()
    print("For context: a real self-play game averages ~80-100 agent")
    print("decisions (this session's PPO runs' own ep_len_mean). At N")
    print("simulations/decision, one self-play game costs roughly")
    print("N x 90 x (ms/decision at that N) -- extrapolate below.")
