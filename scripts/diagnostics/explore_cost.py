"""Cost of the self-play acting temperature: a net exploring at (tau_start -> tau_end over 15
plies, no Dirichlet) vs its own greedy depth-1 self, in duplicate pairs (docs/V2_PLAN.md §10.1).

Usage: explore_cost.py <model.pt> <tau_end> <n_pairs> [tau_start=0.03]
"""
import sys
import time

import numpy as np
from _common import setup

setup()
from parchis.az import champion_pool, search, targets
from parchis.az.agent import NetEvaluator, make_search_agent_factory
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation import duplicate, stats

model, tau_end, n_pairs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
tau_start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
ev = NetEvaluator(champion_pool.load_numpy_net(model, 216, 2, (256, 256)))
rng = np.random.default_rng(1)


def explore_factory(game, seat, roll_box):
    tracker = TurnContextTracker()
    ply = {"n": 0}

    def choose_move(legal_moves):
        if not legal_moves:
            tracker.record_move(game, None)
            return None
        roll, pb, cs = tracker.context_for(roll_box)
        _, mv, _ = search.search(game, roll=roll, pending_bonus=pb, consecutive_sixes=cs, depth=1, evaluator=ev)
        tau = targets.anneal_temperature(ply["n"], tau_start=tau_start, tau_end=tau_end, anneal_plies=15)
        ply["n"] += 1
        probs, legal = targets.dirichlet_mixed_probs(mv, seat, tau, rng, epsilon=0.0)
        pid = targets.sample_move_piece_id(probs, legal, rng)
        m = next(x for x in legal_moves if x[0].piece_id == pid)
        tracker.record_move(game, m)
        return m

    return choose_move


t = time.time()
r = duplicate.play_duplicate_match(explore_factory, make_search_agent_factory(ev, 1), n_pairs=n_pairs,
                                   num_players=2, max_turns=500, seed=7)
lo, hi = stats.wilson_score_interval(r["wins_a"], r["n_games"])
print(f"tau {tau_start}->{tau_end}: exploring vs greedy self: {r['wins_a']}/{r['n_games']} = "
      f"{r['wins_a'] / r['n_games']:.3f} [{lo:.3f}, {hi:.3f}] pairs={r.get('pair_record')} ({time.time() - t:.0f}s)")
