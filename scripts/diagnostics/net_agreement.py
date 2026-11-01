"""Move agreement between nets at depth-1 search, on greedy games driven by one of them
(docs/V2_PLAN.md §10.1: successive V2.0 champions agree on ~2/3 of multi-move decisions).

Usage: net_agreement.py <driver_name> <n_games> name=model.pt [name=model.pt ...]

Also reports plies/decisions per game, truncations, no-legal-move plies, the driver's best-vs-
second gap, and how often the driver's POLICY head argmax matches its own depth-1 search.
"""
import sys
import time

import numpy as np
from _common import setup

setup()
from parchis.az import champion_pool, encoding, search
from parchis.az.agent import NetEvaluator
from parchis.az.net import masked_policy_probs
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation import arena

driver, n_games = sys.argv[1], int(sys.argv[2])
paths = dict(a.split("=", 1) for a in sys.argv[3:])
nets = {k: NetEvaluator(champion_pool.load_numpy_net(p, 216, 2, (256, 256))) for k, p in paths.items()}
names = list(paths)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
S = {"plies": 0, "no_move": 0, "multi": 0, "policy_agree": 0, "decisions": 0, "gap_sum": 0.0, "multi_gap3": 0}
for a, b in pairs:
    S[f"agree_{a}_{b}"] = 0
    S[f"agree3_{a}_{b}"] = 0


def factory(game, seat, roll_box):
    tracker = TurnContextTracker()

    def choose_move(legal_moves):
        S["plies"] += 1
        if not legal_moves:
            S["no_move"] += 1
            tracker.record_move(game, None)
            return None
        roll, pb, cs = tracker.context_for(roll_box)
        choices, dmove, dvals = {}, None, None
        for name, ev in nets.items():
            mv, move_values, _ = search.search(game, roll=roll, pending_bonus=pb, consecutive_sixes=cs,
                                               depth=1, evaluator=ev)
            choices[name] = mv[0].piece_id
            if name == driver:
                dmove, dvals = mv, move_values
        S["decisions"] += 1
        if len(legal_moves) > 1:
            S["multi"] += 1
            v = sorted((vec[seat] for vec in dvals.values()), reverse=True)
            gap = v[0] - v[1]
            S["gap_sum"] += gap
            big = gap > 0.03
            S["multi_gap3"] += big
            for a, b in pairs:
                ag = choices[a] == choices[b]
                S[f"agree_{a}_{b}"] += ag
                if big:
                    S[f"agree3_{a}_{b}"] += ag
            obs = encoding.encode(game, seat, roll=roll, pending_bonus=pb, consecutive_sixes=cs)
            pl, _ = nets[driver].numpy_net.forward(obs[None, :])
            mask = np.zeros(4)
            for m in legal_moves:
                mask[m[0].piece_id] = 1
            S["policy_agree"] += int(np.argmax(masked_policy_probs(pl[0], mask)) == choices[driver])
        tracker.record_move(game, dmove)
        return dmove

    return choose_move


t = time.time()
wins, trunc = [0, 0], 0
for g in range(n_games):
    w = arena.play_one_game({0: factory, 1: factory}, num_players=2, max_turns=500, seed=100000 + g)
    if w is None:
        trunc += 1
    else:
        wins[w] += 1
print(f"driver={driver} games={n_games} time={time.time() - t:.0f}s truncated={trunc} wins_by_seat={wins}")
print(f"plies/game={S['plies'] / n_games:.1f} decisions/game={S['decisions'] / n_games:.1f} "
      f"no_move_plies={S['no_move'] / S['plies']:.1%} multi_move_decisions={S['multi'] / S['decisions']:.1%}")
print(f"driver best-vs-second gap mean={100 * S['gap_sum'] / S['multi']:.2f}pp; decisions with gap>3pp={S['multi_gap3'] / S['multi']:.1%}")
for a, b in pairs:
    print(f"agree {a} vs {b}: all multi-move {S[f'agree_{a}_{b}'] / S['multi']:.1%} | gap>3pp {S[f'agree3_{a}_{b}'] / max(S['multi_gap3'], 1):.1%}")
print(f"policy-head argmax == depth-1 search ({driver}): {S['policy_agree'] / S['multi']:.1%}")
