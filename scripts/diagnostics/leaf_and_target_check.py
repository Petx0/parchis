"""Three measurements on the champion, sampled from champion-vs-champion games:
 (1) leaf-state mismatch: V(pre-roll state) vs mean over the 6 rolls of V(post-roll state), same position/seat
     (search leaves are pre-roll states; training examples are post-roll states);
 (2) maximisation bias: depth-1 root value minus depth-2 root value, mover's channel;
 (3) policy-target sharpness at the configured tau_target=0.25 vs smaller temperatures."""
import random
import numpy as np
from _common import load_champion

net = load_champion()
from parchis.az import encoding, search, targets
from parchis.az.agent import NetEvaluator
from parchis.az.net import value_probs
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation import arena

ev = NetEvaluator(net)
rec = {k: [] for k in ('pre', 'post_mean', 'post_by_roll', 'd1', 'd2', 'maxp', 'gap', 'nlegal', 'ent')}
counter = {'n': 0}
SUB = 4


def own_value(game, seat, roll):
    _p, v = net.forward(encoding.encode(game, seat, roll=roll)[None, :])
    return float(value_probs(v)[0, 0])


def factory(game, seat, roll_box):
    tracker = TurnContextTracker()

    def choose_move(legal_moves):
        if not legal_moves:
            tracker.record_move(game, None); return None
        roll, pending_bonus, cs = tracker.context_for(roll_box)
        move, mv, rv = search.search(game, roll=roll, pending_bonus=pending_bonus, consecutive_sixes=cs, depth=1, evaluator=ev)
        counter['n'] += 1
        if pending_bonus is None and counter['n'] % SUB == 0:
            pre = own_value(game, seat, None)
            posts = [own_value(game, seat, r) for r in range(1, 7)]
            rec['pre'].append(pre); rec['post_mean'].append(np.mean(posts)); rec['post_by_roll'].append(posts)
            _m2, _mv2, rv2 = search.search(game, roll=roll, pending_bonus=None, consecutive_sixes=cs, depth=2, evaluator=ev)
            rec['d1'].append(rv[seat]); rec['d2'].append(rv2[seat])
            if len(mv) >= 2:
                pt = targets.policy_target_from_move_values(mv, seat, tau_target=0.25)
                vals = sorted([v[seat] for v in mv.values()], reverse=True)
                rec['maxp'].append(pt.max()); rec['gap'].append(vals[0] - vals[1]); rec['nlegal'].append(len(mv))
                p = pt[pt > 0]; rec['ent'].append(float(-(p * np.log(p)).sum() / np.log(len(mv))))
        tracker.record_move(game, move)
        return move
    return choose_move


rng = random.Random(7)
for _ in range(40):
    arena.play_one_game({0: factory, 1: factory}, num_players=2, max_turns=500, seed=rng.randrange(2**31))

pre, pm = np.array(rec['pre']), np.array(rec['post_mean']); d1, d2 = np.array(rec['d1']), np.array(rec['d2'])
print(f"n decisions measured: {len(pre)} (of {counter['n']})")
print(f"(1) V(pre-roll) - mean_r V(post-roll): mean={np.mean(pre-pm):+.4f} mean|.|={np.mean(np.abs(pre-pm)):.4f} p90|.|={np.percentile(np.abs(pre-pm),90):.4f} corr={np.corrcoef(pre,pm)[0,1]:.3f}")
pbr = np.array(rec['post_by_roll']); print(f"    spread across rolls of V(post-roll): mean(max-min)={np.mean(pbr.max(1)-pbr.min(1)):.4f}")
print(f"(2) depth1 root - depth2 root (mover): mean={np.mean(d1-d2):+.4f} mean|.|={np.mean(np.abs(d1-d2)):.4f} p90|.|={np.percentile(np.abs(d1-d2),90):.4f}")
gap, nl, maxp, ent = (np.array(rec[k]) for k in ('gap', 'nlegal', 'maxp', 'ent'))
print(f"(3) decisions with >=2 legal moves: {len(gap)}; mean legal={nl.mean():.2f}")
print(f"    best-vs-second value gap: median={np.median(gap):.4f} mean={gap.mean():.4f} p90={np.percentile(gap,90):.4f}")
print(f"    policy target max prob (tau=0.25): median={np.median(maxp):.3f}; uniform would be {np.mean(1/nl):.3f}; normalised entropy median={np.median(ent):.3f}")
for tau in (0.25, 0.05, 0.02, 0.01):
    print(f"    tau={tau}: two-best-moves implied P(best) mean={np.mean(1/(1+np.exp(-gap/tau))):.3f}")
