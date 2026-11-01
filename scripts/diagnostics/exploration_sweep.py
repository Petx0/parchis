"""Self-play ACTING policy (softmax(move values / tau) mixed with Dirichlet(alpha=1, eps)) vs the greedy
champion, for several (tau, eps) settings. Reports how often the played move equals the greedy move and the
duplicate-match win rate of the exploring side. The project's configured settings are tau in [1.0, 0.25],
eps=0.25; uniform random play is included as the reference floor."""
import numpy as np
from _common import load_champion

net = load_champion()
from parchis.az import search, selfplay, targets
from parchis.az.agent import NetEvaluator, make_search_agent_factory
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation import duplicate

ev = NetEvaluator(net)
greedy = make_search_agent_factory(ev, depth=1)
N_PAIRS = 120


def make_exploring(tau, eps):
    agree = []

    def factory(game, seat, roll_box):
        tracker = TurnContextTracker(); rng = np.random.default_rng(seat + 1000)

        def choose_move(legal_moves):
            if not legal_moves:
                tracker.record_move(game, None); return None
            roll, pb, cs = tracker.context_for(roll_box)
            best, mv, _rv = search.search(game, roll=roll, pending_bonus=pb, consecutive_sixes=cs, depth=1, evaluator=ev)
            probs, legal_ids = targets.dirichlet_mixed_probs(mv, seat, tau, rng, epsilon=eps)
            pid = targets.sample_move_piece_id(probs, legal_ids, rng)
            if len(legal_ids) >= 2:
                agree.append(int(pid == best[0].piece_id))
            move = next(m for m in legal_moves if m[0].piece_id == pid)
            tracker.record_move(game, move); return move
        return choose_move
    return factory, agree


res = duplicate.play_duplicate_match(selfplay.random_factory, greedy, n_pairs=N_PAIRS, num_players=2, max_turns=500, seed=5)
print(f"uniform random vs greedy: win_rate={res['win_rate_a']:.3f} CI=[{res['win_rate_a_ci'][0]:.3f}, {res['win_rate_a_ci'][1]:.3f}]", flush=True)
for tau, eps in ((0.25, 0.25), (0.05, 0.0), (0.03, 0.10), (0.02, 0.0), (0.01, 0.0)):
    f, agree = make_exploring(tau, eps)
    res = duplicate.play_duplicate_match(f, greedy, n_pairs=N_PAIRS, num_players=2, max_turns=500, seed=5)
    print(f"tau={tau:<5} dirichlet_eps={eps:<5}: played==greedy {np.mean(agree):.3f} | exploring vs greedy win_rate={res['win_rate_a']:.3f} "
          f"CI=[{res['win_rate_a_ci'][0]:.3f}, {res['win_rate_a_ci'][1]:.3f}]", flush=True)
