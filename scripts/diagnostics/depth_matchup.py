"""Champion @depth DEEP vs champion @depth 1, duplicate pairs pooled over 8 worker processes."""
import time
from _common import setup, CHAMPION

setup()
DEEP = 2
PAIRS_PER_WORKER = 50


def worker(args):
    seed, n_pairs = args
    from _common import setup as _setup
    _setup()
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.evaluation import duplicate
    ev = NetEvaluator(champion_pool.load_numpy_net(CHAMPION, 216, 2, (256, 256)))
    res = duplicate.play_duplicate_match(make_search_agent_factory(ev, depth=DEEP), make_search_agent_factory(ev, depth=1),
                                         n_pairs=n_pairs, num_players=2, max_turns=500, seed=seed)
    return res['wins_a'], res['n_games']


if __name__ == '__main__':
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from parchis.evaluation import stats as es
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('spawn')) as ex:
        results = list(ex.map(worker, [(500 + i, PAIRS_PER_WORKER) for i in range(8)]))
    wins = sum(r[0] for r in results); n = sum(r[1] for r in results)
    lo, hi = es.wilson_score_interval(wins, n)
    print(f"depth{DEEP} vs depth1 (same champion net): wins={wins}/{n} win_rate={wins/n:.3f} Wilson95=[{lo:.3f}, {hi:.3f}] ({time.perf_counter()-t:.0f}s)")
