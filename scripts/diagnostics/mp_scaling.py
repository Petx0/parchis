"""Aggregate self-play generation throughput with N worker processes (spawn), champion at depth 1."""
import time
from _common import setup, CHAMPION

setup()


def worker(args):
    seed, n_games = args
    from _common import setup as _setup
    _setup()
    from parchis.az import champion_pool, selfplay
    net = champion_pool.load_numpy_net(CHAMPION, 216, 2, (256, 256))
    t = time.perf_counter()
    _ex, stats = selfplay.generate_round_games(net, [], n_games=n_games, num_players=2, max_turns=500, depth=1, seed=seed)
    return stats['n_games'], time.perf_counter() - t


if __name__ == '__main__':
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = mp.get_context('spawn')
    for n_workers in (1, 4, 8, 10):
        n_games = 24
        t = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            results = list(ex.map(worker, [(100 + i, n_games) for i in range(n_workers)]))
        el = time.perf_counter() - t
        total = sum(r[0] for r in results)
        print(f"workers={n_workers:2d}: {total} games in {el:.1f}s wall (per-worker gen {min(r[1] for r in results):.1f}-{max(r[1] for r in results):.1f}s) "
              f"-> {total/el:.1f} games/s aggregate (wall), {total/max(r[1] for r in results):.1f} games/s steady-state", flush=True)
