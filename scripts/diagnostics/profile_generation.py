"""cProfile of self-play generation with the champion at depth 1 (2p)."""
import cProfile, io, pstats, time
from _common import load_champion

net = load_champion()
from parchis.az import selfplay

t = time.perf_counter()
pr = cProfile.Profile(); pr.enable()
_examples, stats = selfplay.generate_round_games(net, [], n_games=40, num_players=2, max_turns=500, depth=1, seed=1)
pr.disable()
el = time.perf_counter() - t
print("STATS", stats)
print(f"ELAPSED {el:.1f}s  games/s={stats['n_games']/el:.2f}  recorded decisions/s={stats['n_recorded_decisions']/el:.0f}")
for key in ("cumulative", "tottime"):
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(25); print(s.getvalue())
