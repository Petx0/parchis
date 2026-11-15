"""Leaderboard from a round-robin pairings.jsonl (PILA-1).

Fits Bradley-Terry ratings with parchis.evaluation.ratings.fit_ratings (anchor: random = 0),
reports them on the Elo scale (x 400/ln 10), and gets 95% CIs from a game-level parametric
bootstrap: each replicate redraws every pairing's wins ~ Binomial(n_games, observed rate) and
refits; Elo gaps to the top-rated participant get their own CI from the same replicates.
(ratings.bootstrap_rating_cis resamples whole pairings instead, which for a complete round
robin measures schedule variation rather than game noise, so it is not used here.) Also writes the
full pairwise win-rate matrix with Wilson 95% CIs.

Usage: analyze.py <pairings.jsonl> <out.json> [reps=500] [seed=0]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from parchis.evaluation.ratings import fit_ratings  # noqa: E402

ELO = 400 / math.log(10)


def main():
    positional = [a for a in sys.argv[1:] if "=" not in a]
    extra = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    pairings = [json.loads(l) for l in Path(positional[0]).read_text().splitlines() if l.strip()]
    reps, seed = int(extra.get("reps", 500)), int(extra.get("seed", 0))
    point = fit_ratings(pairings, anchor="random")
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    for _ in range(reps):
        rs = [dict(p, wins_a=int(rng.binomial(p["n_games"], p["wins_a"] / p["n_games"])))
              for p in pairings]
        for k, v in fit_ratings(rs, anchor="random").items():
            samples[k].append(v)
    names = sorted(point, key=lambda k: -point[k])
    # rank stability: how often each participant is the top-rated one across replicates
    tops = {}
    arr = np.array([samples[k] for k in names])
    for i in arr.argmax(axis=0):
        tops[names[i]] = tops.get(names[i], 0) + 1
    rows = []
    for k in names:
        lo, hi = np.percentile(samples[k], [2.5, 97.5])
        games = sum(p["n_games"] for p in pairings if k in (p["participant_a"], p["participant_b"]))
        wins = sum(p["wins_a"] if p["participant_a"] == k else p["n_games"] - p["wins_a"]
                   for p in pairings if k in (p["participant_a"], p["participant_b"]))
        diff = (np.array(samples[k]) - np.array(samples[names[0]])) * ELO
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        rows.append(dict(name=k, elo=point[k] * ELO, elo_ci=[lo * ELO, hi * ELO], games=games,
                         score=wins / games, p_top=tops.get(k, 0) / reps,
                         delta_vs_top=(point[k] - point[names[0]]) * ELO, delta_vs_top_ci=[dlo, dhi]))
    matrix = {}
    for p in pairings:
        a, b, w, n = p["participant_a"], p["participant_b"], p["wins_a"], p["n_games"]
        lo, hi = p["win_rate_a_ci"]
        matrix.setdefault(a, {})[b] = dict(win_rate=w / n, ci=[lo, hi], n=n)
        matrix.setdefault(b, {})[a] = dict(win_rate=1 - w / n, ci=[1 - hi, 1 - lo], n=n)
    out = dict(ratings=rows, matrix=matrix, order=names, bootstrap_reps=reps, bootstrap_seed=seed,
               n_pairings=len(pairings), total_games=sum(p["n_games"] for p in pairings))
    Path(positional[1]).write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['name']:20s} {r['elo']:7.0f} [{r['elo_ci'][0]:6.0f}, {r['elo_ci'][1]:6.0f}] "
              f"d {r['delta_vs_top']:5.0f} [{r['delta_vs_top_ci'][0]:5.0f}, {r['delta_vs_top_ci'][1]:5.0f}] "
              f"score {r['score']:.3f} n={r['games']} P(top)={r['p_top']:.2f}")


if __name__ == "__main__":
    main()
