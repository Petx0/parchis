"""Champion-promotion protocol (docs/EVAL_PROTOCOL.md, PILA-2): constants, seed-block allocation,
the verdict rule and the report table. Pure logic only -- the games are played by
scripts/eval/protocol.py, which imports this module.

Everything that decides a promotion lives here so it can be unit-tested and changed in one place
(a change to the rule needs the owner's approval, see docs/EVAL_PROTOCOL.md §7).
"""
import json
from pathlib import Path

from parchis.evaluation import stats

# --- Fixed parameters (docs/EVAL_PROTOCOL.md §2-§4) -------------------------------------------
NUM_PLAYERS = 2
MAX_TURNS = 500
DEFAULT_DEPTH = 1
WORKERS = 8                    # one chunk per worker; chunk c of match m uses seed block_base + m.offset + c

DECISIVE_PAIRS = 10_000        # candidate vs champion: 20,000 games
LADDER_PAIRS = 1_000           # every other match: 2,000 games

PROMOTION_LOWER_BOUND = 0.50   # decisive Wilson 95% lower bound must be strictly above this
HEURISTIC_MAX_DROP = 0.03      # candidate vs heuristic may trail the champion vs heuristic by <= 3pp
RANDOM_MIN_RATE = 0.95         # candidate vs random sanity floor
MAX_DRAW_SHARE = 0.001         # more turn-limit draws than this makes the run invalid

# --- Seed blocks (docs/EVAL_PROTOCOL.md §5) ---------------------------------------------------
# Block k owns seeds [BLOCK_BASE + BLOCK_SIZE*k, BLOCK_BASE + BLOCK_SIZE*(k+1)). Below the first
# training base (10_000_000) and above every legacy evaluation range (max 239_007).
BLOCK_BASE = 5_000_000
BLOCK_SIZE = 1_000
MAX_BLOCKS = 5_000             # 5_000_000 + 1_000*5_000 == 10_000_000, the first training base

# (match name, seed offset inside the block, duplicate pairs). The two heuristic matches share
# their seeds on purpose: the non-regression guard compares candidate and champion on the same deals.
MATCHES = [
    ("decisive", 0, DECISIVE_PAIRS),      # candidate vs champion
    ("heuristic", 100, LADDER_PAIRS),     # candidate vs tuned heuristic
    ("champion_heuristic", 100, LADDER_PAIRS),  # champion vs tuned heuristic, SAME seeds (paired guard)
    ("random", 200, LADDER_PAIRS),        # candidate vs random
    ("previous", 300, LADDER_PAIRS),      # candidate vs previous champion (context, not gating)
]

HOME_REGISTRY = Path.home() / "parchis_eval_seeds.jsonl"   # authoritative on the one Mac


def block_seeds(block, offset, workers=WORKERS):
    if not 0 <= block < MAX_BLOCKS:
        raise ValueError(f"block {block} outside [0, {MAX_BLOCKS})")
    if offset < 0 or offset + workers > BLOCK_SIZE:
        raise ValueError(f"offset {offset} + {workers} workers does not fit a {BLOCK_SIZE}-seed block")
    base = BLOCK_BASE + BLOCK_SIZE * block + offset
    return [base + c for c in range(workers)]


def read_registry(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Registry entry kinds that own a block: protocol runs and non-promotion studies (e.g. the PILA-6
# depth study) draw from the same block sequence, so no two evaluations ever share deals.
BLOCK_KINDS = ("protocol", "study")


def next_block(*registries):
    """Smallest block above every block recorded in any of the registries (append-only: a block
    is burned once allocated, even if the run is aborted)."""
    used = [e["block"] for reg in registries for e in reg if e.get("kind") in BLOCK_KINDS]
    k = max(used) + 1 if used else 0
    if k >= MAX_BLOCKS:
        raise RuntimeError("evaluation seed blocks exhausted; extend the registry (EVAL_PROTOCOL §5)")
    return k


def append_registry(paths, entry):
    line = json.dumps(entry, sort_keys=True) + "\n"
    for p in paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(line)


def summarise(wins, draws, n_games):
    """Score = wins + draws/2 (turn-limit draws). Wilson 95% on the score."""
    score = wins + 0.5 * draws
    lo, hi = stats.wilson_score_interval(score, n_games)
    return dict(wins=int(wins), draws=int(draws), n_games=int(n_games), score=float(score),
                win_rate=float(score / n_games), ci=[float(lo), float(hi)])


def verdict(results, selection="fixed", confirmation_of=None):
    """results: match name -> summarise() dict. Returns dict(verdict, checks).

    verdict is one of PASS, FAIL, PROVISIONAL (best-of-N pick that passed but still needs the
    confirmation run on a fresh block) or INVALID (too many draws -- rerun on a new block)."""
    d, h, ch, r = (results[k] for k in ("decisive", "heuristic", "champion_heuristic", "random"))
    checks = [
        dict(name="decisive: Wilson 95% lower bound vs champion > 50%",
             value=d["ci"][0], threshold=PROMOTION_LOWER_BOUND, unit="%", ok=d["ci"][0] > PROMOTION_LOWER_BOUND),
        dict(name="guard: vs heuristic >= champion vs heuristic - 3pp (same seeds)",
             value=h["win_rate"] - ch["win_rate"], threshold=-HEURISTIC_MAX_DROP, unit="pp",
             ok=h["win_rate"] - ch["win_rate"] >= -HEURISTIC_MAX_DROP),
        dict(name="guard: vs random >= 95%",
             value=r["win_rate"], threshold=RANDOM_MIN_RATE, unit="%", ok=r["win_rate"] >= RANDOM_MIN_RATE),
    ]
    for c in checks:
        c["ok"], c["value"] = bool(c["ok"]), float(c["value"])
    draw_share = sum(m["draws"] for m in results.values()) / sum(m["n_games"] for m in results.values())
    if draw_share > MAX_DRAW_SHARE:
        v = "INVALID"
    elif not all(c["ok"] for c in checks):
        v = "FAIL"
    elif selection != "fixed" and confirmation_of is None:
        v = "PROVISIONAL"
    else:
        v = "PASS"
    return dict(verdict=v, checks=checks, draw_share=float(draw_share))


def format_table(results, labels, seeds, depth):
    rows = ["| Match | Win rate | Wilson 95% CI | Games | Draws | Seeds | Depth |",
            "|---|---|---|---|---|---|---|"]
    for name, m in results.items():
        s = seeds[name]
        rows.append(f"| {labels[name]} | {100 * m['win_rate']:.2f}% | "
                    f"[{100 * m['ci'][0]:.2f}, {100 * m['ci'][1]:.2f}] | {m['n_games']:,} | "
                    f"{m['draws']} | {s[0]}..{s[-1]} | {depth} |")
    return "\n".join(rows)


def format_checks(v):
    lines = []
    for c in v["checks"]:
        sign = "+" if c["unit"] == "pp" else ""
        lines.append(f"- [{'x' if c['ok'] else ' '}] {c['name']}: {100 * c['value']:{sign}.2f}{c['unit']}"
                     f" (threshold {100 * c['threshold']:{sign}.2f}{c['unit']})")
    lines.append(f"- draw share {100 * v['draw_share']:.3f}% (max {100 * MAX_DRAW_SHARE:.1f}%)")
    return "\n".join(lines)
