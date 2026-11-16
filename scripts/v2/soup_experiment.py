"""PILA-8: weight soups of existing checkpoints -- build, health-check, screen (one heavy job).

Usage: soup_experiment.py <out_dir under runs/> --base <10_000_000*n> [--scratch DIR] [--issue PILA-8]
                          [--workers 8] [--pairs 2000] [--health-games 1500]

1. Builds the four soups of the issue into <out_dir>/s1..s4/model.pt (+ config.json), uniform weights.
2. Health check: generates --health-games fresh V2.1c-recipe self-play games with the V2.1c
   champion (8 shards on 8 workers; seeds base + i, i < 8), then the value log-loss of every soup
   and ingredient on them. A soup whose value loss is worse than its worst ingredient's by more than
   0.01 (soup.HEALTH_MARGIN) is dropped before any match.
3. Screening: every surviving soup vs V2.1c, --pairs duplicate pairs at depth 1, 8 workers, on the
   SAME deals for every soup (seeds base + 1_000_000 + c, c < 8: the run's own training band, never
   the 5M-10M evaluation band, docs/EVAL_PROTOCOL.md §5). This only ranks the soups. The winner is
   a best-of-N pick, and the verdict is the Evaluator's protocol run (--selection best-of-N).
4. Writes <out_dir>/health_check.json, <out_dir>/screening.json and a run record (meta.json) per
   soup, with the screening result as an `h2h` eval tagged pila8_screening and `selection` on each.

Takes ~/parchis_compute.lock for the whole job (8 workers); exits 3 if it is held by a live PID.
Shards (~0.6 GB) go to --scratch (default ~/parchis_scratch/pila8_health), outside the repo.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/v2"))
from ladder import single_thread_blas  # noqa: E402

LOCK = Path.home() / "parchis_compute.lock"
CHAMPION = REPO / "runs/v2_1c_champion/model.pt"
INGREDIENTS = {
    "v2_1c": "runs/v2_1c_champion/model.pt",
    "v2_1d": "runs/v2_1d_champion/model.pt",
    "win3": "runs/window_test/win3/model.pt",
    "win10": "runs/window_test/win10/model.pt",
}
SOUPS = {
    "s1": ("v2_1c", "win3", "win10"),
    "s2": ("win3", "win10"),
    "s3": ("v2_1c", "v2_1d"),
    "s4": ("v2_1c", "v2_1d", "win3", "win10"),
}
DEPTH = 1
MAX_TURNS = 500
SCREEN_OFFSET = 1_000_000
ETA_MINUTES = 15
TRAINING_BASE_UNIT = 10_000_000


def acquire_lock(issue, what, eta_minutes=ETA_MINUTES):
    if LOCK.exists():
        try:
            held = json.loads(LOCK.read_text())
            os.kill(int(held["pid"]), 0)
            return held
        except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
            print(f"stale compute lock overwritten: {LOCK.read_text().strip()}", flush=True)
        except PermissionError:
            return json.loads(LOCK.read_text())
    now = datetime.now(timezone.utc)
    LOCK.write_text(json.dumps(dict(issue=issue, agent="Engineer", pid=os.getpid(),
                                    started=now.isoformat(timespec="seconds"),
                                    eta=(now + timedelta(minutes=eta_minutes)).isoformat(timespec="seconds"),
                                    what=what)))
    return None


def release_lock():
    try:
        if json.loads(LOCK.read_text()).get("pid") == os.getpid():
            LOCK.unlink()
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def recipe_config(base, run_name="pila8_health"):
    """The V2.1c recipe (scripts/v2/run_v2_1c.py), seeds on the allocated training base."""
    from parchis.az import config as C
    from parchis.registry import record as registry
    cfg = C.SelfPlayRoundConfig(
        run_name=run_name, num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8, tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False, promotion_n_pairs=100,
        learning_rate=3e-4, pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.25, deep_depth=2, deep_targets=True, afterstate_rows=True,
    )
    return registry.apply_seed_base(cfg, base)


def generate_health_shards(cfg, base, n_games, workers, scratch, generator=CHAMPION):
    """n_games of V2.1c-recipe self-play by `generator` (default the V2.1c champion; pool = generator
    + tuned heuristic, as round 1 of a run from it), split over `workers` shards with seeds base + i."""
    import torch
    from parchis.az import encoding, round_loop
    from parchis.az.net import AZNet, NumpyAZNet
    model = AZNet(encoding.encoding_size(2), 2, hidden_sizes=cfg.hidden_sizes)
    model.load_state_dict_compat(torch.load(generator, map_location="cpu"))
    model.eval()
    net = NumpyAZNet.from_torch(model)
    sizes = [n_games // workers + (i < n_games % workers) for i in range(workers)]
    jobs = [(n, base + i, str(scratch / f"health_{i:02d}.npz")) for i, n in enumerate(sizes)]
    cfg.generation_workers = workers
    gen_kwargs = round_loop._generation_kwargs(cfg, 1, 0.0)
    stats = list(round_loop._generate_shards(cfg, net, [], [], jobs, gen_kwargs))
    return [j[2] for j in jobs], [dict(n_games=j[0], seed=j[1], path=j[2], n_recorded_decisions=st.get("n_recorded_decisions"))
                                  for j, st in zip(jobs, stats)]


def _factory(path):
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator, make_search_agent_factory
    from parchis.az.soup import hidden_sizes_from_state_dict
    import torch
    hs = hidden_sizes_from_state_dict(torch.load(path, map_location="cpu"))
    return make_search_agent_factory(NetEvaluator(champion_pool.load_numpy_net(path, 216, 2, hs)), depth=DEPTH)


def _match_job(args):
    path_a, path_b, seed, n_pairs = args
    sys.path.insert(0, str(REPO))
    from parchis.evaluation import duplicate
    r = duplicate.play_duplicate_match(_factory(path_a), _factory(path_b), n_pairs=n_pairs,
                                       num_players=2, max_turns=MAX_TURNS, seed=seed)
    draws = sum(w is None for g in r["groups"] for w in g["winners"])
    return r["wins_a"], draws, r["n_games"]


def screen(path, seeds, pairs, opponent=CHAMPION):
    from parchis.evaluation import stats
    per = pairs // len(seeds)
    with single_thread_blas(), ProcessPoolExecutor(len(seeds), mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(_match_job, [(str(path), str(opponent), s, per) for s in seeds]))
    wins, draws, n = (sum(r[i] for r in res) for i in range(3))
    score = wins + 0.5 * draws
    lo, hi = stats.wilson_score_interval(score, n)
    return dict(wins=wins, draws=draws, n_games=n, win_rate=score / n, ci=[lo, hi])


def write_record(soup_dir, name, cfg_soup, base, seed_entry, health, screening, selection, wall_s, issue):
    from parchis.registry import record as registry
    ckpt = soup_dir / "model.pt"
    config = dict(
        method="uniform weight average (model soup), no gradient steps",
        ingredients=cfg_soup["ingredients"],
        generation_seed=base, train_seed=base, promotion_seed=base,
        seed_use=("generation_seed: health-check shards base+i; screening deals base+1_000_000+c; "
                  "no training, so train_seed/promotion_seed are unused"),
        health_check_recipe="V2.1c recipe (scripts/v2/run_v2_1c.py), generator = V2.1c champion",
    )
    run_id = registry.rel(soup_dir)
    run_id = run_id[len("runs/"):] if run_id.startswith("runs/") else run_id
    meta = registry.new_meta(run_id, "training", config=config, issue=issue,
                             title=f"PILA-8 weight soup {name.upper()} = avg({', '.join(SOUPS[name])})",
                             init=None, script=registry.rel(__file__), command=" ".join(sys.argv),
                             notes=("Weight soup (SWA / model soups) of existing checkpoints; no training. "
                                    "Health check and screening in ../health_check.json and ../screening.json. "
                                    "Screening is a best-of-N ranking, not a verdict."))
    meta["init"] = dict(note="weight average of the ingredients in config.ingredients", path=None, sha256=None,
                        run_id=None)
    meta["seeds"]["registered_to"] = [seed_entry["run_id"]]
    meta["training"] = dict(rounds_completed=0, selfplay_games=0, wall_clock_s=round(wall_s, 1),
                            note="no training: weights averaged; wall-clock is the whole PILA-8 job")
    meta["health_check"] = health
    meta["selection"] = selection
    meta["status"] = "finished"
    meta["finished"] = registry.now()
    meta["sessions"] = [dict(started=meta["created"], git_sha=meta["git"]["sha"], git_dirty=meta["git"]["dirty"],
                             command=meta["command"], pid=os.getpid())]
    registry.set_checkpoint(meta, registry.checkpoint_entry(ckpt, role="final", label=name))
    if screening is not None:
        registry.add_eval(meta, registry.eval_entry(
            "h2h", ckpt, CHAMPION, screening["win_rate"], screening["ci"], screening["n_games"],
            seeds=screening["seeds"], depth=DEPTH, source=soup_dir.parent / "screening.json",
            tag="pila8_screening", draws=screening["draws"], wall_clock_s=screening["wall_clock_s"],
            note="screening only (ranks the soups); not a verdict"))
    registry.save_meta(soup_dir, meta)
    problems = registry.validate(meta)
    print(f"record {run_id}: " + ("OK" if not problems else "INCOMPLETE " + "; ".join(problems)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--base", type=int, required=True)
    ap.add_argument("--issue", default="PILA-8")
    ap.add_argument("--scratch", default=str(Path.home() / "parchis_scratch/pila8_health"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--health-games", type=int, default=1500)
    a = ap.parse_args()
    if a.base < TRAINING_BASE_UNIT or a.base % TRAINING_BASE_UNIT:
        print(f"--base must be a positive multiple of {TRAINING_BASE_UNIT}", file=sys.stderr)
        return 4
    from parchis.registry import record as registry
    seed_entry = next((e for r in (registry.HOME_SEED_REGISTRY, registry.REPO_SEED_REGISTRY)
                       for e in registry._read_jsonl(r) if e.get("kind") == "training" and e.get("base") == a.base), None)
    if seed_entry is None:
        print(f"base {a.base} is not allocated in the seed registry (python -m parchis.registry alloc-seed)",
              file=sys.stderr)
        return 4
    out = Path(a.out_dir).resolve()
    scratch = Path(a.scratch).expanduser()
    scratch.mkdir(parents=True, exist_ok=True)
    held = acquire_lock(a.issue, f"PILA-8 soups: health-check shards + screening {len(SOUPS)}x{a.pairs} pairs")
    if held:
        print(f"compute lock held: {held}", flush=True)
        return 3
    t0 = time.time()
    try:
        from parchis.az import soup
        # 1. soups
        cfgs = {}
        for name, parts in SOUPS.items():
            cfgs[name] = soup.make_soup(out / name / "model.pt", [REPO / INGREDIENTS[p] for p in parts], name=name,
                                        extra=dict(issue=a.issue, ingredient_names=list(parts)))
            print(f"soup {name} = avg({', '.join(parts)}) sha256 {cfgs[name]['output']['sha256'][:12]}", flush=True)
        # 2. health check
        cfg = recipe_config(a.base)
        tg = time.time()
        shard_paths, shard_info = generate_health_shards(cfg, a.base, a.health_games, a.workers, scratch)
        gen_s = time.time() - tg
        print(f"health shards: {sum(s['n_games'] for s in shard_info)} games in {gen_s:.0f}s", flush=True)
        import torch
        data = soup.load_shards(shard_paths)
        losses = {}
        for key, path in list(INGREDIENTS.items()) + [(n, out / n / "model.pt") for n in SOUPS]:
            losses[key] = soup.shard_losses(torch.load(REPO / path if not Path(path).is_absolute() else path,
                                                       map_location="cpu"), data)
            print(f"  {key:6s} value {losses[key]['value_loss']:.5f}  value(decisions) "
                  f"{losses[key]['value_loss_decisions']:.5f}  policy {losses[key]['policy_loss']:.5f}", flush=True)
        health = {}
        for name, parts in SOUPS.items():
            ing = [losses[p]["value_loss"] for p in parts]
            keep = soup.health_verdict(losses[name]["value_loss"], ing)
            health[name] = dict(value_loss=losses[name]["value_loss"], worst_ingredient=max(ing),
                                best_ingredient=min(ing), margin=soup.HEALTH_MARGIN, keep=keep,
                                delta_vs_worst=losses[name]["value_loss"] - max(ing),
                                delta_vs_best=losses[name]["value_loss"] - min(ing))
            print(f"health {name}: Δ vs worst {health[name]['delta_vs_worst']:+.5f}, "
                  f"Δ vs best {health[name]['delta_vs_best']:+.5f} -> {'keep' if keep else 'DROP'}", flush=True)
        (out / "health_check.json").write_text(json.dumps(dict(
            issue=a.issue, recipe="V2.1c (scripts/v2/run_v2_1c.py)", generator=registry.rel(CHAMPION),
            generator_sha256=registry.file_sha256(CHAMPION), games=sum(s["n_games"] for s in shard_info),
            shards=[dict(s, path=registry.rel(s["path"])) for s in shard_info],
            seeds=f"{a.base}+i, i<{len(shard_info)}", generation_s=round(gen_s, 1),
            rows=losses["v2_1c"]["n_rows"], decision_rows=losses["v2_1c"]["n_decision_rows"],
            losses=losses, verdicts=health,
            caveat="data generated by V2.1c with V2.1c depth-2 deep targets: biased towards V2.1c's own value head"),
            indent=2) + "\n")
        # 3. screening
        seeds = [a.base + SCREEN_OFFSET + c for c in range(a.workers)]
        screening = {}
        for name in SOUPS:
            if not health[name]["keep"]:
                continue
            ts = time.time()
            r = screen(out / name / "model.pt", seeds, a.pairs)
            r.update(seeds=f"{seeds[0]}+c, c<{len(seeds)} (same deals for every soup)", depth=DEPTH,
                     pairs=a.pairs, wall_clock_s=round(time.time() - ts, 1), opponent=registry.rel(CHAMPION))
            screening[name] = r
            print(f"screen {name} vs V2.1c: {r['win_rate']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] "
                  f"n={r['n_games']} draws={r['draws']} ({r['wall_clock_s']:.0f}s)", flush=True)
        ranked = sorted(screening, key=lambda k: -screening[k]["win_rate"])
        winner = ranked[0] if ranked and screening[ranked[0]]["win_rate"] > 0.5 else None
        (out / "screening.json").write_text(json.dumps(dict(
            issue=a.issue, opponent=registry.rel(CHAMPION), depth=DEPTH, pairs=a.pairs,
            seeds=[seeds[0], seeds[-1]], results=screening, dropped_by_health=[n for n in SOUPS if not health[n]["keep"]],
            ranking=ranked, winner=winner,
            kill_criterion="every soup below 50% vs V2.1c -> stop after screening, record the null",
            note="ranking only; the winner is best-of-N and needs the protocol (--selection best-of-N) + confirmation"),
            indent=2) + "\n")
        # 4. run records
        wall = time.time() - t0
        n_screened = len(screening)
        for name in SOUPS:
            if name == winner:
                sel = dict(method=f"best-of-{n_screened}", chosen="highest screening score vs V2.1c",
                           candidates=[f"{registry.rel(out / k)}" for k in ranked], n=n_screened,
                           source=registry.rel(out / "screening.json"),
                           note="screening seeds in the run's training band; protocol needs --selection "
                                f"best-of-{n_screened} and a confirmation run")
            elif not health[name]["keep"]:
                sel = dict(method="dropped", reason="health check: value loss > worst ingredient + 0.01")
            else:
                sel = dict(method="not selected", rank=ranked.index(name) + 1, of=n_screened,
                           source=registry.rel(out / "screening.json"),
                           reason=None if winner else "kill criterion: no soup above 50% in screening")
            write_record(out / name, name, cfgs[name], a.base, seed_entry, health[name], screening.get(name), sel,
                         wall, a.issue)
        print(f"winner: {winner or 'none (kill criterion met)'}   ranking {ranked}   wall-clock {wall:.0f}s", flush=True)
    finally:
        release_lock()
    print("SOUP EXPERIMENT DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
