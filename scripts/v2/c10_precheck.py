"""C10 pre-check (docs/V2_PLAN.md §4 C10, revised 2026-11-02).

1. Generate 3 rounds' worth of V2.1c-recipe self-play with the V2.1c champion fixed (24 shards of 750
   games) plus a fixed validation set (3 shards), seeds 500000+i, opponent pool = champion + heuristic.
2. Fit FRESH nets of both sizes (2x256, 3x512) from random weights on 8, 16 and 24 training shards,
   identical hyperparameters, and compare held-out losses. The 24-shard fits are the warm starts for
   arms A and B.

Usage: c10_precheck.py <data_dir> <out_dir>     (data ~7 GB: use a local non-iCloud directory)
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch

INIT = REPO / "runs/v2_1c_champion/model.pt"
SIZES = {"2x256": (256, 256), "3x512": (512, 512, 512)}
N_TRAIN, N_VAL, SEED0 = 24, 3, 500000
FIT = dict(learning_rate=1e-3, weight_decay=1e-4, batch_size=4096, max_epochs=40, patience=4,
           value_loss_weight=2.0, aux_loss_weight=0.2, seed=0, log_every=1)


def generate(data_dir):
    from parchis.az import config as C, encoding, round_loop
    from parchis.az.net import AZNet, NumpyAZNet
    cfg = C.SelfPlayRoundConfig(
        run_name="c10_data", num_players=2, n_games_per_round=6000, games_per_shard=750,
        generation_workers=8, tau_start=0.03, tau_end=0.005, dirichlet_epsilon=0.0, tau_target=0.015,
        enable_escalation=False, aux_loss_weight=0.2, gate_enabled=False, learning_rate=3e-4,
        pool_random_anchor=False, pool_heuristic_fraction=1 / 15,
        deep_fraction=0.25, deep_depth=2, deep_targets=True, afterstate_rows=True,
    )
    model = AZNet(encoding.encoding_size(2), 2, hidden_sizes=cfg.hidden_sizes)
    model.load_state_dict_compat(torch.load(INIT, map_location="cpu"))
    model.eval()
    net = NumpyAZNet.from_torch(model)
    names = [f"train_{i:03d}" for i in range(N_TRAIN)] + [f"val_{i:03d}" for i in range(N_VAL)]
    jobs = [(750, SEED0 + i, str(data_dir / f"{n}.npz")) for i, n in enumerate(names)
            if not (data_dir / f"{n}.npz").exists()]
    t0 = time.time()
    gen_kwargs = round_loop._generation_kwargs(cfg, 1, 0.0)
    for (_, seed, path), st in zip(jobs, round_loop._generate_shards(cfg, net, [], [], jobs, gen_kwargs)):
        print(f"gen {Path(path).name}: {st['n_games']} games, {st['n_recorded_decisions']} decisions, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


def fit_all(data_dir, out_dir):
    from parchis.az import train
    val = [str(data_dir / f"val_{i:03d}.npz") for i in range(N_VAL)]
    res_path = out_dir / "precheck.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for n in (24, 8, 16):  # 24 first: those fits are the arm starts
        tr = [str(data_dir / f"train_{i:03d}.npz") for i in range(n)]
        for name, hs in SIZES.items():
            key = f"{name}_{n}shards"
            if key in results:
                continue
            t0 = time.time()
            model, hist = train.bootstrap_train_sharded(tr, val, num_players=2, hidden_sizes=hs, **FIT)
            best = min(hist, key=lambda e: e["val_loss"])
            results[key] = dict(size=name, n_train_shards=n, minutes=(time.time() - t0) / 60,
                                n_params=sum(p.numel() for p in model.parameters()),
                                best=best, history=hist)
            print(f"FIT {key}: best epoch {best.get('epoch')} val_loss={best['val_loss']:.4f} "
                  f"value={best['val_value_loss']:.4f} policy={best['val_policy_loss']:.4f} "
                  f"train_loss={best.get('train_loss')} ({results[key]['minutes']:.1f} min)", flush=True)
            if n == 24:
                torch.save(model.state_dict(), out_dir / f"start_{name}.pt")
            res_path.write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    generate(data_dir)
    fit_all(data_dir, out_dir)
    print("PRECHECK DONE", flush=True)
