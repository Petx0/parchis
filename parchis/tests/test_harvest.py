"""Tests for the PILA-5 harvest driver (scripts/v2/run_harvest.py): soup plan, recipe, candidate
harvesting, soup building and the winner/kill rule. No games are played here; the end-to-end path is
exercised by `run_harvest.py --smoke`."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from parchis.az import encoding
from parchis.az.net import AZNet

REPO = Path(__file__).resolve().parents[2]
IN = encoding.encoding_size(2)


@pytest.fixture(scope="module")
def H():
    sys.path.insert(0, str(REPO / "scripts/v2"))
    spec = importlib.util.spec_from_file_location("run_harvest", REPO / "scripts/v2/run_harvest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(path, seed):
    torch.manual_seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(AZNet(IN, 2, hidden_sizes=(16, 16)).state_dict(), path)
    return path


def test_plan_for_20_rounds_matches_the_approved_design(H):
    plan = H.soup_plan(20)
    assert list(plan) == ["h1", "h2", "h3", "h4"]
    assert plan["h1"]["ingredients"] == ["s1", "c10", "c20"] and plan["h1"]["weights"] is None
    assert plan["h2"]["ingredients"] == [f"c{r}" for r in range(11, 21)]
    assert plan["h3"]["ingredients"] == ["s1", "h2"] and plan["h3"]["weights"] == [0.5, 0.5]
    assert plan["h4"]["ingredients"] == [f"c{r:02d}" for r in range(1, 21)]
    assert H.screened(20) == ["h1", "h2", "h3", "h4", "c20"]


@pytest.mark.parametrize("rounds", [0, 2, 3, 7])
def test_plan_rejects_bad_round_counts(H, rounds):
    with pytest.raises(ValueError):
        H.soup_plan(rounds)


def test_recipe_is_v2_1c_window_3_on_the_allocated_base(H):
    cfg = H.recipe(30_000_000, "x")
    assert cfg.run_name == "x"
    assert (cfg.generation_seed, cfg.train_seed, cfg.promotion_seed) == (30_000_000,) * 3
    assert cfg.replay_window_rounds == 3 and cfg.n_games_per_round == 6000 and cfg.generation_workers == 8
    assert cfg.deep_fraction == 0.25 and cfg.deep_depth == 2 and cfg.deep_targets and cfg.afterstate_rows
    assert not cfg.gate_enabled and cfg.learning_rate == 3e-4
    # the run_window.py win3 control used the same recipe: compare every field but name and seeds
    win3 = json.loads((REPO / "runs/window_test/win3/config.json").read_text())
    ours = json.loads(json.dumps(cfg.__dict__, default=list))
    skip = {"run_name", "generation_seed", "train_seed", "promotion_seed"}
    diff = {k: (win3.get(k), ours[k]) for k in ours if k not in skip and win3.get(k, ours[k]) != ours[k]}
    assert diff == {}


def test_seed_bands_do_not_overlap(H):
    base, rounds, shards = 30_000_000, 20, 8
    gen = {base + r * 1000 + s for r in range(rounds) for s in range(shards)}
    promo = {base + r for r in range(rounds)}
    health = {base + H.HEALTH_OFFSET + i for i in range(8)}
    screen = {base + H.SCREEN_OFFSET + c for c in range(8)}
    assert not (health & (gen | promo)) and not (screen & (gen | promo | health))
    assert max(screen) < base + 10_000_000


def _fake_run(tmp_path, rounds, complete=None):
    from parchis.az import round_loop
    run = tmp_path / "scratch" / "r"
    for r in range(rounds):
        rd = round_loop._round_dir(run, r)
        _save(rd / "candidate.pt", 100 + r)
        if complete is None or r < complete:
            (rd / "done.json").write_text("{}")
    return run


def test_harvest_copies_every_candidate_with_one_based_labels(H, tmp_path):
    run = _fake_run(tmp_path, 4)
    paths = H.harvest_candidates(run, tmp_path / "out", 4)
    assert list(paths) == ["c01", "c02", "c03", "c04"]
    from parchis.az import round_loop
    for r in range(4):
        assert paths[f"c{r + 1:02d}"].read_bytes() == (round_loop._round_dir(run, r) / "candidate.pt").read_bytes()


def test_harvest_refuses_an_unfinished_run(H, tmp_path):
    run = _fake_run(tmp_path, 4, complete=3)
    with pytest.raises(FileNotFoundError):
        H.harvest_candidates(run, tmp_path / "out", 4)


def test_build_soups_h3_is_half_start_half_swa(H, tmp_path):
    nets = dict(s1=_save(tmp_path / "s1.pt", 1))
    nets.update({f"c{r:02d}": _save(tmp_path / f"c{r:02d}.pt", 10 + r) for r in range(1, 5)})
    cfgs = H.build_soups(H.soup_plan(4), nets, tmp_path / "soups", "PILA-5")
    assert set(cfgs) == {"h1", "h2", "h3", "h4"}
    sd = {k: torch.load(v) for k, v in nets.items()}
    for k in sd["s1"]:
        if not sd["s1"][k].is_floating_point():
            continue
        swa = (sd["c03"][k] + sd["c04"][k]) / 2
        assert torch.allclose(sd["h2"][k], swa, atol=1e-6)
        assert torch.allclose(sd["h3"][k], 0.5 * sd["s1"][k] + 0.25 * sd["c03"][k] + 0.25 * sd["c04"][k], atol=1e-6)
        assert torch.allclose(sd["h1"][k], (sd["s1"][k] + sd["c02"][k] + sd["c04"][k]) / 3, atol=1e-6)
        assert torch.allclose(sd["h4"][k], sum(sd[f"c{r:02d}"][k] for r in range(1, 5)) / 4, atol=1e-6)
    assert json.loads((tmp_path / "soups/h3/config.json").read_text())["ingredient_names"] == ["s1", "h2"]


def _s(wr):
    return dict(win_rate=wr)


def test_winner_is_best_screened_net(H):
    ranked, winner, killed = H.pick_winner(dict(h1=_s(.51), h2=_s(.53), c20=_s(.49)), ["h1", "h2"])
    assert ranked == ["h2", "h1", "c20"] and winner == "h2" and not killed


def test_kill_when_every_soup_is_below_half_even_if_the_endpoint_is_above(H):
    ranked, winner, killed = H.pick_winner(dict(h1=_s(.49), h2=_s(.495), c20=_s(.51)), ["h1", "h2"])
    assert killed and winner is None and ranked[0] == "c20"


def test_start_checkpoint_is_the_champion(H):
    pointer = json.loads((REPO / "runs/champions.json").read_text())["2p"]["current"]
    assert H.S1 == REPO / pointer["path"]
    from parchis.registry import record
    assert record.file_sha256(H.S1) == pointer["sha256"]
