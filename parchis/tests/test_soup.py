"""Tests for weight soups (parchis/az/soup.py, scripts/v2/soup.py; PILA-8)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from parchis.az import champion_pool, encoding, soup
from parchis.az.net import AZNet

REPO = Path(__file__).resolve().parents[2]
IN = encoding.encoding_size(2)


def _net(seed, hidden=(16, 16)):
    torch.manual_seed(seed)
    return AZNet(IN, 2, hidden_sizes=hidden)


def _save(tmp_path, name, seed, hidden=(16, 16)):
    p = tmp_path / name / "model.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_net(seed, hidden).state_dict(), p)
    return p


def test_uniform_average_is_elementwise_mean():
    sds = [_net(s).state_dict() for s in (1, 2, 3)]
    out = soup.average_state_dicts(sds)
    assert list(out) == list(sds[0])
    for k in out:
        expected = (sds[0][k].double() + sds[1][k].double() + sds[2][k].double()) / 3
        assert out[k].dtype == sds[0][k].dtype
        assert torch.allclose(out[k].double(), expected, atol=1e-7)


def test_weights_are_normalised_and_order_free():
    a, b = _net(1).state_dict(), _net(2).state_dict()
    out = soup.average_state_dicts([a, b], weights=[3, 1])
    rev = soup.average_state_dicts([b, a], weights=[1, 3])
    for k in out:
        assert torch.allclose(out[k], 0.75 * a[k] + 0.25 * b[k], atol=1e-6)
        assert torch.equal(out[k], rev[k])


def test_soup_of_identical_nets_is_the_net():
    a = _net(5).state_dict()
    out = soup.average_state_dicts([a, a, a])
    for k in a:
        assert torch.allclose(out[k], a[k], atol=1e-7)


@pytest.mark.parametrize("weights", [[1.0], [1.0, -1.0], [1.0, 0.0], [1.0, float("nan")]])
def test_bad_weights_rejected(weights):
    with pytest.raises(ValueError):
        soup.average_state_dicts([_net(1).state_dict(), _net(2).state_dict()], weights=weights)


def test_needs_two_ingredients():
    with pytest.raises(ValueError):
        soup.average_state_dicts([_net(1).state_dict()])


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError, match="shape"):
        soup.average_state_dicts([_net(1, (16, 16)).state_dict(), _net(2, (16, 32)).state_dict()])


def test_mismatched_keys_rejected():
    a, b = _net(1).state_dict(), _net(2).state_dict()
    del b["aux_head.bias"]
    with pytest.raises(ValueError, match="keys differ"):
        soup.average_state_dicts([a, b])
    with pytest.raises(ValueError, match="keys differ"):
        soup.average_state_dicts([_net(1, (16, 16)).state_dict(), _net(2, (16, 16, 16)).state_dict()])


def test_mismatched_dtype_rejected():
    a, b = _net(1).state_dict(), _net(2).state_dict()
    b["value_head.bias"] = b["value_head.bias"].double()
    with pytest.raises(ValueError, match="dtype"):
        soup.average_state_dicts([a, b])


def test_make_soup_writes_loadable_checkpoint_and_config(tmp_path):
    a, b = _save(tmp_path, "a", 1), _save(tmp_path, "b", 2)
    out = tmp_path / "soup" / "model.pt"
    cfg = soup.make_soup(out, [a, b])
    on_disk = json.loads((out.parent / "config.json").read_text())
    assert on_disk == cfg
    assert [i["weight"] for i in cfg["ingredients"]] == [0.5, 0.5] and cfg["weighting"] == "uniform"
    assert len(cfg["ingredients"][0]["sha256"]) == 64 and len(cfg["output"]["sha256"]) == 64
    net = champion_pool.load_numpy_net(str(out), IN, 2, (16, 16))     # the path every match uses
    assert net is not None
    assert soup.hidden_sizes_from_state_dict(torch.load(out)) == (16, 16)


def _shard(tmp_path, n=64, seed=0, prob_policy=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, IN)).astype(np.float32)
    p = rng.integers(0, 4, n).astype(np.int64)
    if prob_policy:                                          # round shards store visit distributions
        p = rng.dirichlet(np.ones(4), n).astype(np.float32)
    winner = rng.integers(0, 2, n)
    v = np.eye(2, dtype=np.float32)[winner]
    pw = (rng.random(n) > 0.3).astype(np.float32)          # afterstate rows: policy weight 0
    path = tmp_path / "shard.npz"
    np.savez(path, X=X, policy_targets=p, value_targets=v, aux_targets=np.zeros((n, 4), np.float32),
             policy_weights=pw)
    return path, X, p, v, pw


@pytest.mark.parametrize("prob_policy", [False, True])
def test_shard_losses_match_direct_cross_entropy(tmp_path, prob_policy):
    path, X, p, v, pw = _shard(tmp_path, prob_policy=prob_policy)
    net = _net(3)
    data = soup.load_shards([path])
    got = soup.shard_losses(net.state_dict(), data, batch_size=10)      # several batches
    with torch.no_grad():
        pl, vl, _ = net.eval()(torch.from_numpy(X))
        v_rows = torch.nn.functional.cross_entropy(vl, torch.from_numpy(v), reduction="none")
        p_rows = torch.nn.functional.cross_entropy(pl, torch.from_numpy(p), reduction="none")
    w = torch.from_numpy(pw)
    assert got["value_loss"] == pytest.approx(v_rows.mean().item(), rel=1e-5)
    assert got["value_loss_decisions"] == pytest.approx(v_rows[w > 0].mean().item(), rel=1e-5)
    assert got["policy_loss"] == pytest.approx(((p_rows * w).sum() / w.sum()).item(), rel=1e-5)
    assert got["n_rows"] == len(X) and got["n_decision_rows"] == int((pw > 0).sum())


def test_health_verdict_margin():
    assert soup.health_verdict(0.50, [0.48, 0.49])
    assert soup.health_verdict(0.4999, [0.48, 0.49])          # within 0.01 of the worst
    assert not soup.health_verdict(0.5001 + 0.0001, [0.48, 0.49])


def _load_cli():
    spec = importlib.util.spec_from_file_location("soup_cli", REPO / "scripts/v2/soup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_weights_and_usage(tmp_path, capsys):
    cli = _load_cli()
    a, b, c = (_save(tmp_path, n, s) for n, s in (("a", 1), ("b", 2), ("c", 3)))
    out = tmp_path / "s" / "model.pt"
    assert cli.main([str(out), str(a), str(b), str(c), "weights=1,1,2"]) == 0
    cfg = json.loads((out.parent / "config.json").read_text())
    assert [i["weight"] for i in cfg["ingredients"]] == [0.25, 0.25, 0.5] and cfg["weighting"] == "explicit"
    assert cli.main([str(out), str(a)]) == 4                    # one ingredient: usage
    assert cli.main([str(out), str(a), str(b), "bogus=1"]) == 4


def test_real_ingredients_are_soupable():
    """The four PILA-8 ingredients share keys and shapes (skipped where the checkpoints are absent)."""
    paths = [REPO / p for p in ("runs/v2_1c_champion/model.pt", "runs/v2_1d_champion/model.pt",
                                "runs/window_test/win3/model.pt", "runs/window_test/win10/model.pt")]
    if not all(p.exists() for p in paths):
        pytest.skip("ingredient checkpoints not present")
    sd = soup.average_state_dicts([torch.load(p, map_location="cpu") for p in paths])
    m = AZNet(IN, 2, hidden_sizes=soup.hidden_sizes_from_state_dict(sd))
    m.load_state_dict(sd)                                      # strict: every key present
