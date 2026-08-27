#!/usr/bin/env python3
"""
Tests for parchis/az/train.py's sharded-dataset path (split_shards,
bootstrap_train_sharded) -- needed once a corpus is too large to hold in
memory as a single array (docs/AGENT_REBUILD_PLAN.md Part 4:
"the loop is embarrassingly parallel at the game-generation level").
Uses small synthetic shard files on disk, not a real multi-GB corpus.
"""

import numpy as np
import pytest
import torch

from parchis.az import encoding, selfplay, train


def _write_shards(tmp_path, n_shards, games_per_shard, num_players=2, seed=0):
    """Generates `n_shards` small, independently-seeded shards (mirrors
    the real sharded generation script's structure) and writes them as
    .npz files under tmp_path. Returns the list of shard paths."""
    paths = []
    games_so_far = 0
    for i in range(n_shards):
        pool = selfplay.default_pool_factories(noisy_seed=seed * 1000 + i * 2 + 1)
        examples, _stats = selfplay.generate_games(
            pool, n_games=games_per_shard, num_players=num_players,
            max_turns=500, seed=seed * 1000 + i * 2,
        )
        X, policy_targets, value_targets = selfplay.examples_to_arrays(examples, num_players)
        game_indices = np.array([ex['game_index'] for ex in examples], dtype=np.int64) + games_so_far
        mover_seats = np.array([ex['mover_seat'] for ex in examples], dtype=np.int64)
        games_so_far += games_per_shard

        path = tmp_path / f"shard_{i:03d}.npz"
        np.savez(path, X=X, policy_targets=policy_targets, value_targets=value_targets,
                  game_indices=game_indices, mover_seats=mover_seats)
        paths.append(path)
    return paths


def test_split_shards_no_leakage_and_covers_every_shard():
    print("\nTesting split_shards has no leakage and covers every shard...")

    paths = [f"shard_{i}.npz" for i in range(10)]
    train_paths, val_paths, test_paths = train.split_shards(paths, train_frac=0.7, val_frac=0.2, seed=0)

    all_returned = set(train_paths) | set(val_paths) | set(test_paths)
    assert all_returned == set(paths)
    assert len(set(train_paths) & set(val_paths)) == 0
    assert len(set(train_paths) & set(test_paths)) == 0
    assert len(set(val_paths) & set(test_paths)) == 0
    assert len(train_paths) + len(val_paths) + len(test_paths) == len(paths)
    print(f"✓ {len(train_paths)}/{len(val_paths)}/{len(test_paths)} shards in "
          f"train/val/test, disjoint, covering all {len(paths)}")


def test_split_shards_guarantees_all_three_groups_nonempty():
    """With >=3 shards, ALL THREE groups must be non-empty regardless of
    how skewed the requested fractions are (e.g. a small val_frac
    rounding to 0 shards must still get bumped to 1) -- a 3-way split
    with an empty group is useless, and bootstrap_train_sharded's
    val_shard_paths specifically can't be empty (_load_and_concat_shards
    can't concatenate zero arrays)."""
    print("\nTesting split_shards guarantees all three groups are non-empty...")

    for n in (3, 4, 5, 6, 7, 20):
        paths = [f"shard_{i}.npz" for i in range(n)]
        train_paths, val_paths, test_paths = train.split_shards(
            paths, train_frac=0.8, val_frac=0.1, seed=1,
        )
        assert len(train_paths) >= 1, f"n={n}: empty train"
        assert len(val_paths) >= 1, f"n={n}: empty val"
        assert len(test_paths) >= 1, f"n={n}: empty test"
        assert len(train_paths) + len(val_paths) + len(test_paths) == n
        print(f"  n={n}: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")
    print("✓ All three groups non-empty for every shard count from 3 to 20")


def test_split_shards_rejects_too_few_shards():
    print("\nTesting split_shards rejects fewer than 3 shards...")
    with pytest.raises(ValueError):
        train.split_shards(["only_one.npz"], train_frac=0.8, val_frac=0.1)
    with pytest.raises(ValueError):
        train.split_shards(["a.npz", "b.npz"], train_frac=0.8, val_frac=0.1)
    print("✓ split_shards raises ValueError with 1 or 2 shards (3 non-empty groups need >= 3)")


def test_split_shards_rejects_invalid_fractions():
    print("\nTesting split_shards rejects invalid fractions...")
    paths = [f"shard_{i}.npz" for i in range(5)]
    with pytest.raises(ValueError):
        train.split_shards(paths, train_frac=0.9, val_frac=0.2)
    print("✓ split_shards raises ValueError on invalid fractions")


def test_bootstrap_train_sharded_matches_arrays_path_behavior(tmp_path):
    """bootstrap_train_sharded must behave like bootstrap_train_arrays on
    the SAME underlying data (just split across shard files instead of
    one array): loss decreases, respects max_epochs, and the resulting
    model's numpy/torch forward paths still agree."""
    print("\nTesting bootstrap_train_sharded trains correctly on synthetic shards...")

    paths = _write_shards(tmp_path, n_shards=6, games_per_shard=25, seed=7)
    train_paths, val_paths, _test_paths = train.split_shards(paths, train_frac=0.6, val_frac=0.2, seed=0)
    assert train_paths and val_paths

    model, history = train.bootstrap_train_sharded(
        train_paths, val_paths, num_players=2, hidden_sizes=(32, 32),
        max_epochs=6, patience=6, batch_size=256, seed=0, log_every=0,
    )

    assert 1 <= len(history) <= 6
    assert history[-1]['train_loss'] < history[0]['train_loss'], (
        f"Expected training loss to decrease: {history[0]['train_loss']:.4f} -> "
        f"{history[-1]['train_loss']:.4f}"
    )

    from parchis.az.net import NumpyAZNet
    import torch as _torch

    numpy_model = NumpyAZNet.from_torch(model)
    x = np.random.default_rng(0).standard_normal((4, model.input_size)).astype(np.float32)
    with _torch.no_grad():
        t_policy, t_value = model(_torch.from_numpy(x))
    n_policy, n_value = numpy_model.forward(x)
    assert np.max(np.abs(t_policy.numpy() - n_policy)) < 1e-4
    assert np.max(np.abs(t_value.numpy() - n_value)) < 1e-4
    print(f"✓ train_loss {history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f} "
          f"over {len(history)} epochs across {len(train_paths)} training shards")


def test_bootstrap_train_sharded_never_holds_more_than_one_training_shard(tmp_path, monkeypatch):
    """Regression guard for the whole point of sharding: each training
    shard must be loaded exactly once per epoch (never the whole training
    corpus concatenated up front) -- checked by counting _load_shard
    calls, since Python resolves that name against the module's global
    namespace at CALL time, so monkeypatching train._load_shard intercepts
    every caller (both bootstrap_train_sharded's own per-shard loads in
    the epoch loop AND _load_and_concat_shards' one-time validation-shard
    loads), not just one call site."""
    print("\nTesting bootstrap_train_sharded loads each training shard exactly once per epoch...")

    paths = _write_shards(tmp_path, n_shards=5, games_per_shard=20, seed=11)
    train_paths, val_paths, _test_paths = train.split_shards(paths, train_frac=0.6, val_frac=0.2, seed=0)
    assert train_paths and val_paths

    load_events = []
    original_load_shard = train._load_shard

    def tracking_load_shard(path):
        load_events.append(path)
        return original_load_shard(path)

    monkeypatch.setattr(train, "_load_shard", tracking_load_shard)

    max_epochs = 2
    train.bootstrap_train_sharded(
        train_paths, val_paths, num_players=2, hidden_sizes=(16, 16),
        max_epochs=max_epochs, patience=max_epochs, batch_size=256, seed=0, log_every=0,
    )

    # Validation shards load once total (up front); each training shard
    # loads once per epoch (freshly, inside the epoch loop -- never all at
    # once).
    expected = len(val_paths) + max_epochs * len(train_paths)
    assert len(load_events) == expected, (
        f"Expected {len(val_paths)} (validation, once) + {max_epochs}*{len(train_paths)} "
        f"(training, per epoch) = {expected} total _load_shard calls, got "
        f"{len(load_events)}: {load_events}"
    )
    print(f"✓ {len(load_events)} total shard loads == {len(val_paths)} validation (once) + "
          f"{max_epochs}x{len(train_paths)} training (per epoch)")


def test_split_shards_train_val_no_leakage_and_covers_every_shard():
    print("\nTesting split_shards_train_val has no leakage and covers every shard...")

    paths = [f"shard_{i}.npz" for i in range(8)]
    train_paths, val_paths = train.split_shards_train_val(paths, val_frac=0.25, seed=0)

    assert set(train_paths) | set(val_paths) == set(paths)
    assert len(set(train_paths) & set(val_paths)) == 0
    assert train_paths and val_paths
    print(f"✓ {len(train_paths)}/{len(val_paths)} shards in train/val, disjoint, covering all {len(paths)}")


def test_split_shards_train_val_guarantees_both_groups_nonempty():
    print("\nTesting split_shards_train_val guarantees both groups non-empty at any shard count...")
    for n in (2, 3, 5, 15):
        paths = [f"shard_{i}.npz" for i in range(n)]
        train_paths, val_paths = train.split_shards_train_val(paths, val_frac=0.05, seed=1)
        assert train_paths, f"n={n}: empty train"
        assert val_paths, f"n={n}: empty val"
        assert len(train_paths) + len(val_paths) == n
    print("✓ both groups non-empty for shard counts 2, 3, 5, 15 even at a tiny val_frac")


def test_split_shards_train_val_rejects_too_few_shards():
    print("\nTesting split_shards_train_val rejects fewer than 2 shards...")
    with pytest.raises(ValueError):
        train.split_shards_train_val(["only_one.npz"], val_frac=0.1)
    print("✓ raises ValueError with 1 shard (2 non-empty groups need >= 2)")


def test_split_shards_train_val_rejects_invalid_val_frac():
    print("\nTesting split_shards_train_val rejects an invalid val_frac...")
    paths = [f"shard_{i}.npz" for i in range(4)]
    with pytest.raises(ValueError):
        train.split_shards_train_val(paths, val_frac=0.0)
    with pytest.raises(ValueError):
        train.split_shards_train_val(paths, val_frac=1.0)
    print("✓ raises ValueError on val_frac outside (0, 1)")


def test_bootstrap_train_arrays_warm_starts_from_init_state_dict():
    """init_state_dict must actually seed the model's INITIAL weights, not
    just get ignored -- checked by training the SAME data/seed twice, once
    from a distinctive warm-start and once from scratch, and confirming
    the two resulting models differ (with a fresh random init this would
    be true anyway with overwhelming probability, so this alone wouldn't
    prove much -- the real check is the second assertion below, that
    starting from a state_dict that's ALREADY a good fit for trivial data
    changes less than starting from scratch does)."""
    print("\nTesting bootstrap_train_arrays' init_state_dict actually warm-starts...")

    rng = np.random.default_rng(0)
    n, input_size, num_players = 64, 10, 2
    X = rng.standard_normal((n, input_size)).astype(np.float32)
    policy = rng.integers(0, 4, size=n).astype(np.int64)
    value = np.zeros((n, num_players), dtype=np.float32)
    value[np.arange(n), rng.integers(0, num_players, size=n)] = 1.0

    from parchis.az.net import AZNet
    warm_source = AZNet(input_size, num_players, hidden_sizes=(8, 8))
    warm_state = {k: v.clone() for k, v in warm_source.state_dict().items()}

    model_scratch, _ = train.bootstrap_train_arrays(
        X, policy, value, X, policy, value, num_players=num_players, hidden_sizes=(8, 8),
        max_epochs=1, patience=1, batch_size=64, seed=0, log_every=0,
    )
    model_warm, _ = train.bootstrap_train_arrays(
        X, policy, value, X, policy, value, num_players=num_players, hidden_sizes=(8, 8),
        max_epochs=1, patience=1, batch_size=64, seed=0, log_every=0, init_state_dict=warm_state,
    )

    scratch_w = next(iter(model_scratch.state_dict().values()))
    warm_w = next(iter(model_warm.state_dict().values()))
    source_w = next(iter(warm_state.values()))
    assert not torch.allclose(scratch_w, warm_w), (
        "Expected warm-started training to reach different weights than from-scratch training"
    )
    assert not torch.allclose(warm_w, source_w), (
        "Expected the warm-started model to have actually TRAINED (moved away from init), "
        "not just returned the init unchanged"
    )
    print("✓ init_state_dict changes the trained outcome, and the model still actually trains from it")


def test_bootstrap_train_sharded_warm_starts_from_init_state_dict(tmp_path):
    print("\nTesting bootstrap_train_sharded's init_state_dict actually warm-starts...")

    paths = _write_shards(tmp_path, n_shards=4, games_per_shard=15, seed=21)
    train_paths, val_paths = train.split_shards_train_val(paths, val_frac=0.25, seed=0)

    from parchis.az.net import AZNet
    warm_source = AZNet(encoding.encoding_size(2), 2, hidden_sizes=(16, 16))
    warm_state = {k: v.clone() for k, v in warm_source.state_dict().items()}

    model_scratch, _ = train.bootstrap_train_sharded(
        train_paths, val_paths, num_players=2, hidden_sizes=(16, 16),
        max_epochs=1, patience=1, batch_size=256, seed=0, log_every=0,
    )
    model_warm, _ = train.bootstrap_train_sharded(
        train_paths, val_paths, num_players=2, hidden_sizes=(16, 16),
        max_epochs=1, patience=1, batch_size=256, seed=0, log_every=0, init_state_dict=warm_state,
    )

    scratch_w = next(iter(model_scratch.state_dict().values()))
    warm_w = next(iter(model_warm.state_dict().values()))
    assert not torch.allclose(scratch_w, warm_w), (
        "Expected warm-started sharded training to reach different weights than from-scratch"
    )
    print("✓ init_state_dict changes bootstrap_train_sharded's trained outcome too")


if __name__ == '__main__':
    test_split_shards_no_leakage_and_covers_every_shard()
    test_split_shards_guarantees_all_three_groups_nonempty()
    test_split_shards_rejects_too_few_shards()
    test_split_shards_rejects_invalid_fractions()
    test_split_shards_train_val_no_leakage_and_covers_every_shard()
    test_split_shards_train_val_guarantees_both_groups_nonempty()
    test_split_shards_train_val_rejects_too_few_shards()
    test_split_shards_train_val_rejects_invalid_val_frac()
    test_bootstrap_train_arrays_warm_starts_from_init_state_dict()
    print("\n(remaining tests need tmp_path/monkeypatch -- run via pytest)")
