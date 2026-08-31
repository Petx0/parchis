#!/usr/bin/env python3
"""
Tests for parchis/az/train.py (docs/AGENT_REBUILD_PLAN.md Part 3 item 11 /
Part 4): the game-level train/val/test split and the supervised bootstrap
training loop.
"""

import pytest

from parchis.az import selfplay, train


def _small_dataset(n_games=120, num_players=2, seed=5):
    pool = selfplay.default_pool_factories(noisy_seed=seed + 1)
    examples, _stats = selfplay.generate_games(pool, n_games=n_games, num_players=num_players,
                                                max_turns=500, seed=seed)
    return examples


def test_split_by_game_has_no_leakage_and_covers_every_game():
    print("\nTesting split_by_game has no leakage and covers every game...")

    examples = _small_dataset()
    train_ex, val_ex, test_ex = train.split_by_game(examples, train_frac=0.8, val_frac=0.1, seed=0)

    train_games = {ex['game_index'] for ex in train_ex}
    val_games = {ex['game_index'] for ex in val_ex}
    test_games = {ex['game_index'] for ex in test_ex}
    all_games = {ex['game_index'] for ex in examples}

    assert train_games.isdisjoint(val_games)
    assert train_games.isdisjoint(test_games)
    assert val_games.isdisjoint(test_games)
    assert train_games | val_games | test_games == all_games
    assert len(train_ex) + len(val_ex) + len(test_ex) == len(examples)
    print(f"✓ {len(train_games)}/{len(val_games)}/{len(test_games)} games in "
          f"train/val/test, disjoint, covering all {len(all_games)} games")


def test_split_by_game_respects_requested_fractions():
    print("\nTesting split_by_game respects the requested fractions (by game count)...")

    examples = _small_dataset(n_games=200)
    train_ex, val_ex, test_ex = train.split_by_game(examples, train_frac=0.7, val_frac=0.2, seed=1)

    n_games = len({ex['game_index'] for ex in examples})
    n_train_games = len({ex['game_index'] for ex in train_ex})
    n_val_games = len({ex['game_index'] for ex in val_ex})
    n_test_games = len({ex['game_index'] for ex in test_ex})

    assert abs(n_train_games / n_games - 0.7) < 0.02
    assert abs(n_val_games / n_games - 0.2) < 0.02
    assert abs(n_test_games / n_games - 0.1) < 0.02
    print(f"✓ game-count fractions: train={n_train_games/n_games:.3f} "
          f"val={n_val_games/n_games:.3f} test={n_test_games/n_games:.3f}")


def test_split_indices_by_game_agrees_with_split_by_game():
    """The array-native split_indices_by_game (used for an already-packed,
    disk-scale dataset) must produce the SAME partition as split_by_game
    given the same game_indices/fractions/seed -- they're meant to be
    equivalent, just operating on different input shapes."""
    print("\nTesting split_indices_by_game agrees with split_by_game...")

    import numpy as np

    examples = _small_dataset(n_games=150, seed=41)
    train_ex, val_ex, test_ex = train.split_by_game(examples, train_frac=0.7, val_frac=0.2, seed=3)

    game_indices = np.array([ex['game_index'] for ex in examples])
    train_mask, val_mask, test_mask = train.split_indices_by_game(
        game_indices, train_frac=0.7, val_frac=0.2, seed=3,
    )

    assert set(ex['game_index'] for ex in train_ex) == set(game_indices[train_mask].tolist())
    assert set(ex['game_index'] for ex in val_ex) == set(game_indices[val_mask].tolist())
    assert set(ex['game_index'] for ex in test_ex) == set(game_indices[test_mask].tolist())
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == len(examples)
    print("✓ split_indices_by_game produces the identical partition as split_by_game")


def test_split_by_game_rejects_invalid_fractions():
    print("\nTesting split_by_game rejects invalid fractions...")

    examples = _small_dataset(n_games=10)
    with pytest.raises(ValueError):
        train.split_by_game(examples, train_frac=0.9, val_frac=0.2)  # sums > 1
    with pytest.raises(ValueError):
        train.split_by_game(examples, train_frac=0.0, val_frac=0.1)
    print("✓ split_by_game raises ValueError on invalid fractions")


def test_bootstrap_train_reduces_loss_and_produces_usable_model():
    """End-to-end smoke test on a small dataset: training must actually
    reduce the training loss (learning is happening, not a no-op), respect
    max_epochs, and produce a model whose numpy/torch forward paths still
    agree (i.e. it's a well-formed AZNet, not something malformed by the
    training loop)."""
    print("\nTesting bootstrap_train reduces loss and produces a usable model...")

    examples = _small_dataset(n_games=150, seed=11)
    train_ex, val_ex, _test_ex = train.split_by_game(examples, train_frac=0.7, val_frac=0.2, seed=0)

    model, history = train.bootstrap_train(
        train_ex, val_ex, num_players=2, hidden_sizes=(32, 32),
        max_epochs=8, patience=8, batch_size=256, seed=0, log_every=0,
    )

    assert 1 <= len(history) <= 8
    assert history[-1]['train_loss'] < history[0]['train_loss'], (
        f"Expected training loss to decrease: {history[0]['train_loss']:.4f} -> "
        f"{history[-1]['train_loss']:.4f}"
    )

    from parchis.az.net import NumpyAZNet
    import numpy as np
    import torch as _torch

    numpy_model = NumpyAZNet.from_torch(model)
    x = np.random.default_rng(0).standard_normal((4, model.input_size)).astype(np.float32)
    with _torch.no_grad():
        t_policy, t_value, _t_aux = model(_torch.from_numpy(x))
    n_policy, n_value = numpy_model.forward(x)
    assert np.max(np.abs(t_policy.numpy() - n_policy)) < 1e-4
    assert np.max(np.abs(t_value.numpy() - n_value)) < 1e-4
    print(f"✓ train_loss {history[0]['train_loss']:.4f} -> {history[-1]['train_loss']:.4f} "
          f"over {len(history)} epochs; trained model's numpy/torch paths still agree")


def test_bootstrap_train_early_stopping_bounds_epochs():
    """With a tiny dataset and patience=1, training must stop well before
    an artificially large max_epochs -- the early-stopping mechanism must
    actually fire, not just be present in the code."""
    print("\nTesting early stopping actually bounds the number of epochs...")

    examples = _small_dataset(n_games=40, seed=17)
    train_ex, val_ex, _test_ex = train.split_by_game(examples, train_frac=0.6, val_frac=0.3, seed=0)

    _model, history = train.bootstrap_train(
        train_ex, val_ex, num_players=2, hidden_sizes=(16, 16),
        max_epochs=200, patience=1, batch_size=256, seed=0, log_every=0,
    )
    assert len(history) < 200, (
        f"Expected early stopping to fire well before 200 epochs on a tiny "
        f"dataset with patience=1, ran the full {len(history)}"
    )
    print(f"✓ Early stopping fired at epoch {len(history)} (< 200)")


def test_save_checkpoint_writes_expected_files(tmp_path):
    print("\nTesting save_checkpoint writes model.pt and metrics.jsonl...")

    from parchis.az.config import BootstrapConfig

    examples = _small_dataset(n_games=40, seed=23)
    train_ex, val_ex, _test_ex = train.split_by_game(examples, train_frac=0.6, val_frac=0.3, seed=0)
    model, history = train.bootstrap_train(
        train_ex, val_ex, num_players=2, hidden_sizes=(16, 16),
        max_epochs=3, patience=3, batch_size=256, seed=0, log_every=0,
    )

    config = BootstrapConfig(run_name="test_run")
    run_dir = config.save(runs_dir=str(tmp_path))
    train.save_checkpoint(model, config, history, run_dir)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()

    import json
    with open(run_dir / "metrics.jsonl") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == len(history)
    print(f"✓ config.json, model.pt, metrics.jsonl ({len(lines)} lines) all written correctly")


if __name__ == '__main__':
    test_split_by_game_has_no_leakage_and_covers_every_game()
    test_split_by_game_respects_requested_fractions()
    test_split_indices_by_game_agrees_with_split_by_game()
    test_split_by_game_rejects_invalid_fractions()
    test_bootstrap_train_reduces_loss_and_produces_usable_model()
    test_bootstrap_train_early_stopping_bounds_epochs()
    print("\nAll train tests passed (except tmp_path-dependent one, run via pytest)!")
