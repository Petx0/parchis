#!/usr/bin/env python3
"""
Tests for parchis/az/champion_pool.py (docs/AGENT_REBUILD_PLAN.md Part 3
Phase 3): the promoted-checkpoint FIFO history and the {champion, last 4
promoted, tuned heuristic, random} pool construction.
"""

import numpy as np
import torch

from parchis.az import champion_pool
from parchis.az.net import AZNet, NumpyAZNet


def test_append_promoted_caps_at_four_fifo():
    print("\nTesting append_promoted caps at 4, evicting oldest-first...")
    history = []
    for i in range(6):
        history = champion_pool.append_promoted(history, f"round_{i:03d}/model.pt")
    assert history == ["round_002/model.pt", "round_003/model.pt",
                        "round_004/model.pt", "round_005/model.pt"]
    print(f"✓ after 6 appends, history={history} (oldest 2 evicted)")


def test_append_promoted_does_not_mutate_input():
    print("\nTesting append_promoted does not mutate its input list...")
    original = ["a.pt", "b.pt"]
    updated = champion_pool.append_promoted(original, "c.pt")
    assert original == ["a.pt", "b.pt"], "append_promoted must not mutate its input"
    assert updated == ["a.pt", "b.pt", "c.pt"]
    print("✓ input list unchanged; a new list was returned")


def test_save_and_load_promoted_history_round_trips(tmp_path):
    print("\nTesting save/load_promoted_history round-trips...")
    path = tmp_path / "nested" / "promoted_history.json"
    history = ["runs/x/rounds/round_000/model.pt", "runs/x/rounds/round_003/model.pt"]

    champion_pool.save_promoted_history(path, history)
    loaded = champion_pool.load_promoted_history(path)

    assert loaded == history
    print(f"✓ round-tripped {loaded} through {path}, including creating parent dirs")


def test_load_promoted_history_missing_file_returns_empty(tmp_path):
    print("\nTesting load_promoted_history returns [] for a missing file...")
    result = champion_pool.load_promoted_history(tmp_path / "does_not_exist.json")
    assert result == []
    print("✓ [] for a fresh run with no promotions yet")


def test_build_pool_composition_with_no_promotions():
    print("\nTesting build_pool with zero promoted checkpoints...")
    champion = object()
    nets, anchors = champion_pool.build_pool(champion, [])
    assert nets == (champion,), "Expected nets = (champion,) with no promotions yet"
    assert len(anchors) == 2, "Expected anchor_factories = (tuned heuristic, random)"
    print(f"✓ nets={len(nets)} (just the champion), anchors={len(anchors)}")


def test_build_pool_composition_with_four_promotions():
    print("\nTesting build_pool with four promoted checkpoints...")
    champion = object()
    promoted = [object(), object(), object(), object()]
    nets, anchors = champion_pool.build_pool(champion, promoted)
    assert nets == (champion, *promoted), "Expected nets = (champion, *4 promoted)"
    assert len(nets) == 5
    assert len(anchors) == 2
    print(f"✓ nets={len(nets)} (champion + 4 promoted), anchors={len(anchors)}")


def test_append_recent_caps_at_n_fifo():
    print("\nTesting append_recent caps at MAX_RECENT_HISTORY, evicting oldest-first...")
    history = []
    n = champion_pool.MAX_RECENT_HISTORY
    for i in range(n + 2):
        history = champion_pool.append_recent(history, f"round_{i:03d}/candidate.pt")
    assert len(history) == n
    assert history == [f"round_{i:03d}/candidate.pt" for i in range(2, n + 2)]
    print(f"✓ after {n + 2} appends, history has {len(history)} entries (oldest 2 evicted)")


def test_append_recent_does_not_mutate_input():
    print("\nTesting append_recent does not mutate its input list...")
    original = ["a.pt", "b.pt"]
    updated = champion_pool.append_recent(original, "c.pt")
    assert original == ["a.pt", "b.pt"], "append_recent must not mutate its input"
    assert updated == ["a.pt", "b.pt", "c.pt"]
    print("✓ input list unchanged; a new list was returned")


def test_save_and_load_recent_history_round_trips(tmp_path):
    print("\nTesting save/load_recent_history round-trips...")
    path = tmp_path / "nested" / "recent_history.json"
    history = ["runs/x/rounds/round_010/candidate.pt", "runs/x/rounds/round_011/candidate.pt"]

    champion_pool.save_recent_history(path, history)
    loaded = champion_pool.load_recent_history(path)

    assert loaded == history
    print(f"✓ round-tripped {loaded} through {path}, including creating parent dirs")


def test_load_recent_history_missing_file_returns_empty(tmp_path):
    print("\nTesting load_recent_history returns [] for a missing file...")
    result = champion_pool.load_recent_history(tmp_path / "does_not_exist.json")
    assert result == []
    print("✓ [] for a fresh run with no rounds completed yet")


def test_build_pool_composition_with_recent_and_promoted_combined():
    print("\nTesting build_pool combines promoted AND recent checkpoints...")
    champion = object()
    promoted = [object(), object()]
    recent = [object(), object(), object()]
    nets, anchors = champion_pool.build_pool(champion, promoted, recent_numpy_nets=recent)
    assert nets == (champion, *promoted, *recent), (
        "Expected nets = (champion, *promoted, *recent), in that order"
    )
    assert len(nets) == 1 + len(promoted) + len(recent)
    assert len(anchors) == 2
    print(f"✓ nets={len(nets)} (champion + {len(promoted)} promoted + {len(recent)} recent), "
          f"anchors={len(anchors)}")


def test_build_pool_recent_defaults_to_empty():
    print("\nTesting build_pool's recent_numpy_nets defaults to empty (backward compatible)...")
    champion = object()
    promoted = [object()]
    nets, _anchors = champion_pool.build_pool(champion, promoted)
    assert nets == (champion, *promoted), "Omitting recent_numpy_nets must not add anything"
    print(f"✓ nets={nets} unaffected when recent_numpy_nets is omitted")


def test_build_pool_anchor_factories_are_callable_arena_factories():
    print("\nTesting build_pool's anchor_factories are valid arena-style factories...")
    import inspect
    _nets, anchors = champion_pool.build_pool(object(), [object()])
    for factory in anchors:
        assert callable(factory)
        sig = inspect.signature(factory)
        assert len(sig.parameters) == 3, f"Expected factory(game, seat, roll_box), got {sig}"
    print(f"✓ all {len(anchors)} anchor factories are 3-arg callables")


def test_load_numpy_net_matches_saved_weights(tmp_path):
    print("\nTesting load_numpy_net reproduces a saved checkpoint's forward pass...")
    torch.manual_seed(0)
    input_size, num_players, hidden_sizes = 12, 2, (8, 8)
    model = AZNet(input_size, num_players, hidden_sizes=hidden_sizes)
    model.eval()

    model_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), model_path)

    loaded_numpy_net = champion_pool.load_numpy_net(model_path, input_size, num_players, hidden_sizes)
    original_numpy_net = NumpyAZNet.from_torch(model)

    x = np.random.default_rng(0).standard_normal((5, input_size)).astype(np.float32)
    orig_policy, orig_value = original_numpy_net.forward(x)
    loaded_policy, loaded_value = loaded_numpy_net.forward(x)

    assert np.max(np.abs(orig_policy - loaded_policy)) < 1e-6
    assert np.max(np.abs(orig_value - loaded_value)) < 1e-6
    print("✓ load_numpy_net's forward pass matches the original model's to < 1e-6")


if __name__ == '__main__':
    test_append_promoted_caps_at_four_fifo()
    test_append_promoted_does_not_mutate_input()
    print("\n(remaining tests need tmp_path -- run via pytest)")
