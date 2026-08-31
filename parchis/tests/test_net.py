#!/usr/bin/env python3
"""
Tests for parchis/az/net.py (docs/AGENT_REBUILD_PLAN.md §2.2 / Phase 1
item 7): the dual-head value/policy net's torch and numpy forward paths.
"""

import numpy as np
import torch

from parchis.az import encoding, net


def test_numpy_and_torch_forward_agree():
    """Part 3 item 7's explicit requirement: numpy and torch outputs must
    agree to 1e-5, across both heads, several batch sizes, and both
    player counts (input size and value-head width both change with N)."""
    print("\nTesting numpy and torch forward paths agree to 1e-5...")

    torch.manual_seed(0)
    for num_players in (2, 4):
        input_size = encoding.encoding_size(num_players)
        model = net.AZNet(input_size, num_players)
        model.eval()
        numpy_model = net.NumpyAZNet.from_torch(model)

        for batch_size in (1, 8, 64):
            x = np.random.default_rng(num_players * 1000 + batch_size).standard_normal(
                (batch_size, input_size)
            ).astype(np.float32)

            with torch.no_grad():
                torch_policy, torch_value, _torch_aux = model(torch.from_numpy(x))
            torch_policy = torch_policy.numpy()
            torch_value = torch_value.numpy()

            numpy_policy, numpy_value = numpy_model.forward(x)

            policy_diff = np.max(np.abs(torch_policy - numpy_policy))
            value_diff = np.max(np.abs(torch_value - numpy_value))
            assert policy_diff < 1e-5, (
                f"num_players={num_players} batch={batch_size}: policy logits "
                f"disagree by {policy_diff} (> 1e-5)"
            )
            assert value_diff < 1e-5, (
                f"num_players={num_players} batch={batch_size}: value logits "
                f"disagree by {value_diff} (> 1e-5)"
            )
        print(f"  num_players={num_players}: max diffs within 1e-5 across batch sizes 1/8/64")
    print("✓ NumpyAZNet and AZNet agree to 1e-5 on both heads")


def test_forward_output_shapes():
    """Both heads' output shapes must be exactly (batch, 4) and
    (batch, num_players), for both a single example and a real batch, and
    for both forward paths identically. AZNet's third (aux) head must be
    (batch, 4) too -- NumpyAZNet has no aux path at all (see net.py's
    module docstring), so it's checked only on the torch side."""
    print("\nTesting forward output shapes...")

    for num_players in (2, 3, 4):
        input_size = encoding.encoding_size(num_players)
        model = net.AZNet(input_size, num_players)
        model.eval()
        numpy_model = net.NumpyAZNet.from_torch(model)

        for batch_size in (1, 5):
            x_np = np.zeros((batch_size, input_size), dtype=np.float32)
            with torch.no_grad():
                t_policy, t_value, t_aux = model(torch.from_numpy(x_np))
            n_policy, n_value = numpy_model.forward(x_np)

            assert tuple(t_policy.shape) == (batch_size, net.NUM_ACTIONS)
            assert tuple(t_value.shape) == (batch_size, num_players)
            assert tuple(t_aux.shape) == (batch_size, net.NUM_AUX_TARGETS)
            assert n_policy.shape == (batch_size, net.NUM_ACTIONS)
            assert n_value.shape == (batch_size, num_players)
    print("✓ Both heads produce (batch, 4) / (batch, num_players) from both forward paths, "
          "AZNet's aux head produces (batch, 4)")


def test_load_state_dict_compat_tolerates_a_pre_aux_head_checkpoint():
    """A checkpoint saved before the aux head existed (missing
    'aux_head.weight'/'aux_head.bias') must load cleanly via
    load_state_dict_compat, leaving the aux head at ITS OWN fresh random
    init -- every other weight (trunk, policy, value) must match the old
    checkpoint exactly."""
    print("\nTesting load_state_dict_compat loads an old, aux-head-less checkpoint...")
    torch.manual_seed(0)
    old_model = net.AZNet(12, 2, hidden_sizes=(8, 8))
    old_state = old_model.state_dict()
    pre_aux_head_state = {k: v for k, v in old_state.items() if not k.startswith("aux_head.")}
    assert "aux_head.weight" not in pre_aux_head_state, "Test setup error: aux_head not actually stripped"

    torch.manual_seed(99)  # different seed -- aux_head's fresh init must NOT match old_model's
    new_model = net.AZNet(12, 2, hidden_sizes=(8, 8))
    fresh_aux_weight = new_model.aux_head.weight.detach().clone()

    new_model.load_state_dict_compat(pre_aux_head_state)

    for key, value in pre_aux_head_state.items():
        assert torch.equal(new_model.state_dict()[key], value), f"{key} should match the old checkpoint exactly"
    assert torch.equal(new_model.aux_head.weight, fresh_aux_weight), (
        "aux_head should be left at its own fresh init, untouched by the old checkpoint"
    )
    print("✓ trunk/policy/value loaded exactly from the old checkpoint; aux_head kept its fresh init")


def test_load_state_dict_compat_still_raises_on_a_genuine_mismatch():
    """load_state_dict_compat's leniency is narrowly scoped to EXACTLY the
    'missing aux_head' case -- any other mismatch (e.g. a real shape
    error, a typo'd key) must still raise, exactly like the strict=True
    default, so this can't silently mask an unrelated bug."""
    print("\nTesting load_state_dict_compat still raises on a genuine mismatch...")
    model = net.AZNet(12, 2, hidden_sizes=(8, 8))
    bad_state = {k: v for k, v in model.state_dict().items() if k != "policy_head.weight"}
    try:
        model.load_state_dict_compat(bad_state)
        assert False, "Expected load_state_dict_compat to raise on a missing NON-aux_head key"
    except RuntimeError:
        pass
    print("✓ raised RuntimeError on a missing policy_head key, as strict loading should")


def test_masked_policy_probs_zeros_illegal_actions():
    """Illegal actions must get EXACTLY 0.0 probability (not a tiny nonzero
    leak from softmax), and the legal entries must sum to 1.0."""
    print("\nTesting masked_policy_probs zeros illegal actions exactly...")

    logits = np.array([5.0, -3.0, 100.0, 0.2], dtype=np.float32)
    mask = np.array([1, 0, 1, 0], dtype=bool)
    probs = net.masked_policy_probs(logits, mask)

    assert probs[1] == 0.0 and probs[3] == 0.0
    assert abs(probs.sum() - 1.0) < 1e-6
    assert probs[2] > probs[0], "The much larger legal logit should dominate"

    # Batched form too.
    batch_logits = np.stack([logits, logits])
    batch_mask = np.stack([mask, mask])
    batch_probs = net.masked_policy_probs(batch_logits, batch_mask)
    assert batch_probs.shape == (2, 4)
    assert np.allclose(batch_probs[0], probs) and np.allclose(batch_probs[1], probs)
    print("✓ masked_policy_probs zeros illegal actions and sums to 1 over legal ones")


def test_value_probs_sums_to_one():
    """The value head's softmax must sum to 1.0 across seats, for any
    num_players, single example or batch."""
    print("\nTesting value_probs sums to 1 across seats...")

    for num_players in (2, 3, 4):
        logits = np.random.default_rng(num_players).standard_normal((7, num_players)).astype(np.float32)
        probs = net.value_probs(logits)
        assert probs.shape == (7, num_players)
        sums = probs.sum(axis=-1)
        assert np.allclose(sums, 1.0), f"num_players={num_players}: sums={sums}"
        assert np.all(probs >= 0.0)
    print("✓ value_probs sums to 1 across seats for num_players in {2,3,4}")


if __name__ == '__main__':
    test_numpy_and_torch_forward_agree()
    test_forward_output_shapes()
    test_load_state_dict_compat_tolerates_a_pre_aux_head_checkpoint()
    test_load_state_dict_compat_still_raises_on_a_genuine_mismatch()
    test_masked_policy_probs_zeros_illegal_actions()
    test_value_probs_sums_to_one()
    print("\nAll net tests passed!")
