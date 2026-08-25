#!/usr/bin/env python3
"""
Tests for parchis/training/cli.py's shared --arch/ARCHITECTURES wiring.

Regression guard for a real question raised during development: does
policy_kwargs built from ARCHITECTURES actually reach the constructed
MaskablePPO policy's network, or could it be silently dropped somewhere
between CLI parsing and model construction? These tests build a real
MaskablePPO against ARCHITECTURES["medium"] and inspect the resulting
policy/mlp_extractor directly, rather than just checking that training
runs without crashing (a silently-ignored architecture flag would still
"work" without erroring).
"""
import argparse

import torch.nn as nn
import pytest
from sb3_contrib import MaskablePPO

from parchis.training import cli
from parchis.training.common import make_env


def test_architectures_presets_shape():
    assert set(cli.ARCHITECTURES.keys()) == {"small", "medium", "large"}
    assert cli.ARCHITECTURES["small"]["net_arch"] == [64, 64]
    assert cli.ARCHITECTURES["small"]["activation_fn"] is nn.Tanh
    assert cli.ARCHITECTURES["medium"]["net_arch"] == [256, 256]
    assert cli.ARCHITECTURES["medium"]["activation_fn"] is nn.ReLU
    assert cli.ARCHITECTURES["large"]["net_arch"] == [512, 256, 128]
    assert cli.ARCHITECTURES["large"]["activation_fn"] is nn.ReLU


def test_add_network_args_exposes_arch_flag_with_correct_choices_and_default():
    parser = argparse.ArgumentParser()
    cli.add_network_args(parser, default_arch="small")

    args = parser.parse_args([])
    assert args.arch == "small"

    args = parser.parse_args(["--arch", "medium"])
    assert args.arch == "medium"

    with pytest.raises(SystemExit):
        parser.parse_args(["--arch", "not-a-real-architecture"])


def test_medium_policy_kwargs_actually_produce_a_256_wide_network():
    """The wiring this project actually cares about: policy_kwargs built
    from ARCHITECTURES["medium"] must genuinely change the constructed
    policy's network, not just be accepted and ignored."""
    env = make_env(num_players=2, seed=1)
    policy_kwargs = dict(
        net_arch=cli.ARCHITECTURES["medium"]["net_arch"],
        activation_fn=cli.ARCHITECTURES["medium"]["activation_fn"],
    )
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    assert model.policy.net_arch == [256, 256]
    assert model.policy.activation_fn is nn.ReLU
    assert _linear_out_features(model.policy.mlp_extractor.policy_net) == [256, 256]
    assert any(isinstance(layer, nn.ReLU) for layer in model.policy.mlp_extractor.policy_net)
    env.close()


def _linear_out_features(sequential_module):
    return [layer.out_features for layer in sequential_module if isinstance(layer, nn.Linear)]


def test_default_small_architecture_matches_sb3s_own_unconfigured_default():
    """Confirms the claim this project relies on elsewhere: every training
    run before --arch existed (no policy_kwargs at all) really was
    equivalent to today's explicit "small" preset, not some other
    unconfigured fallback. SB3 normalizes an unset net_arch=None into an
    internal {'pi': [64,64], 'vf': [64,64]} dict form (vs. our explicit
    flat-list [64,64], applied to both pi/vf identically) -- representation
    differs, so compare the actually-built layers, the ground truth of
    what the policy uses."""
    env_default = make_env(num_players=2, seed=1)
    model_default = MaskablePPO("MlpPolicy", env_default, verbose=0)

    env_small = make_env(num_players=2, seed=1)
    policy_kwargs = dict(
        net_arch=cli.ARCHITECTURES["small"]["net_arch"],
        activation_fn=cli.ARCHITECTURES["small"]["activation_fn"],
    )
    model_small = MaskablePPO("MlpPolicy", env_small, policy_kwargs=policy_kwargs, verbose=0)

    assert model_default.policy.activation_fn is model_small.policy.activation_fn is nn.Tanh
    assert (_linear_out_features(model_default.policy.mlp_extractor.policy_net)
            == _linear_out_features(model_small.policy.mlp_extractor.policy_net)
            == [64, 64])
    assert (_linear_out_features(model_default.policy.mlp_extractor.value_net)
            == _linear_out_features(model_small.policy.mlp_extractor.value_net)
            == [64, 64])
    env_default.close()
    env_small.close()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
