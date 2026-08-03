#!/usr/bin/env python3
"""
Regression tests for self-play opponent-model wiring.

These specifically target the bug where ParchisEnv.step() already fully
resolved opponent turns with random moves before ParchisSelfPlayEnv's own
(now-removed) opponent-interception loop ever ran, making the opponent
model's predict() unreachable dead code -- so "self-play" training was
silently training against a random opponent the entire time.
"""

import numpy as np

from parchis.rl.env_selfplay import ParchisSelfPlayEnv


class CountingFakeModel:
    """
    Minimal stand-in for a MaskablePPO model: picks the first legal action
    from the mask and counts how many times predict() is called, so tests
    can assert the model was actually consulted.
    """

    def __init__(self):
        self.predict_calls = 0

    def predict(self, obs, action_masks=None, deterministic=False):
        self.predict_calls += 1
        legal = np.where(action_masks)[0]
        action = int(legal[0]) if len(legal) else 0
        return action, None


def _play_until_opponent_moves(env, max_steps=200):
    """Drive the learning agent (whichever seat was randomly assigned this
    episode) with legal moves for up to max_steps, giving opponents plenty
    of opportunity to move (or the model to be consulted)."""
    obs, info = env.reset(seed=0)
    for _ in range(max_steps):
        mask = info['action_masks']
        legal = np.where(mask)[0]
        action = int(legal[0]) if len(legal) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=0)


def test_opponent_model_is_actually_invoked():
    """
    Regression test for the self-play dead-code bug: the opponent model's
    predict() must be called at least once during play, not silently
    bypassed by an internal random auto-play loop.
    """
    print("\nTesting that the opponent model is actually invoked during self-play...")

    fake_model = CountingFakeModel()
    env = ParchisSelfPlayEnv(opponent_model=fake_model, num_players=4)

    _play_until_opponent_moves(env)

    assert fake_model.predict_calls > 0, (
        "Opponent model's predict() was never called -- opponents are "
        "silently falling back to random play (the self-play dead-code bug)"
    )
    assert env.opponent_move_count > 0, (
        "opponent_move_count should increase when the model's action is used"
    )
    print(f"✓ Opponent model was consulted {fake_model.predict_calls} times "
          f"({env.opponent_move_count} moves actually used)")
    env.close()


def test_opponent_model_used_for_bonus_moves_too():
    """
    Chained bonus moves (capture/finish bonuses) for opponents must also go
    through the opponent model, not just their first move of the turn.
    """
    print("\nTesting that opponent bonus moves also consult the model...")

    fake_model = CountingFakeModel()
    env = ParchisSelfPlayEnv(opponent_model=fake_model, num_players=4)

    # Play enough steps that captures/finishes (and their bonus chains) are
    # very likely to occur among the opponents at least once.
    _play_until_opponent_moves(env, max_steps=400)

    # We can't deterministically force a bonus, but predict_calls should
    # exceed opponent_move_count only when the model is asked for a move it
    # can't use, and should track normal + bonus moves together. The key
    # regression check is just that the model is being used repeatedly
    # across many opponent decisions, not just a single call.
    assert fake_model.predict_calls > 5, (
        f"Expected many opponent-move predictions over 400 agent steps, "
        f"got only {fake_model.predict_calls} -- suggests opponents are "
        f"mostly bypassing the model"
    )
    print(f"✓ Model consulted {fake_model.predict_calls} times across the episode")
    env.close()


def test_no_opponent_model_falls_back_to_random():
    """Without an opponent model, self-play should still work (random opponents)."""
    print("\nTesting self-play with no opponent model falls back to random...")

    env = ParchisSelfPlayEnv(opponent_model=None, num_players=4)
    _play_until_opponent_moves(env)

    assert env.opponent_move_count == 0, (
        "opponent_move_count should stay 0 when there is no model to consult"
    )
    print("✓ No-model self-play runs correctly with random opponents")
    env.close()


def test_pool_sampling_uniform_covers_multiple_models():
    """Uniform pool sampling should consult every model in the pool over
    many episodes, not just always pick the first."""
    print("\nTesting uniform opponent-pool sampling covers multiple models...")

    models = [CountingFakeModel() for _ in range(3)]
    env = ParchisSelfPlayEnv(num_players=4, pool_seed=123)
    env.update_opponent_pool(models)  # uniform weights by default

    for _ in range(30):
        _play_until_opponent_moves(env, max_steps=60)

    calls = [m.predict_calls for m in models]
    assert all(c > 0 for c in calls), (
        f"Expected every pool member to be consulted at least once over "
        f"many episodes, got predict_calls={calls}"
    )
    print(f"✓ Uniform pool sampling consulted all {len(models)} models: {calls}")
    env.close()


def test_pool_sampling_skewed_weights_concentrate_on_dominant_model():
    """A heavily skewed weight should concentrate opponent selection on the
    dominant pool member, not spread evenly (deterministic given a seeded
    pool_seed)."""
    print("\nTesting skewed opponent-pool weights concentrate sampling...")

    models = [CountingFakeModel() for _ in range(3)]
    env = ParchisSelfPlayEnv(num_players=4, pool_seed=7)
    env.update_opponent_pool(models, weights=[100.0, 1.0, 1.0])

    for _ in range(30):
        _play_until_opponent_moves(env, max_steps=60)

    calls = [m.predict_calls for m in models]
    assert calls[0] > calls[1] + calls[2], (
        f"Expected the heavily-weighted model to dominate sampling, "
        f"got predict_calls={calls}"
    )
    print(f"✓ Skewed weights concentrated sampling as expected: {calls}")
    env.close()


def test_update_opponent_model_still_behaves_as_single_model_pool():
    """update_opponent_model() must still behave exactly like the old
    single-opponent API: the pool always has exactly that one model, and
    reset() always selects it deterministically (backward-compat check for
    the pool-of-1 == pre-pool steady-state behavior)."""
    print("\nTesting update_opponent_model() backward-compat single-model pool...")

    fake_model = CountingFakeModel()
    env = ParchisSelfPlayEnv(num_players=4, pool_seed=5)
    env.update_opponent_model(fake_model)

    assert env.opponent_pool == [fake_model]
    assert env.opponent_pool_weights == [1.0]

    _play_until_opponent_moves(env, max_steps=200)

    assert fake_model.predict_calls > 0
    assert env.opponent_model is fake_model
    print("✓ update_opponent_model() behaves as an exact single-model pool")
    env.close()


if __name__ == '__main__':
    test_opponent_model_is_actually_invoked()
    test_opponent_model_used_for_bonus_moves_too()
    test_no_opponent_model_falls_back_to_random()
    test_pool_sampling_uniform_covers_multiple_models()
    test_pool_sampling_skewed_weights_concentrate_on_dominant_model()
    test_update_opponent_model_still_behaves_as_single_model_pool()
    print("\nAll self-play tests passed!")
