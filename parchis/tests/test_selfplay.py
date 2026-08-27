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


class ObservationCapturingFakeModel:
    """Like CountingFakeModel, but on every predict() call also checks the
    observation's own-piece block against the ACTUAL current player's own
    pieces, read live from the env at that exact moment -- catches the bug
    where the observation instead always described the learning agent's
    pieces, whoever was actually deciding (docs/AGENT_REBUILD_PLAN.md
    §1.3)."""

    def __init__(self, env):
        self.env = env
        self.predict_calls = 0
        self.mismatches = []

    def predict(self, obs, action_masks=None, deterministic=False):
        self.predict_calls += 1
        base_env = self.env.base_env
        acting_player = base_env.game.get_current_player()
        offset = (
            base_env.board_state_size + base_env.global_state_size
            - base_env.OWN_PIECE_FEATURES_SIZE - base_env.STRATEGIC_FEATURES_SIZE
        )
        stride = base_env.PIECE_FEATURES_PER_PIECE
        for piece in acting_player.pieces:
            slot = offset + piece.piece_id * stride
            expected_in_base = 1.0 if piece.in_base else 0.0
            if obs[slot] != expected_in_base:
                self.mismatches.append(
                    (self.predict_calls, piece.piece_id, expected_in_base, float(obs[slot]))
                )

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


def test_opponent_model_observation_reflects_acting_players_own_pieces():
    """Regression test for docs/AGENT_REBUILD_PLAN.md §1.3: every
    observation handed to an opponent's model must describe the ACTING
    player's own pieces, not the learning agent's -- checked across every
    decision (including bonus-chain moves) over a full episode. Before
    _get_observation(perspective_seat=...) existed, _choose_opponent_move
    had no way to ask for anything but agent_player_idx's perspective, so
    this would have failed as soon as any opponent's in_base pieces
    differed from the agent's own (virtually always, within a few turns)."""
    print("\nTesting opponent model observations reflect the acting player's own pieces...")

    env = ParchisSelfPlayEnv(num_players=4)
    model = ObservationCapturingFakeModel(env)
    env.update_opponent_model(model)

    _play_until_opponent_moves(env, max_steps=400)

    assert model.predict_calls > 20, (
        f"Expected many opponent decisions, got only {model.predict_calls}"
    )
    assert not model.mismatches, (
        f"{len(model.mismatches)} opponent observation(s) did not reflect "
        f"the acting player's own pieces (call#, piece_id, expected, got): "
        f"{model.mismatches[:5]}"
    )
    print(f"✓ All {model.predict_calls} opponent-model observations correctly "
          f"reflected the acting player's own pieces")
    env.close()


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


def test_opponent_seats_can_hold_different_pool_members_within_one_episode():
    """At num_players=4 there are 3 opponent seats. With a real (>=2) pool,
    at least one reset should assign different pool members to different
    seats within the SAME episode, not just across episodes -- regression
    guard for the old design, where self.opponent_model was one shared
    scalar and every non-agent seat necessarily got the same model."""
    print("\nTesting opponent seats can hold different pool members within one episode...")

    models = [CountingFakeModel() for _ in range(3)]
    env = ParchisSelfPlayEnv(num_players=4, pool_seed=11)
    env.update_opponent_pool(models)

    saw_heterogeneous_episode = False
    for _ in range(30):
        env.reset(seed=0)
        seat_models = list(env.opponent_models.values())
        assert len(seat_models) == 3, (
            f"Expected exactly 3 opponent seats populated at num_players=4, "
            f"got {len(seat_models)}"
        )
        if len(set(id(m) for m in seat_models)) > 1:
            saw_heterogeneous_episode = True
            break

    assert saw_heterogeneous_episode, (
        "Expected at least one episode (over 30 resets) where different "
        "opponent seats held different pool members -- got the same model "
        "in every seat every time, suggesting per-seat sampling isn't wired up"
    )
    print("✓ Different opponent seats held different pool members within one episode")
    env.close()


def test_two_player_games_have_exactly_one_opponent_seat():
    """At num_players=2 there's only one opponent seat -- heterogeneous
    seat sampling is a strict generalization with no behavior change here."""
    print("\nTesting num_players=2 still has exactly one opponent seat...")

    models = [CountingFakeModel() for _ in range(3)]
    env = ParchisSelfPlayEnv(num_players=2, pool_seed=11)
    env.update_opponent_pool(models)

    env.reset(seed=0)
    assert len(env.opponent_models) == 1
    print("✓ num_players=2 has exactly one opponent seat, as before")
    env.close()


def _play_episode(env, seed, max_steps=150):
    """Like _play_until_opponent_moves, but resets with a caller-chosen seed
    instead of a fixed seed=0 -- needed to actually vary the dice-determined
    starting player (and hence Game.__init__'s seat rotation) across
    episodes, rather than replaying the same rotation every time."""
    obs, info = env.reset(seed=seed)
    for _ in range(max_steps):
        mask = info['action_masks']
        legal = np.where(mask)[0]
        action = int(legal[0]) if len(legal) else 0
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_opponent_model_used_for_every_decision_across_rotations(monkeypatch):
    """
    Regression test for the seat-vs-player_id lookup bug: self.opponent_models
    is keyed by seat (list position in game.players), but was previously
    looked up via player.player_id, which Game.__init__ deliberately
    reorders self.players (rotating it so the dice-determined starting
    player lands at index 0 -- parchis/game/game.py:75-76) without ever
    updating player_id to match. Whenever that rotation isn't the identity
    (most games), the old lookup silently missed and fell back to
    Player.choose_move's random selection instead of raising -- exactly the
    kind of bug a weak "model called at least once" assertion can't catch.

    Directly instruments _choose_opponent_move (rather than just counting
    Player.choose_move calls, which also fires for a separate, pre-existing,
    documented edge case around bonus-move action masks -- see the "shouldn't
    happen given action_masks" guard a few lines below the lookup) to record,
    for every opponent decision: whether player.player_id actually diverged
    from its live seat this call (proving the test scenario genuinely
    exercises rotated seating, not just the identity case), and whether the
    seat-keyed lookup found a model. With a full pool (one distinct model
    per opponent seat) run across many seeds, divergence must occur at least
    once, and the lookup must never miss even when it does.
    """
    print("\nTesting the opponent model lookup survives non-identity seat rotations...")

    import parchis.rl.env_selfplay as env_selfplay_module

    original_choose = env_selfplay_module.ParchisSelfPlayEnv._choose_opponent_move
    models = [CountingFakeModel() for _ in range(3)]  # one per opponent seat at num_players=4
    observations = []  # (player_id, seat, diverged, model_actually_consulted)

    def instrumented_choose(self, player, legal_moves):
        if not legal_moves:
            # Nothing to decide -- _choose_opponent_move returns None
            # immediately without ever reaching the model lookup, so this
            # isn't a decision the model needed to be consulted for.
            return original_choose(self, player, legal_moves)
        # Ground truth, computed independently of whatever lookup the real
        # method below uses -- this must reflect what SHOULD happen, not
        # what the (possibly buggy) implementation happens to compute.
        true_seat = self.base_env.game.players.index(player)
        diverged = player.player_id != true_seat
        calls_before = sum(m.predict_calls for m in models)
        result = original_choose(self, player, legal_moves)
        model_consulted = sum(m.predict_calls for m in models) > calls_before
        observations.append((player.player_id, true_seat, diverged, model_consulted))
        return result

    monkeypatch.setattr(env_selfplay_module.ParchisSelfPlayEnv, "_choose_opponent_move",
                         instrumented_choose)

    env = ParchisSelfPlayEnv(num_players=4, pool_seed=42)
    env.update_opponent_pool(models)

    for seed in range(20):  # varied seeds -> varied dice-determined starting players/rotations
        _play_episode(env, seed=seed)

    assert len(observations) > 50, (
        f"Expected many opponent decisions across 20 episodes, got only "
        f"{len(observations)} -- test isn't exercising enough play"
    )
    diverged = [o for o in observations if o[2]]
    assert diverged, (
        "Expected at least one opponent decision where player_id != seat "
        "(a non-identity starting-player rotation) across 20 varied seeds -- "
        "got none, so this test isn't actually exercising the rotation the "
        "bug depends on"
    )
    missed_lookups = [o for o in diverged if not o[3]]
    assert not missed_lookups, (
        f"{len(missed_lookups)} opponent decisions had player_id != seat AND "
        f"no model found at that seat -- the seat-vs-player_id lookup bug is "
        f"back: {missed_lookups[:5]}"
    )
    print(f"✓ {len(observations)} opponent decisions across 20 varied-seed episodes, "
          f"{len(diverged)} with player_id != seat, all correctly resolved to a model")
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
    test_opponent_model_observation_reflects_acting_players_own_pieces()
    test_opponent_model_is_actually_invoked()
    test_opponent_model_used_for_bonus_moves_too()
    test_no_opponent_model_falls_back_to_random()
    test_pool_sampling_uniform_covers_multiple_models()
    test_pool_sampling_skewed_weights_concentrate_on_dominant_model()
    test_opponent_seats_can_hold_different_pool_members_within_one_episode()
    test_two_player_games_have_exactly_one_opponent_seat()
    test_update_opponent_model_still_behaves_as_single_model_pool()
    print("\nAll self-play tests passed!")
