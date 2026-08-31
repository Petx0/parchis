#!/usr/bin/env python3
"""
Tests for parchis/az/selfplay.py's Phase 3 addition, generate_round_games
(docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3), and round_examples_to_arrays.
Named test_selfplay_round.py (parallel to test_selfplay_generation.py,
which covers Phase 2's generate_games).
"""

import numpy as np
import torch

from parchis.az import encoding, selfplay, targets
from parchis.az.net import AZNet, NumpyAZNet


def _tiny_numpy_net(num_players, hidden_sizes=(8, 8), seed=0):
    """A small, randomly-initialized net of the right input shape for
    `num_players` -- fast enough to run real search.search() calls inside
    a test without a trained checkpoint (correctness of these tests
    doesn't depend on the net being any good, only on the plumbing around
    it)."""
    torch.manual_seed(seed)
    input_size = encoding.encoding_size(num_players)
    model = AZNet(input_size, num_players, hidden_sizes=hidden_sizes)
    model.eval()
    return NumpyAZNet.from_torch(model)


def _capture_search_calls(monkeypatch):
    """Monkeypatches parchis.az.selfplay's own `search.search` reference
    to record (mover_seat, move_values, root_value_absolute) for every
    call, in call order -- generate_round_games calls search.search()
    exactly once per RECORDED decision, immediately before appending to
    `examples`, so zipping this list against the returned `examples` (both
    in call/append order) lets a test check exactly what search actually
    returned against what ended up recorded, independent of the
    recording code's own (roll, mover-relative remap, softmax) logic."""
    calls = []
    original_search = selfplay.search.search

    def recording_search(game, **kwargs):
        result = original_search(game, **kwargs)
        calls.append((game.current_player_idx, result[1], result[2]))
        return result

    monkeypatch.setattr(selfplay.search, "search", recording_search)
    return calls


def test_champion_seat_is_always_recorded():
    print("\nTesting generate_round_games records at least the champion seat every game...")
    net = _tiny_numpy_net(num_players=2)
    examples, stats = selfplay.generate_round_games(
        net, [], n_games=30, num_players=2, max_turns=300, depth=1, seed=1,
    )
    assert examples, "Test setup error: expected at least one recorded decision"
    assert stats['n_unrecorded_games'] == 0, (
        f"Expected every game to record at least the guaranteed champion seat, "
        f"got {stats['n_unrecorded_games']}/{stats['n_games']} unrecorded"
    )
    assert stats['n_recorded_decisions'] == len(examples)
    assert stats['n_total_plies'] >= stats['n_recorded_decisions']
    print(f"✓ 0/{stats['n_games']} unrecorded games, {stats['n_recorded_decisions']} recorded "
          f"decisions out of {stats['n_total_plies']} total plies")


def test_root_value_is_mover_relative_not_absolute(monkeypatch):
    """Precise regression test for the SAME mover-relative convention bug
    Phase 2 found the hard way (docs/AZ_DESIGN.md), checked here from the
    start rather than after the fact: search.search() returns root_value
    in ABSOLUTE seat order, but the recorded example must carry it rolled
    into mover-relative order (index 0 = this decision's own mover)."""
    print("\nTesting recorded root_value is mover-relative, not absolute...")
    calls = _capture_search_calls(monkeypatch)

    net = _tiny_numpy_net(num_players=3)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=15, num_players=3, max_turns=300, depth=1, seed=2,
    )
    assert examples
    assert len(calls) == len(examples), "Expected one search.search() call per recorded example"

    checked_seats = set()
    for (mover_seat, _move_values, root_value_absolute), ex in zip(calls, examples):
        assert ex['mover_seat'] == mover_seat
        expected = np.roll(root_value_absolute, -mover_seat)
        assert np.allclose(ex['root_value'], expected), (
            f"mover_seat={mover_seat}: expected root_value={expected}, got {ex['root_value']}"
        )
        checked_seats.add(mover_seat)
    assert len(checked_seats) >= 2, f"Expected multiple distinct movers, saw {checked_seats}"
    print(f"✓ root_value correctly mover-relative across {len(examples)} decisions, "
          f"movers {sorted(checked_seats)}")


def test_aux_target_matches_the_recording_seats_own_final_piece_status(monkeypatch):
    """Phase 4.1: aux_target must be exactly THIS decision's own mover's
    own final piece-finished flags (piece_id-indexed, never seat-rotated
    -- 'my own pieces' has no rotation to apply, unlike root_value/
    outcome), from the SAME game's arena.play_one_game(
    return_piece_status=True) call. Controlled via a monkeypatched
    play_one_game so the expected per-seat status is known exactly,
    rather than inferred from a real game's actual outcome."""
    print("\nTesting aux_target matches the recording seat's own final piece status...")
    fixed_piece_status = {0: [True, True, False, True], 1: [False, False, False, True]}
    original_play_one_game = selfplay.arena.play_one_game

    def fake_play_one_game(*args, **kwargs):
        assert kwargs.get("return_piece_status") is True, (
            "generate_round_games must ask for return_piece_status=True"
        )
        # Let the REAL game actually play out (so choose_move/recording
        # happens normally) -- only substitute a KNOWN piece_status for
        # the assertion below, discarding the real (unpredictable) one.
        real_winner_seat, _real_piece_status = original_play_one_game(*args, **kwargs)
        return real_winner_seat, fixed_piece_status

    monkeypatch.setattr(selfplay.arena, "play_one_game", fake_play_one_game)

    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=5, num_players=2, max_turns=300, depth=1, seed=30,
    )
    assert examples
    seats_seen = set()
    for ex in examples:
        expected = np.array(fixed_piece_status[ex['mover_seat']], dtype=np.float32)
        assert np.array_equal(ex['aux_target'], expected), (
            f"mover_seat={ex['mover_seat']}: expected aux_target={expected}, got {ex['aux_target']}"
        )
        seats_seen.add(ex['mover_seat'])
    assert seats_seen, "Test setup error: no examples recorded"

    monkeypatch.setattr(selfplay.arena, "play_one_game", original_play_one_game)
    print(f"✓ aux_target matched fixed_piece_status[mover_seat] for all {len(examples)} "
          f"decisions, movers seen: {sorted(seats_seen)}")


def test_policy_target_matches_independently_recomputed_softmax(monkeypatch):
    print("\nTesting recorded policy_target matches policy_target_from_move_values on the "
          "same captured move_values...")
    calls = _capture_search_calls(monkeypatch)

    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=15, num_players=2, max_turns=300, depth=1, seed=3,
    )
    assert examples
    assert len(calls) == len(examples)

    for (mover_seat, move_values, _root_value), ex in zip(calls, examples):
        expected = targets.policy_target_from_move_values(
            move_values, mover_seat, tau_target=targets.DEFAULT_TAU_TARGET,
        )
        assert np.allclose(ex['policy_target'], expected, atol=1e-6)
        assert abs(float(ex['policy_target'].sum()) - 1.0) < 1e-5
    print(f"✓ policy_target matches an independently-recomputed softmax over move_values "
          f"for all {len(examples)} decisions")


def test_low_temperature_no_noise_matches_greedy_search(monkeypatch):
    """Near-zero temperature + zero Dirichlet noise must always pick a
    VALUE-optimal (or practically-tied) move -- checked with a small
    absolute tolerance rather than exact equality, for two independent
    reasons: (1) a genuine exact tie between two candidate moves (e.g. two
    same-owner pieces in base both entering on a 6 land on an identical
    square, hence an identical value vector -- the same tie-breaking
    caveat search.py's own tests already carve out) has no single
    "correct" key; and (2) softmax(values / tau) only resolves a value gap
    once it's large relative to tau itself -- a real but tiny (~1e-6, near
    float32 precision) gap between two near-symmetric positions can still
    get a non-negligible probability at tau=1e-6, which is a property of
    temperature scaling, not a bug. A tolerance of 1e-3 comfortably covers
    both cases while still catching a genuinely wrong (meaningfully worse)
    choice."""
    print("\nTesting near-zero temperature + zero Dirichlet noise recovers a value-optimal "
          "move...")
    calls = _capture_search_calls(monkeypatch)

    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=20, num_players=2, max_turns=300, depth=1, seed=4,
        tau_start=1e-6, tau_end=1e-6, dirichlet_epsilon=0.0,
    )
    assert examples
    for (mover_seat, move_values, _root_value), ex in zip(calls, examples):
        best_value = max(move_values[pid][mover_seat] for pid in move_values)
        chosen_value = move_values[ex['chosen_piece_id']][mover_seat]
        assert chosen_value >= best_value - 1e-3, (
            f"Expected near-zero temperature/no noise to always pick a value-optimal move "
            f"(best={best_value}), got piece_id={ex['chosen_piece_id']} (value={chosen_value})"
        )
    print(f"✓ all {len(examples)} decisions picked a value-optimal (or practically-tied) move "
          f"at tau~0, epsilon=0")


def test_exploration_sometimes_picks_a_non_greedy_move(monkeypatch):
    print("\nTesting exploration (temperature + Dirichlet noise) sometimes picks a "
          "non-greedy move...")
    calls = _capture_search_calls(monkeypatch)

    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=60, num_players=2, max_turns=300, depth=1, seed=5,
        tau_start=2.0, tau_end=2.0, dirichlet_epsilon=0.9, dirichlet_alpha=1.0,
    )
    assert examples
    non_greedy = 0
    for (mover_seat, move_values, _root_value), ex in zip(calls, examples):
        expected_greedy = max(move_values, key=lambda pid: move_values[pid][mover_seat])
        if ex['chosen_piece_id'] != expected_greedy:
            non_greedy += 1
    assert non_greedy > 0, (
        "Expected at least one non-greedy choice with high temperature/noise over "
        f"{len(examples)} decisions"
    )
    print(f"✓ {non_greedy}/{len(examples)} decisions picked a non-greedy move under "
          f"high temperature + noise")


def test_anchor_wrapper_never_records_and_still_counts_plies():
    print("\nTesting _make_ply_counting_factory passes through moves untouched and "
          "counts plies without recording...")
    examples = []  # not referenced by the wrapper at all -- confirms no accidental coupling

    def fake_base_factory(game, seat, roll_box):
        def choose_move(legal_moves):
            return "SENTINEL_MOVE"
        return choose_move

    ply_box = {'ply': 0}
    factory = selfplay._make_ply_counting_factory(fake_base_factory, ply_box)
    choose_move = factory(game=None, seat=0, roll_box=None)

    result1 = choose_move(["fake_legal_move"])
    result2 = choose_move(["fake_legal_move"])

    assert result1 == "SENTINEL_MOVE" and result2 == "SENTINEL_MOVE"
    assert ply_box['ply'] == 2
    assert examples == []
    print("✓ base factory's move passed through unchanged, ply_box incremented twice, "
          "nothing recorded")


def test_non_champion_anchor_seats_reduce_recorded_fraction():
    print("\nTesting sampled anchor opponents leave some plies unrecorded overall...")
    net = _tiny_numpy_net(num_players=2)
    examples, stats = selfplay.generate_round_games(
        net, [], n_games=60, num_players=2, max_turns=300, depth=1, seed=6,
    )
    assert stats['n_total_plies'] > stats['n_recorded_decisions'], (
        "With 2 anchors in a 3-entry pool (champion, heuristic, random), expected the "
        "non-champion seat to land on an anchor often enough that some plies go unrecorded"
    )
    print(f"✓ {stats['n_recorded_decisions']}/{stats['n_total_plies']} plies recorded "
          f"(< 100%, confirming anchor-seat plies are excluded)")


def test_truncated_round_game_value_target_outcome_component_is_draw():
    print("\nTesting a truncated round game's value_target degenerates to the draw vector "
          "at lam=0.0 (isolating the outcome component from the root_value blend)...")
    net = _tiny_numpy_net(num_players=2)
    examples, stats = selfplay.generate_round_games(
        net, [], n_games=5, num_players=2, max_turns=2, depth=1, seed=7, lam=0.0,
    )
    assert stats['n_truncated'] == 5, (
        f"Expected all 5 games to truncate at max_turns=2, got {stats['n_truncated']}"
    )
    assert examples, "Test setup error: expected at least one recorded decision before truncation"
    for ex in examples:
        assert np.allclose(ex['value_target'], 0.5), (
            f"At lam=0.0, a truncated game's value_target must equal the draw vector, "
            f"got {ex['value_target']}"
        )
    print(f"✓ all {len(examples)} decisions from truncated games got the exact draw vector "
          f"at lam=0.0")


def test_round_examples_to_arrays_shapes_and_row_sums():
    print("\nTesting round_examples_to_arrays produces correctly-shaped, valid arrays...")
    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=15, num_players=2, max_turns=300, depth=1, seed=8,
    )
    X, policy_targets, value_targets, aux_targets = selfplay.round_examples_to_arrays(
        examples, num_players=2,
    )

    n = len(examples)
    assert X.shape == (n, encoding.encoding_size(2))
    assert policy_targets.shape == (n, 4)
    assert value_targets.shape == (n, 2)
    assert aux_targets.shape == (n, 4)
    assert X.dtype == np.float32 and policy_targets.dtype == np.float32 and value_targets.dtype == np.float32
    assert aux_targets.dtype == np.float32
    assert np.allclose(policy_targets.sum(axis=1), 1.0, atol=1e-4)
    assert np.allclose(value_targets.sum(axis=1), 1.0, atol=1e-4)
    assert set(np.unique(aux_targets).tolist()) <= {0.0, 1.0}, "aux_targets must be 0.0/1.0 flags"
    print(f"✓ X={X.shape}, policy_targets={policy_targets.shape} (rows sum to 1), "
          f"value_targets={value_targets.shape} (rows sum to 1), aux_targets={aux_targets.shape} "
          f"(0.0/1.0 flags)")


def test_promoted_nets_can_occupy_the_non_champion_seat(monkeypatch):
    """With a promoted net in the pool, its decisions must ALSO be
    recordable when sampled for the non-champion seat -- true self-play,
    not just champion-vs-anchor."""
    print("\nTesting a promoted net's own decisions get recorded when sampled...")
    champion_net = _tiny_numpy_net(num_players=2, seed=0)
    promoted_net = _tiny_numpy_net(num_players=2, seed=99)  # distinct weights

    calls = _capture_search_calls(monkeypatch)
    examples, stats = selfplay.generate_round_games(
        champion_net, [promoted_net], n_games=80, num_players=2, max_turns=300,
        depth=1, seed=9,
    )
    assert examples
    assert len(calls) == len(examples)
    # Every recorded decision came from search.search() against SOME net (champion or
    # promoted) -- both are search-capable and indistinguishable from stats alone, but
    # this at minimum confirms promoted nets don't crash the pipeline and games still
    # yield MORE recorded decisions than champion-only would need (a loose but real check
    # that the promoted net is actually being exercised across many games/seats).
    assert stats['n_recorded_decisions'] > stats['n_games'], (
        "Expected more recorded decisions than games (both seats sometimes recordable)"
    )
    print(f"✓ {stats['n_recorded_decisions']} recorded decisions across {stats['n_games']} "
          f"games with a promoted net in the pool")


def test_rollout_target_fraction_zero_never_sets_rollout_value():
    print("\nTesting rollout_target_fraction=0.0 (default) leaves every rollout_value as None...")
    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=20, num_players=2, max_turns=300, depth=1, seed=20,
    )
    assert examples
    assert all(ex['rollout_value'] is None for ex in examples), (
        "Expected no rollout_value to be set when rollout_target_fraction defaults to 0.0"
    )
    print(f"✓ all {len(examples)} recorded decisions have rollout_value=None")


def test_rollout_target_fraction_one_sets_rollout_value_for_every_decision():
    print("\nTesting rollout_target_fraction=1.0 rolls out every recorded decision...")
    net = _tiny_numpy_net(num_players=2)
    examples, _stats = selfplay.generate_round_games(
        net, [], n_games=6, num_players=2, max_turns=150, depth=1, seed=21,
        rollout_target_fraction=1.0, rollout_n=4,
    )
    assert examples
    assert all(ex['rollout_value'] is not None for ex in examples), (
        "Expected every recorded decision to have a rollout_value with rollout_target_fraction=1.0"
    )
    for ex in examples:
        assert ex['rollout_value'].shape == (2,)
        assert abs(float(ex['rollout_value'].sum()) - 1.0) < 1e-4
    print(f"✓ all {len(examples)} recorded decisions have a valid rollout_value")


def test_rollout_value_used_as_bootstrap_term_changes_value_target():
    print("\nTesting value_target uses rollout_value (not root_value) when one was sampled...")
    net = _tiny_numpy_net(num_players=2)
    kwargs = dict(n_games=6, num_players=2, max_turns=150, depth=1, seed=22, lam=0.9)

    examples_root, _ = selfplay.generate_round_games(net, [], **kwargs)
    examples_rollout, _ = selfplay.generate_round_games(
        net, [], rollout_target_fraction=1.0, rollout_n=4, **kwargs,
    )

    assert len(examples_root) == len(examples_rollout), (
        "Same seed must produce the same recorded decisions regardless of rollout settings "
        "(rollout_rng is a separate stream from the move-selection RNGs)"
    )
    # lam=0.9 weights the bootstrap term heavily, so a genuinely different
    # bootstrap source (rollout_value vs root_value) should show up as a
    # different value_target for at least some decisions -- outcome
    # (the other 10%) is identical between the two runs since dirichlet_rng
    # and the outer game-sampling rng are untouched by rollout settings.
    differs = [
        not np.allclose(a['value_target'], b['value_target'], atol=1e-6)
        for a, b in zip(examples_root, examples_rollout)
    ]
    assert any(differs), (
        "Expected at least one decision's value_target to differ between root_value and "
        "rollout modes"
    )
    print(f"✓ {sum(differs)}/{len(differs)} decisions' value_targets differed between modes, "
          f"confirming rollout_value is actually used as the bootstrap term when sampled")


if __name__ == '__main__':
    test_champion_seat_is_always_recorded()
    test_anchor_wrapper_never_records_and_still_counts_plies()
    test_non_champion_anchor_seats_reduce_recorded_fraction()
    test_truncated_round_game_value_target_outcome_component_is_draw()
    test_round_examples_to_arrays_shapes_and_row_sums()
    test_rollout_target_fraction_zero_never_sets_rollout_value()
    test_rollout_target_fraction_one_sets_rollout_value_for_every_decision()
    test_rollout_value_used_as_bootstrap_term_changes_value_target()
    print("\n(remaining tests need monkeypatch -- run via pytest)")
