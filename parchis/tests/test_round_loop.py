#!/usr/bin/env python3
"""
Tests for parchis/az/round_loop.py (docs/AGENT_REBUILD_PLAN.md Part 3
Phase 3): the continuous self-play round loop -- generate, train
(warm-start), promote-or-not, escalate, resume. Uses tiny configs (few
games, tiny nets, few duplicate pairs) so a "real" round still completes
in a couple of seconds; the promotion-outcome-dependent tests monkeypatch
parchis.evaluation.duplicate.play_duplicate_match to force a deterministic
result rather than relying on actual (stochastic) match luck.
"""

import json

from parchis.az import config as config_module
from parchis.az import round_loop


def _tiny_cfg(tmp_path, **overrides):
    kwargs = dict(
        run_name="test_round", num_players=2, n_games_per_round=20, games_per_shard=10,
        max_turns=200, hidden_sizes=(8, 8), warm_start_max_epochs=2, warm_start_patience=1,
        batch_size=64, promotion_n_pairs=3,
    )
    kwargs.update(overrides)
    return config_module.SelfPlayRoundConfig(**kwargs)


def _forced_promotion_result(promoted):
    """A play_duplicate_match-shaped dict whose win_rate_a_ci's lower
    bound is on whichever side of 0.5 `promoted` asks for."""
    ci = (0.9, 0.99) if promoted else (0.1, 0.3)
    return {
        'wins_a': 5 if promoted else 1, 'n_games': 6, 'win_rate_a': ci[0] + 0.01,
        'win_rate_a_ci': ci, 'groups': [], 'pair_record': {'a_better': 1, 'split': 0, 'b_better': 0},
    }


def test_run_round_creates_expected_files_and_is_internally_consistent(tmp_path):
    print("\nTesting run_round creates the expected on-disk files and updates state consistently...")
    cfg = _tiny_cfg(tmp_path)
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    champion_state, meta = round_loop.load_champion_state(run_dir, cfg)
    new_state, new_meta, new_history, new_recent = round_loop.run_round(
        0, champion_state, meta, [], [], cfg, run_dir,
    )

    round_dir = round_loop._round_dir(run_dir, 0)
    assert (round_dir / "shards" / "shard_000.npz").exists()
    assert (round_dir / "shards" / "shard_001.npz").exists()
    assert (round_dir / "metrics.jsonl").exists()
    assert (round_dir / "candidate.pt").exists()
    assert (round_dir / "promotion_result.json").exists()
    assert (round_dir / "done.json").exists()
    assert (run_dir / "champion.pt").exists()
    assert (run_dir / "champion_meta.json").exists()
    assert (run_dir / "recent_history.json").exists()

    with open(round_dir / "promotion_result.json") as f:
        promotion_result = json.load(f)

    if promotion_result['promoted']:
        assert new_meta['promotions'] == 1
        assert new_meta['consecutive_failures'] == 0
        assert len(new_history) == 1
        assert new_history[0] == str(round_dir / "candidate.pt")
    else:
        assert new_meta['promotions'] == 0
        assert new_meta['consecutive_failures'] == 1
        assert new_history == []
    assert new_meta['round'] == 0
    # Unlike promoted_history above, recent_history grows EVERY round
    # regardless of promotion outcome (Phase 3.1's whole point).
    assert new_recent == [str(round_dir / "candidate.pt")], (
        f"Expected recent_history to record this round's candidate regardless of "
        f"promotion outcome, got {new_recent}"
    )
    print(f"✓ round 0 completed (promoted={promotion_result['promoted']}), all expected files "
          f"present, state updated consistently")


def test_champion_updates_only_on_forced_promotion(tmp_path, monkeypatch):
    print("\nTesting the champion state only changes when promotion is forced True...")
    cfg = _tiny_cfg(tmp_path, run_name="forced_promote")
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match",
                         lambda *a, **k: _forced_promotion_result(True))

    champion_state, meta = round_loop.load_champion_state(run_dir, cfg)
    new_state, new_meta, new_history, new_recent = round_loop.run_round(
        0, champion_state, meta, [], [], cfg, run_dir,
    )

    assert new_meta['promotions'] == 1
    assert new_meta['consecutive_failures'] == 0
    assert len(new_history) == 1
    assert len(new_recent) == 1, "recent_history should also grow on a promoted round"
    scratch_key = next(iter(champion_state))
    assert not (new_state[scratch_key] == champion_state[scratch_key]).all(), (
        "Expected the promoted candidate's weights to differ from the untrained random-init champion"
    )
    print("✓ forced promotion: champion state replaced, promoted_history and recent_history both grew, "
          "failures reset")


def test_champion_unchanged_on_forced_non_promotion(tmp_path, monkeypatch):
    print("\nTesting the champion state is unchanged when promotion is forced False...")
    cfg = _tiny_cfg(tmp_path, run_name="forced_reject")
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match",
                         lambda *a, **k: _forced_promotion_result(False))

    champion_state, meta = round_loop.load_champion_state(run_dir, cfg)
    new_state, new_meta, new_history, new_recent = round_loop.run_round(
        0, champion_state, meta, [], [], cfg, run_dir,
    )

    assert new_meta['promotions'] == 0
    assert new_meta['consecutive_failures'] == 1
    assert new_history == []
    assert len(new_recent) == 1, (
        "recent_history should grow even on a NON-promoted round -- this is the whole "
        "point of Phase 3.1's separate history: a candidate that didn't clear the "
        "promotion bar is still kept around as an opponent, unlike promoted_history"
    )
    for key in champion_state:
        assert (new_state[key] == champion_state[key]).all(), f"Expected {key} to be unchanged"
    print("✓ forced non-promotion: champion state byte-identical, promoted_history empty "
          "but recent_history still grew, failures=1")


def test_escalation_triggers_after_n_failures_and_resets(tmp_path, monkeypatch):
    print("\nTesting escalation triggers after N consecutive failures and resets after...")
    cfg = _tiny_cfg(tmp_path, run_name="escalation", escalate_after_failures=2, escalation_depth=2, base_depth=1)
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match",
                         lambda *a, **k: _forced_promotion_result(False))

    champion_state, meta = round_loop.load_champion_state(run_dir, cfg)
    generation_depths = []
    eval_depths = []
    failures_after = []
    for round_num in range(3):
        champion_state, meta, _history, _recent = round_loop.run_round(
            round_num, champion_state, meta, [], [], cfg, run_dir,
        )
        with open(round_loop._round_dir(run_dir, round_num) / "promotion_result.json") as f:
            result = json.load(f)
        generation_depths.append(result['generation_depth'])
        eval_depths.append(result['eval_depth'])
        failures_after.append(meta['consecutive_failures'])

    assert generation_depths == [1, 1, 2], f"Expected base,base,escalated GENERATION depths, got {generation_depths}"
    assert eval_depths == [1, 1, 1], (
        f"Expected eval_depth to ALWAYS be base_depth, even on the escalated round -- "
        f"got {eval_depths} (this is the escalation-confound fix: an escalated round's "
        f"promotion match must not also hand the champion a search-time boost)"
    )
    assert failures_after == [1, 2, 0], (
        f"Expected failures to climb to escalate_after_failures then reset after the "
        f"escalated round, got {failures_after}"
    )
    print(f"✓ generation_depths={generation_depths} (varies with escalation), "
          f"eval_depths={eval_depths} (always base_depth), "
          f"consecutive_failures={failures_after}")


def test_escalation_disabled_when_enable_escalation_false(tmp_path, monkeypatch):
    print("\nTesting enable_escalation=False keeps generation at base_depth "
          "regardless of consecutive_failures...")
    cfg = _tiny_cfg(tmp_path, run_name="escalation_disabled", escalate_after_failures=2,
                     escalation_depth=2, base_depth=1, enable_escalation=False)
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match",
                         lambda *a, **k: _forced_promotion_result(False))

    champion_state, meta = round_loop.load_champion_state(run_dir, cfg)
    generation_depths = []
    failures_after = []
    for round_num in range(3):
        champion_state, meta, _history, _recent = round_loop.run_round(
            round_num, champion_state, meta, [], [], cfg, run_dir,
        )
        with open(round_loop._round_dir(run_dir, round_num) / "promotion_result.json") as f:
            result = json.load(f)
        generation_depths.append(result['generation_depth'])
        failures_after.append(meta['consecutive_failures'])

    assert generation_depths == [1, 1, 1], (
        f"Expected generation depth to STAY at base_depth every round with "
        f"enable_escalation=False, even past escalate_after_failures, got {generation_depths}"
    )
    assert failures_after == [1, 2, 3], (
        f"Expected consecutive_failures to keep climbing (never reset by a "
        f"never-firing escalation) with enable_escalation=False, got {failures_after}"
    )
    print(f"✓ generation_depths={generation_depths} (never escalates), "
          f"consecutive_failures={failures_after} (keeps climbing)")


def test_gather_replay_buffer_shards_respects_the_recency_window(tmp_path):
    print("\nTesting _gather_replay_buffer_shards only includes the recent window...")
    run_dir = tmp_path / "runs" / "buffer_test"
    for round_num in range(5):
        shards_dir = round_loop._round_dir(run_dir, round_num) / "shards"
        shards_dir.mkdir(parents=True)
        (shards_dir / "shard_000.npz").write_bytes(b"")

    this_round_paths = ["round4_own_shard.npz"]
    buffer = round_loop._gather_replay_buffer_shards(run_dir, round_num=4,
                                                       this_round_shard_paths=this_round_paths,
                                                       replay_window_rounds=3)

    assert "round4_own_shard.npz" in buffer
    assert any("round_0002" in p for p in buffer), "Expected round 2's shard (within the window)"
    assert any("round_0003" in p for p in buffer), "Expected round 3's shard (within the window)"
    assert not any("round_0000" in p for p in buffer), "Round 0 is outside a window of 3"
    assert not any("round_0001" in p for p in buffer), "Round 1 is outside a window of 3"
    print(f"✓ buffer includes rounds {{2,3,4}}'s shards only, {len(buffer)} total paths")


def test_find_resume_round_and_run_continuous_never_redoes_a_completed_round(tmp_path, monkeypatch):
    print("\nTesting run_continuous resumes at the right round without redoing completed ones...")
    cfg = _tiny_cfg(tmp_path, run_name="resumable")
    monkeypatch.setattr(round_loop.duplicate, "play_duplicate_match",
                         lambda *a, **k: _forced_promotion_result(False))

    assert round_loop.find_resume_round(tmp_path / "runs" / cfg.run_name) == 0

    round_loop.run_continuous(cfg, runs_dir=str(tmp_path / "runs"), max_rounds=2)
    run_dir = tmp_path / "runs" / cfg.run_name
    assert round_loop.find_resume_round(run_dir) == 2

    round0_done_mtime = (round_loop._round_dir(run_dir, 0) / "done.json").stat().st_mtime_ns

    round_loop.run_continuous(cfg, runs_dir=str(tmp_path / "runs"), max_rounds=3)
    assert round_loop.find_resume_round(run_dir) == 3
    assert (round_loop._round_dir(run_dir, 0) / "done.json").stat().st_mtime_ns == round0_done_mtime, (
        "round 0 must not have been redone by the second run_continuous call"
    )
    print("✓ first call ran rounds 0-1, second call ran only round 2 (round 0 untouched)")


def test_load_champion_state_fresh_then_round_trips_after_save(tmp_path):
    print("\nTesting load_champion_state's fresh-init fallback and save round-trip...")
    cfg = _tiny_cfg(tmp_path, run_name="champion_state_test")
    run_dir = tmp_path / "runs" / cfg.run_name
    run_dir.mkdir(parents=True)

    fresh_state, fresh_meta = round_loop.load_champion_state(run_dir, cfg)
    assert fresh_meta == {'round': -1, 'promotions': 0, 'consecutive_failures': 0}
    assert fresh_state  # a real, non-empty state_dict

    round_loop._save_champion_state(run_dir, fresh_state, {'round': 3, 'promotions': 1,
                                                            'consecutive_failures': 2})
    loaded_state, loaded_meta = round_loop.load_champion_state(run_dir, cfg)
    assert loaded_meta == {'round': 3, 'promotions': 1, 'consecutive_failures': 2}
    for key in fresh_state:
        assert (loaded_state[key] == fresh_state[key]).all()
    print("✓ fresh run_dir gives a random-init state_dict + meta; save/load round-trips exactly")


if __name__ == '__main__':
    print("All tests in this file need tmp_path/monkeypatch -- run via pytest.")
