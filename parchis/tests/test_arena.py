#!/usr/bin/env python3
"""
Tests for parchis/evaluation/arena.py and parchis/search/agents.py.

Drives real games via Game.play_turn() (arena's own approach, not the
Gym-API env) with both dummy callables and a tiny real trained checkpoint
(mirrors test_elo_ladder.py's "tiny real model, no mocks" convention).
"""

import pytest
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from parchis.rl.env import ParchisEnv
from parchis.evaluation import arena
from parchis.search.agents import make_plain_ppo_agent_factory, make_mcts_ppo_agent_factory


def _mask_fn(env):
    return env.unwrapped._get_info()['action_masks']


def _always_first_legal_move_factory(game, seat, roll_box):
    def choose_move(legal_moves):
        return legal_moves[0] if legal_moves else None
    return choose_move


@pytest.fixture(scope="module")
def tiny_model():
    """A tiny real trained checkpoint (mirrors test_elo_ladder.py's
    established convention) -- training is the expensive part, reused
    across every test in this file."""
    env = ParchisEnv(num_players=2)
    env = ActionMasker(env, _mask_fn)
    model = MaskablePPO("MlpPolicy", env, verbose=0, tensorboard_log=None,
                         n_steps=64, batch_size=32)
    model.learn(total_timesteps=200)
    env.close()
    return model


def test_play_one_game_with_dummy_agents_returns_a_valid_winner_seat():
    print("\nTesting play_one_game() with dummy always-first-legal-move agents...")
    winner_seat = arena.play_one_game(
        {0: _always_first_legal_move_factory, 1: _always_first_legal_move_factory},
        num_players=2, seed=0,
    )
    assert winner_seat in (0, 1), f"Expected a valid winner seat, got {winner_seat}"
    print(f"✓ Game completed, winner seat={winner_seat}")


def test_play_one_game_return_piece_status_matches_the_actual_winner():
    """return_piece_status=True (Phase 4.1's aux-target source,
    parchis.az.selfplay) must give every seat's own final piece-finished
    flags -- and since Player.has_won() requires ALL 4 pieces finished,
    the winning seat's own 4 flags must be all True."""
    print("\nTesting play_one_game(return_piece_status=True) against the real winner...")
    for seed in range(5):
        winner_seat, piece_status = arena.play_one_game(
            {0: _always_first_legal_move_factory, 1: _always_first_legal_move_factory},
            num_players=2, seed=seed, return_piece_status=True,
        )
        assert set(piece_status) == {0, 1}
        for seat, flags in piece_status.items():
            assert len(flags) == 4 and all(isinstance(f, bool) for f in flags)
        if winner_seat is not None:
            assert all(piece_status[winner_seat]), (
                f"seed={seed}: the winner must have all 4 pieces finished, got {piece_status}"
            )
    print("✓ piece_status shape is correct and the winner always has all 4 pieces finished")


def test_play_one_game_default_return_is_unaffected_by_return_piece_status_existing():
    """The default (return_piece_status omitted) must still return a bare
    winner_seat -- every existing caller of play_one_game (play_match,
    duplicate.py, ladder.py, selfplay.generate_games, etc.) depends on
    this NOT becoming a tuple just because the parameter now exists."""
    print("\nTesting play_one_game's default return is still a bare winner_seat...")
    result = arena.play_one_game(
        {0: _always_first_legal_move_factory, 1: _always_first_legal_move_factory},
        num_players=2, seed=0,
    )
    assert result is None or isinstance(result, int)
    print(f"✓ default return is a bare value ({result!r}), not a tuple")


def test_play_match_win_rate_and_ci_are_well_formed():
    print("\nTesting play_match() produces a well-formed win rate + CI...")
    result = arena.play_match(
        _always_first_legal_move_factory, _always_first_legal_move_factory,
        n_games=20, num_players=2, seed=1,
    )
    assert 0 <= result["wins_a"] <= result["n_games"] == 20
    assert 0.0 <= result["win_rate_a"] <= 1.0
    lo, hi = result["win_rate_a_ci"]
    assert 0.0 <= lo <= result["win_rate_a"] <= hi <= 1.0
    print(f"✓ win_rate_a={result['win_rate_a']:.2f} CI=[{lo:.2f}, {hi:.2f}]")


def test_plain_ppo_agent_completes_real_games_against_random(tiny_model):
    print("\nTesting make_plain_ppo_agent_factory() drives real games to completion...")
    ppo_factory = make_plain_ppo_agent_factory(tiny_model, num_players=2)
    result = arena.play_match(ppo_factory, _always_first_legal_move_factory,
                               n_games=6, num_players=2, seed=2)
    assert result["n_games"] == 6
    print(f"✓ {result['wins_a']}/{result['n_games']} games completed cleanly, "
          f"win_rate_a={result['win_rate_a']:.2f}")


def test_mcts_ppo_agent_completes_real_games_against_random(tiny_model):
    print("\nTesting make_mcts_ppo_agent_factory() (Phase B: search on the "
          "existing checkpoint) drives real games to completion...")
    mcts_factory = make_mcts_ppo_agent_factory(tiny_model, num_players=2, n_simulations=10)
    result = arena.play_match(mcts_factory, _always_first_legal_move_factory,
                               n_games=3, num_players=2, seed=3)
    assert result["n_games"] == 3
    print(f"✓ {result['wins_a']}/{result['n_games']} games completed cleanly, "
          f"win_rate_a={result['win_rate_a']:.2f}")


def test_mcts_ppo_agent_survives_its_own_bonus_chains(tiny_model):
    """Regression test: the MCTS-searching seat's OWN bonus-chain moves
    (capture/finish bonuses within one of its turns) must fall through to
    the default random policy, not recurse back into mcts.search() on an
    already-simulated copy. This previously raised RecursionError --
    root's children deepcopy the REAL game, whose agent player's
    choose_move IS the mcts.search()-backed closure the arena installed,
    and without an explicit reset (see _expand's fix) that stale override
    survived the deepcopy and got called again inside the simulation's own
    handle_bonus_moves(). Enough games at a real simulation budget to make
    a bonus chain very likely to occur for the searching seat."""
    print("\nTesting MCTS agent doesn't recurse when it captures/finishes "
          "(triggering its own bonus chain) mid-search...")
    mcts_factory = make_mcts_ppo_agent_factory(tiny_model, num_players=2, n_simulations=30)
    result = arena.play_match(mcts_factory, _always_first_legal_move_factory,
                               n_games=10, num_players=2, seed=99)
    assert result["n_games"] == 10
    print(f"✓ {result['wins_a']}/{result['n_games']} games completed cleanly "
          f"(no RecursionError), win_rate_a={result['win_rate_a']:.2f}")


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v', '-s']))
