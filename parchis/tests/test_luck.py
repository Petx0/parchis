"""C8 luck-adjusted scoring (parchis/evaluation/luck.py): the tracker never changes the game,
each roll's luck has exactly zero mean over the six faces, the six-streak it hands the
evaluator matches the turn's rolls, and the duplicate-match plumbing reduces to the raw
result when the evaluator is roll-blind."""
import math

import numpy as np

from parchis.agents import heuristic
from parchis.evaluation import arena, duplicate
from parchis.az import encoding
from parchis.evaluation.luck import LuckTracker, NUM_FACES, face_values
from parchis.game.game import Game


class _ConstantEvaluator:
    """Roll-blind: V = 1/P for every seat, so every luck term is exactly 0."""

    def __init__(self, num_players=2):
        self.num_players = num_players

    def encode(self, game, seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        return np.zeros(1, dtype=np.float32)

    def evaluate_batch(self, rows, seats):
        return np.full((len(rows), self.num_players), 1.0 / self.num_players)


class _RollSensitiveEvaluator:
    """Two players; the mover's value is a deterministic function of (roll, streak)."""

    def encode(self, game, seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        return np.array([roll, seat, consecutive_sixes], dtype=np.float32)

    def evaluate_batch(self, rows, seats):
        rows = np.asarray(rows)
        out = np.empty((len(rows), 2))
        for i, (roll, seat, streak) in enumerate(rows):
            p = 0.3 + 0.05 * roll - 0.04 * streak
            out[i, int(seat)] = p
            out[i, 1 - int(seat)] = 1.0 - p
        return out


class _SpyEvaluator(_RollSensitiveEvaluator):
    """Records (turn_number, mover, streak) once per dice roll (on the face-1 row)."""

    def __init__(self):
        self.calls = []

    def encode(self, game, seat, roll=None, pending_bonus=None, consecutive_sixes=0):
        if roll == 1:
            self.calls.append((game.turn_number, game.current_player_idx, seat, consecutive_sixes))
        return super().encode(game, seat, roll=roll, consecutive_sixes=consecutive_sixes)


def test_tracker_changes_neither_dice_nor_result_and_luck_sums_to_zero_across_seats():
    tuned = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)
    factories = {0: tuned, 1: default}
    for seed in range(4):
        plain = arena.play_one_game(factories, num_players=2, seed=seed)
        luck_out = {}
        tracked = arena.play_one_game(factories, num_players=2, seed=seed,
                                      luck_evaluator=_RollSensitiveEvaluator(), luck_out=luck_out)
        assert tracked == plain
        assert luck_out['n_rolls'] > 20
        assert luck_out['luck'].shape == (2,)
        # value vectors sum to 1 for every face, so the seats' lucks are exact opposites
        assert math.isclose(luck_out['luck'][0], -luck_out['luck'][1], abs_tol=1e-9)
        assert abs(luck_out['luck'][0]) > 0.0


def test_each_roll_has_exactly_zero_mean_luck_over_the_six_faces():
    game = Game(num_players=2)
    total = np.zeros(2)
    for face in range(1, NUM_FACES + 1):
        tracker = LuckTracker(game, _RollSensitiveEvaluator())
        tracker._original_roll = lambda face=face: face
        assert tracker._roll() == face
        total += tracker.luck
        game.dice.roll = tracker._original_roll  # uninstall before the next tracker
    assert np.allclose(total, 0.0, atol=1e-12)


def test_six_streak_handed_to_the_evaluator_counts_the_turns_sixes():
    game = Game(num_players=2)
    spy = _SpyEvaluator()
    LuckTracker(game, spy)
    turns = []
    while len(turns) < 60 and not game.game_over:
        turns.append((game.turn_number + 1, game.current_player_idx, game.play_turn()))
    assert any(len(info.rolls) >= 2 for _, _, info in turns)  # at least one six-chain happened
    for turn_number, seat, info in turns:
        calls = [c for c in spy.calls if c[0] == turn_number]
        assert len(calls) == len(info.rolls)
        for k, (_, mover, observer, streak) in enumerate(calls):
            assert mover == seat and observer == seat
            assert streak == k  # every earlier roll of this turn was a six
        # the k-th roll is preceded by exactly k sixes
        rolls = [r.dice_roll for r in info.rolls]
        assert all(r == 6 for r in rolls[:-1])


def test_duplicate_match_with_a_roll_blind_evaluator_reports_the_raw_result():
    tuned = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)
    plain = duplicate.play_duplicate_match(tuned, default, n_pairs=6, num_players=2, seed=11)
    lucky = duplicate.play_duplicate_match(tuned, default, n_pairs=6, num_players=2, seed=11,
                                           luck_evaluator=_ConstantEvaluator(2))
    assert lucky['wins_a'] == plain['wins_a'] and lucky['groups'][0]['winners'] == plain['groups'][0]['winners']
    assert lucky['adjusted_groups'] == lucky['raw_groups']
    assert lucky['adjusted_score_a'] == lucky['win_rate_a']
    assert lucky['n_groups'] == 6 and lucky['mean_abs_luck'] == 0.0
    assert lucky['luck_variance_multiplier'] == 1.0 or math.isnan(lucky['luck_variance_multiplier'])
    assert 'adjusted_groups' not in plain


def test_duplicate_match_adjusted_scores_subtract_each_games_luck():
    tuned = heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS)
    default = heuristic.make_heuristic_agent_factory(heuristic.DEFAULT_WEIGHTS)
    r = duplicate.play_duplicate_match(tuned, default, n_pairs=4, num_players=2, seed=5,
                                       luck_evaluator=_RollSensitiveEvaluator())
    for g in r['groups']:
        assert len(g['a_adjusted']) == 2
        for a_seat, adjusted in enumerate(g['a_adjusted']):
            outcome = 1.0 if g['winners'][a_seat] == a_seat else 0.0
            assert adjusted != outcome  # luck is never exactly zero with a roll-sensitive evaluator
    assert len(r['adjusted_groups']) == 4 and len(r['adjusted_score_a_ci']) == 2
    assert r['mean_abs_luck'] > 0.0


def test_summarise_luck_groups_multiplier_is_variance_ratio():
    raw = [1.0, 0.0, 0.5, 1.0, 0.0, 0.5]
    adj = [0.6, 0.4, 0.5, 0.6, 0.4, 0.5]
    s = duplicate.summarise_luck_groups(raw, adj)
    assert math.isclose(s['raw_score_a'], 0.5) and math.isclose(s['adjusted_score_a'], 0.5)
    assert math.isclose(s['luck_variance_multiplier'], np.var(raw, ddof=1) / np.var(adj, ddof=1))
    assert s['adjusted_score_a_ci'][1] - s['adjusted_score_a_ci'][0] < s['raw_score_a_ci_groups'][1] - s['raw_score_a_ci_groups'][0]


def test_encode_all_rolls_equals_six_separate_encodes():
    import random
    random.seed(3)
    checked = 0
    for num_players in (2, 3, 4):
        game = Game(num_players=num_players)
        for _ in range(120):
            if game.game_over:
                break
            game.play_turn()
            for seat in range(num_players):
                for streak in (0, 1, 2):
                    fast = encoding.encode_all_rolls(game, seat, consecutive_sixes=streak)
                    slow = np.stack([encoding.encode(game, seat, roll=f, consecutive_sixes=streak)
                                     for f in range(1, 7)])
                    assert fast.dtype == slow.dtype and fast.shape == slow.shape
                    assert np.array_equal(fast, slow)
                    checked += 1
    assert checked > 300


def test_face_values_dispatches_to_face_values_or_duck_types():
    game = Game(num_players=2)

    class _Direct:
        def face_values(self, g, mover, streak):
            return np.full((6, 2), 0.25)

    assert np.array_equal(face_values(_Direct(), game, 0, 0), np.full((6, 2), 0.25))
    duck = face_values(_RollSensitiveEvaluator(), game, 1, 2)
    assert duck.shape == (6, 2) and np.allclose(duck.sum(axis=1), 1.0)
    assert np.allclose(duck[:, 1], [0.3 + 0.05 * f - 0.08 for f in range(1, 7)])


def test_net_luck_evaluator_matches_the_duck_typed_net_evaluator_path():
    """One encode + one-hot patching (NetLuckEvaluator) gives exactly the values of six separate
    encodes through the same NetEvaluator, on real positions with a real (committed) net."""
    import random
    from pathlib import Path
    from parchis.az import champion_pool
    from parchis.az.agent import NetEvaluator
    from parchis.evaluation.luck import NetLuckEvaluator
    ckpt = Path(__file__).resolve().parents[2] / "runs/selfplay_2p_v1_champion/model.pt"
    net_ev = NetEvaluator(champion_pool.load_numpy_net(str(ckpt), 216, 2, (256, 256)))
    fast = NetLuckEvaluator(net_ev)
    random.seed(9)
    game = Game(num_players=2)
    checked = 0
    for _ in range(80):
        if game.game_over:
            break
        game.play_turn()
        for streak in (0, 1, 2):
            a = face_values(fast, game, game.current_player_idx, streak)
            b = face_values(net_ev, game, game.current_player_idx, streak)
            assert a.shape == (6, 2) and np.array_equal(a, b)
            checked += 1
    assert checked > 100
