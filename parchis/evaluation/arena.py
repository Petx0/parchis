"""
Game-level head-to-head arena for comparing pluggable agents (Phase B/C of
the AlphaZero-style plan) -- an MCTS-based agent isn't a saved-weights-only
checkpoint evaluate_agent()'s MaskablePPO.load() can handle (parchis/evaluation/evaluate.py:69),
it needs live search access to game state at inference time. Rather than
bending that existing, SB3-checkpoint-specific stack, this reuses
Game.play_turn() directly -- the same already-tested six-again/bonus/
three-sixes mechanics every other evaluation path in this codebase relies
on -- via per-player choose_move overrides, the same generic extension
point Game.play_turn()/_execute_bonus_move already call. evaluate_agent()/
elo_ladder.py/multiplayer_matrix.py are untouched; this is a new, parallel
tool, not a replacement.

Agents are factories (see parchis/search/agents.py): factory(game, seat,
roll_box) -> choose_move_fn, instantiated fresh per real game so each can
close over that game's own state and dice-roll recorder.
"""

import random

from parchis.game.game import Game
from parchis.evaluation import stats as eval_stats

DEFAULT_MAX_TURNS = 1000


def _install_roll_recorder(game):
    """Wrap game.dice.roll to record the most recent roll -- Player.choose_move
    never receives it (see module docstring), so agents that need it for
    observation construction read it back out of this box. Mirrors
    mcts.py's own _install_roll_recorder exactly (duplicated rather than
    imported: it's 6 lines, and importing an mcts.py-private helper into
    the evaluation package would be an odd cross-package dependency for
    something this small)."""
    box = {"last_roll": None}
    original_roll = game.dice.roll

    def recording_roll():
        value = original_roll()
        box["last_roll"] = value
        return value

    game.dice.roll = recording_roll
    return box


def play_one_game(agent_factories, num_players=2, max_turns=DEFAULT_MAX_TURNS, seed=None,
                   return_piece_status=False):
    """
    agent_factories: {seat_index: factory(game, seat, roll_box) -> choose_move_fn}.
        A seat not present keeps Player.choose_move's own default (random).
    seed: seeds Python's global `random` module before constructing Game()
        (which itself rolls dice for color/starting-player determination --
        see Game.__init__) and for the whole game's dice sequence. Pass
        None to use whatever ambient random state already exists.
    return_piece_status: False (default) preserves this function's
        original single-value return exactly, for every existing caller.
        True additionally returns each seat's own final piece-finished
        status (parchis.az.selfplay's aux-target computation, Phase 4.1 --
        see its module docstring); no other caller needs this.

    Returns: the winning seat's index (int), or None if max_turns was hit
        without a winner (extremely unlikely at max_turns=1000 given real
        games run ~150-300 turns; a safety cap, not an expected outcome).
        If return_piece_status=True, returns
        (winner_seat_or_None, piece_status) instead, where piece_status is
        {seat: [bool, bool, bool, bool]} (piece_id-indexed, True = that
        piece finished by the time the game ended or max_turns was hit).
    """
    if seed is not None:
        random.seed(seed)
    game = Game(num_players=num_players)
    roll_box = _install_roll_recorder(game)
    for seat, factory in agent_factories.items():
        game.players[seat].choose_move = factory(game, seat, roll_box)

    turns = 0
    while not game.game_over and turns < max_turns:
        game.play_turn()
        turns += 1

    winner_seat = None if game.winner is None else game.players.index(game.winner)
    if not return_piece_status:
        return winner_seat
    piece_status = {
        seat: [piece.finished for piece in player.pieces]
        for seat, player in enumerate(game.players)
    }
    return winner_seat, piece_status


def play_match(agent_a_factory, agent_b_factory, n_games, num_players=2,
                max_turns=DEFAULT_MAX_TURNS, seed=42):
    """Play n_games between two agents, randomizing which seat agent_a
    occupies each game (mirrors ParchisEnv.agent_player_idx's own
    randomization -- avoids a fixed-seat bias skewing the result).

    Returns: dict with wins_a, n_games, win_rate_a, win_rate_a_ci (Wilson,
    reusing parchis.evaluation.stats -- the same formula every other
    evaluation tool in this codebase uses).
    """
    rng = random.Random(seed)
    wins_a = 0
    for _ in range(n_games):
        a_seat = rng.randrange(num_players)
        agent_factories = {
            seat: (agent_a_factory if seat == a_seat else agent_b_factory)
            for seat in range(num_players)
        }
        game_seed = rng.randrange(2**31)
        winner_seat = play_one_game(agent_factories, num_players=num_players,
                                     max_turns=max_turns, seed=game_seed)
        if winner_seat == a_seat:
            wins_a += 1

    win_rate_a = wins_a / n_games
    ci = eval_stats.wilson_score_interval(wins_a, n_games)
    return {
        "wins_a": wins_a,
        "n_games": n_games,
        "win_rate_a": win_rate_a,
        "win_rate_a_ci": ci,
    }
