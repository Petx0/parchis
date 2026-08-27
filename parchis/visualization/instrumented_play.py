"""
Plays one real game with real agents (search-based, heuristic, or random)
AND logs it -- the capability neither existing "play a game" path in this
repo offers on its own:

- parchis.evaluation.arena.play_one_game plays with real agents, but its
  Game is built without a logger (no move-by-move JSON comes out of it).
- parchis.visualization.demo_visualization attaches a GameLogger, but only
  ever plays with Player.choose_move's random default.

Mirrors demo_visualization.py's own manual game-loop pattern (Game(...,
logger=GameLogger(...)), log_game_start/play_turn loop/log_game_end,
save_to_file) rather than arena.play_one_game, since arena's Game is
logger-less and has no per-seat recorder hook -- and duplicates arena.py's
_install_roll_recorder rather than importing its private name, the same
small-duplication tradeoff arena.py itself already made against mcts.py's
identical helper.
"""

import random

from parchis.agents import heuristic
from parchis.agents.decision_recorder import DecisionRecorder
from parchis.az.agent import make_recording_search_agent_factory
from parchis.game.game import Game
from parchis.utils.logger import GameLogger
from parchis.visualization import agentinfo_io

DEFAULT_MAX_TURNS = 1000


def _install_roll_recorder(game):
    """See parchis.evaluation.arena._install_roll_recorder -- identical,
    deliberately duplicated (that module's own docstring explains why: it's
    6 lines, and importing a private helper across an unrelated package
    boundary for something this small is an odd dependency to take on)."""
    box = {"last_roll": None}
    original_roll = game.dice.roll

    def recording_roll():
        value = original_roll()
        box["last_roll"] = value
        return value

    game.dice.roll = recording_roll
    return box


def _build_factory(kind, params, seat, recorders_by_seat, agent_labels):
    if kind == "search":
        evaluator, depth = params
        recorder = DecisionRecorder()
        recorders_by_seat[seat] = recorder
        agent_labels[seat] = f"search (depth={depth})"
        return make_recording_search_agent_factory(evaluator, depth, recorder)
    if kind == "heuristic":
        weights = params
        recorder = DecisionRecorder()
        recorders_by_seat[seat] = recorder
        agent_labels[seat] = "heuristic"
        return heuristic.make_recording_heuristic_agent_factory(weights, recorder=recorder)
    if kind == "random":
        agent_labels[seat] = "random"
        return None
    raise ValueError(f"Unknown agent kind {kind!r} (expected 'search', 'heuristic', or 'random')")


def play_and_record(agent_specs, num_players=2, max_turns=DEFAULT_MAX_TURNS, seed=None,
                     log_dir="logs"):
    """
    agent_specs: {seat: (kind, params)} where kind/params is one of:
        ("search", (evaluator, depth))   -- parchis.az.agent.NetEvaluator or
                                             heuristic_position_evaluator, and
                                             a search depth >= 1
        ("heuristic", weights_or_None)   -- e.g. heuristic.TUNED_WEIGHTS
        ("random", None)
    A seat not present in agent_specs keeps Player.choose_move's own random
    default (same convention as arena.play_one_game).

    Returns (log_filepath, agentinfo_filepath_or_None) -- the latter is
    None iff no seat was instrumented (every seat was "random" or simply
    absent), matching agentinfo_io.load_agentinfo's own None-means-nothing-
    to-show contract.
    """
    if seed is not None:
        random.seed(seed)

    logger = GameLogger(log_dir=log_dir)
    game = Game(num_players=num_players, logger=logger)
    roll_box = _install_roll_recorder(game)

    recorders_by_seat = {}
    agent_labels = {}
    for seat, (kind, params) in agent_specs.items():
        factory = _build_factory(kind, params, seat, recorders_by_seat, agent_labels)
        if factory is not None:
            game.players[seat].choose_move = factory(game, seat, roll_box)

    logger.log_game_start(game.players, game.starting_player, game.initial_dice_rolls)

    turns = 0
    while not game.game_over and turns < max_turns:
        game.play_turn()
        turns += 1

    if game.game_over:
        logger.log_game_end(game.winner, game.turn_number)
    log_path = logger.save_to_file()

    agentinfo_path = None
    if recorders_by_seat:
        agentinfo_path = agentinfo_io.save_agentinfo(
            recorders_by_seat, agent_labels, log_path, num_players=num_players,
        )
    return log_path, agentinfo_path
