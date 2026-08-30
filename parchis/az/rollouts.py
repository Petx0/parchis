"""
Rollout-refined value-target estimation
(.claude/plans/twinkly-marinating-hinton.md Phase 2.2).

Phase 1.4's diagnostic (a fresh 400-decision sample from champion-vs-
champion self-play at base_depth) found search.py's root_value
systematically OVERESTIMATES the mover's own win probability by ~2.9
points relative to an independent rollout estimate (mean diff +0.0287,
stderr 0.0068, one-sample t-test p<0.0001) -- consistent with
parchis.az.targets.blend_value_target's existing 0.5*outcome +
0.5*root_value formula partly bootstrapping its training target from the
net's own biased self-estimate, rather than correcting it with fresh
ground truth every round.

estimate_rollout_value gives an independent, non-net alternative for that
bootstrap term: play out n_rollouts continuations from the EXACT decision
point (via Game.snapshot()/restore(), not a fresh Game()) using the tuned
heuristic -- fast, no search, no shared net bias -- on every seat, and
average the resulting mover-relative outcomes. This mirrors Tesauro &
Galperin's backgammon "rollout" policy-improvement idea (see
docs/AZ_DESIGN.md's literature-review section): a cheap Monte-Carlo
refinement of a value estimate, layered on top of an existing evaluator
rather than replacing it.

Deliberately NOT layered into search.py itself, and NOT applied to every
recorded decision: running a full rollout for every decision would
multiply generation cost by n_rollouts, repeating exactly the mistake the
now-retired escalation mechanism made (see round_loop.py's module
docstring) -- spending real compute on search-side effort without first
confirming the payoff. Callers (parchis.az.selfplay.generate_round_games,
gated by SelfPlayRoundConfig.value_target_mode /
rollout_target_fraction) apply this to only a small, randomly-sampled
subset of decisions per round.

Runs entirely under parchis.search.isolated_random's save/restore of
Python's global `random` module state -- Dice.roll() (parchis/game/dice.py)
draws from that global module directly (see isolated_random's own
docstring), so a rollout's own internal re-seeding and dice draws would
otherwise silently perturb the REAL game's subsequent dice sequence the
moment control returns to it after this function call. This is the exact
failure mode parchis/search/mcts.py already solved for its own simulated
rollouts; reused here rather than re-solved.
"""

import random

import numpy as np

from parchis.agents import heuristic as heuristic_module
from parchis.search.isolated_random import isolated_random

DEFAULT_MAX_TURNS = 500


def estimate_rollout_value(game, snapshot, mover_seat, n_rollouts, rng,
                            max_turns=DEFAULT_MAX_TURNS, tuned_weights=None):
    """Restores `game` to `snapshot` and plays out `n_rollouts` independent
    continuations with the tuned heuristic on every seat, returning the
    mean mover-relative outcome vector (index 0 = did the ORIGINAL mover
    go on to win, 1/num_players each for a truncated rollout -- matching
    parchis.az.encoding's own _ordered_seats convention and
    parchis.az.search's own mover-relative contract, so the result can be
    used as a drop-in alternative to search()'s root_value).

    `game` is left restored to `snapshot` when this returns, and every
    seat's `choose_move` is restored to whatever it was before this call
    -- both game STATE (board/pieces, via snapshot()/restore()) and the
    monkey-patched `choose_move` methods this function temporarily
    installs are undone, since Game.snapshot() only covers the former and
    leaving the latter overwritten would silently swap the real game's
    remaining turns over to the heuristic for every seat, not just this
    one rollout call. Restoration happens in a `finally` block, so it
    still runs if a rollout game raises partway through.

    `rng`: a random.Random instance, reused across calls by the caller so
    a whole generation run stays reproducible from one top-level seed
    (matching every other RNG-consuming function in this package's
    convention -- see parchis.az.selfplay.generate_round_games's own
    `rng`/`dirichlet_rng`). `rng` itself picks each rollout's own seed;
    the resulting re-seeding of Python's global `random` module (which
    Game's dice draws read from) is entirely undone before this function
    returns -- see isolated_random's own docstring and this module's.
    """
    weights = heuristic_module.TUNED_WEIGHTS if tuned_weights is None else tuned_weights
    factory = heuristic_module.make_heuristic_agent_factory(weights)
    num_players = game.num_players
    outcomes = np.empty((n_rollouts, num_players), dtype=np.float64)

    original_choose_move = [game.players[seat].choose_move for seat in range(num_players)]
    try:
        with isolated_random(rng.randrange(2 ** 31)):
            for i in range(n_rollouts):
                game.restore(snapshot)
                random.seed(rng.randrange(2 ** 31))
                for seat in range(num_players):
                    game.players[seat].choose_move = factory(game, seat, {})

                turns = 0
                while not game.game_over and turns < max_turns:
                    game.play_turn()
                    turns += 1

                if game.winner is not None:
                    winner_seat = game.players.index(game.winner)
                    absolute = np.zeros(num_players, dtype=np.float64)
                    absolute[winner_seat] = 1.0
                else:
                    absolute = np.full(num_players, 1.0 / num_players, dtype=np.float64)
                outcomes[i] = np.roll(absolute, -mover_seat)
    finally:
        game.restore(snapshot)
        for seat in range(num_players):
            game.players[seat].choose_move = original_choose_move[seat]

    return outcomes.mean(axis=0).astype(np.float32)
