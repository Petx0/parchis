"""
Phase 3's opponent pool (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3:
"Pool: opponents sampled from {champion, last 4 promoted, tuned heuristic,
random}. The heuristic anchor is what stops a single-lineage collapse.").

parchis.az.selfplay.generate_round_games always assigns the CURRENT
champion to one (randomly chosen) seat every game -- its decisions are
always recorded. Every OTHER seat samples independently from the pool
built here; when that sample is itself search-capable (the champion
again, or a promoted net), ITS decisions are recorded too (true
self-play -- both sides can contribute), but the two hand-built anchors
(tuned heuristic, random) never are, since neither has a root_value/
move_values breakdown to build a target from. This guarantees every round
game yields at least one recorded seat (never a wholly-wasted game),
while still letting heuristic/random opponents diversify the data the
champion seat is exposed to.

The "last 4 promoted" history is a small, on-disk FIFO of checkpoint
paths (oldest evicted first past 4), persisted separately from
round_loop.py's own single "current champion" pointer -- so a self-play
lineage always has a few historical opponents around even while the
champion itself is being actively retrained, which is what stops the
single-lineage collapse the plan calls out.
"""

import json
from pathlib import Path

import torch

from parchis.agents import heuristic
from parchis.az.net import AZNet, NumpyAZNet

MAX_PROMOTED_HISTORY = 4


def load_promoted_history(path):
    """Returns a list of checkpoint path strings (oldest-first, i.e. most-
    recently-promoted last), or [] if `path` doesn't exist yet (a fresh
    run with no promotions so far)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_promoted_history(path, history):
    """Writes `history` (a list of path strings) to `path` as JSON,
    creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(history), f, indent=2)


def append_promoted(history, checkpoint_path, max_history=MAX_PROMOTED_HISTORY):
    """Returns a NEW list (does not mutate `history`) with `checkpoint_path`
    appended and the oldest entries evicted past `max_history` -- a plain
    FIFO cap, oldest-first / most-recent-last."""
    updated = list(history) + [str(checkpoint_path)]
    return updated[-max_history:]


def load_numpy_net(model_path, input_size, num_players, hidden_sizes):
    """Loads a saved AZNet state_dict (parchis.az.train.save_checkpoint's
    model.pt) back into a fresh AZNet of the given shape, then wraps it as
    a NumpyAZNet for inference (see parchis/az/net.py's module docstring
    for why search always runs the numpy path, not torch, at inference
    time). `input_size`/`num_players`/`hidden_sizes` must match what the
    checkpoint was actually trained with -- every checkpoint in one Phase 3
    lineage shares the same shape, so round_loop.py passes its own config's
    values down for every load, never per-checkpoint metadata."""
    model = AZNet(input_size, num_players, hidden_sizes=hidden_sizes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return NumpyAZNet.from_torch(model)


def build_pool(champion_numpy_net, promoted_numpy_nets, tuned_weights=None):
    """Returns (nets, anchor_factories):

    nets: tuple of NumpyAZNet, `(champion_numpy_net, *promoted_numpy_nets)`
        -- every search-capable pool member. Deliberately left as raw nets,
        NOT pre-wrapped into an opaque arena factory here: generation
        (parchis.az.selfplay.generate_round_games) needs to run
        search.search() itself and see its move_values/root_value
        breakdown for target construction and exploration, which an
        opaque parchis.az.agent.make_search_agent_factory result (as used
        for evaluation/promotion matches, where only the single greedy
        move matters) does not expose.
    anchor_factories: tuple of plain arena-style
        factory(game, seat, roll_box) -> choose_move_fn callables --
        (tuned heuristic, random). Never search-capable, so never
        recordable; used as-is, purely for opponent diversity.
    """
    # Deferred import: parchis.az.selfplay imports THIS module (to build
    # the pool for generate_round_games), so a module-level import here
    # the other direction would be circular. By the time build_pool() is
    # actually CALLED (never at import time), selfplay.py is already
    # fully loaded, so this resolves fine.
    from parchis.az.selfplay import random_factory

    w = heuristic.TUNED_WEIGHTS if tuned_weights is None else tuned_weights
    nets = (champion_numpy_net, *promoted_numpy_nets)
    anchor_factories = (heuristic.make_heuristic_agent_factory(w), random_factory)
    return nets, anchor_factories
