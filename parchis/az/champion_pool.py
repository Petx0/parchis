"""
Phase 3's opponent pool (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3:
"Pool: opponents sampled from {champion, last 4 promoted, tuned heuristic,
random}. The heuristic anchor is what stops a single-lineage collapse.").

parchis.az.selfplay.generate_round_games always assigns the CURRENT
champion to one (randomly chosen) seat every game -- its decisions are
always recorded. Every OTHER seat samples independently from the pool
built here; when that sample is itself search-capable (the champion
again, a promoted net, or a recent net -- see below), ITS decisions are
recorded too (true self-play -- both sides can contribute), but the two
hand-built anchors (tuned heuristic, random) never are, since neither has
a root_value/move_values breakdown to build a target from. This
guarantees every round game yields at least one recorded seat (never a
wholly-wasted game), while still letting heuristic/random opponents
diversify the data the champion seat is exposed to.

The "last 4 promoted" history is a small, on-disk FIFO of checkpoint
paths (oldest evicted first past 4), persisted separately from
round_loop.py's own single "current champion" pointer -- so a self-play
lineage always has a few historical opponents around even while the
champion itself is being actively retrained, which is what stops the
single-lineage collapse the plan calls out.

"Recent" history (.claude/plans/twinkly-marinating-hinton.md's Phase 3.1)
is a second, larger FIFO covering EVERY round's candidate, promoted or
not -- round_loop.py already saves every round's candidate.pt regardless
of outcome, but until this was added only promoted ones were ever reused
as opponents. With this lineage's actual promotion rate (3 promotions in
68 rounds), the promoted-only pool was nearly static for most of a
44-round plateau: diagnostics (Phase 1.1's flat val_value_loss, Phase
1.2's ladder showing rounds 10-67 are all statistically tied in playing
strength) point at stale opponent diversity, not insufficient opponent
STRENGTH, as a contributor -- so recent history intentionally includes
non-promoted candidates even though they aren't "better," on the theory
that an independently-warm-started, differently-weighted net of similar
strength is still a materially different opponent to self-play against
than an exact copy of the same static champion.
"""

import json
from pathlib import Path

import torch

from parchis.agents import heuristic
from parchis.az.net import AZNet, NumpyAZNet

MAX_PROMOTED_HISTORY = 4
# Larger than MAX_PROMOTED_HISTORY: these are lower-confidence pool
# members (no promotion gate vouched for them), so more of them are kept
# to buy diversity, matching the plan's own reasoning above.
MAX_RECENT_HISTORY = 8


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


def load_recent_history(path):
    """Same shape/contract as load_promoted_history, for the separate
    "every round's candidate" FIFO (see module docstring)."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_recent_history(path, history):
    """Same shape/contract as save_promoted_history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(list(history), f, indent=2)


def append_recent(history, checkpoint_path, max_history=MAX_RECENT_HISTORY):
    """Same FIFO contract as append_promoted, just a larger cap (see
    MAX_RECENT_HISTORY). Unlike append_promoted, round_loop.py calls this
    unconditionally every round -- promoted or not."""
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
    values down for every load, never per-checkpoint metadata. Uses
    load_state_dict_compat rather than a raw load_state_dict: every
    promoted/recent checkpoint saved before the aux head existed (Phase
    4.1) is still a perfectly valid pool member -- NumpyAZNet never reads
    aux_head weights anyway (see net.py's module docstring)."""
    model = AZNet(input_size, num_players, hidden_sizes=hidden_sizes)
    model.load_state_dict_compat(torch.load(model_path, map_location="cpu"))
    model.eval()
    return NumpyAZNet.from_torch(model)


def build_pool(champion_numpy_net, promoted_numpy_nets, recent_numpy_nets=(), tuned_weights=None):
    """Returns (nets, anchor_factories):

    nets: tuple of NumpyAZNet, `(champion_numpy_net, *promoted_numpy_nets,
        *recent_numpy_nets)` -- every search-capable pool member.
        Deliberately left as raw nets, NOT pre-wrapped into an opaque
        arena factory here: generation
        (parchis.az.selfplay.generate_round_games) needs to run
        search.search() itself and see its move_values/root_value
        breakdown for target construction and exploration, which an
        opaque parchis.az.agent.make_search_agent_factory result (as used
        for evaluation/promotion matches, where only the single greedy
        move matters) does not expose. `recent_numpy_nets` (module
        docstring) are sampled uniformly alongside the champion and
        promoted nets -- no weighting toward promoted ones yet; see the
        plan's own note to revisit this once round 1 under the broadened
        pool is visible.
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
    nets = (champion_numpy_net, *promoted_numpy_nets, *recent_numpy_nets)
    anchor_factories = (heuristic.make_heuristic_agent_factory(w), random_factory)
    return nets, anchor_factories
