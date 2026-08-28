"""
Shared "SPEC string -> agent" parsing, used by both
parchis.visualization.play_instrumented_game (per-seat, human-facing
replay) and parchis.evaluation.ladder (rung participants, win-rate
comparison) -- one spec grammar instead of two independently-drifting ones.

Grammar:
    checkpoint:<run_dir>[:depth=N]   -- a trained AZ net, search-driven
    heuristic:tuned|default          -- parchis.agents.heuristic's two weight sets
    random                           -- parchis.az.selfplay.random_factory

`parse_spec` returns (kind, params, label) in the same shape
parchis.visualization.instrumented_play.play_and_record's agent_specs
values already use ("search": (evaluator, depth), "heuristic": weights,
"random": None), so both callers share the exact same downstream contract.
`build_factory` turns that into a plain (non-recording) arena-style
factory(game, seat, roll_box) -> choose_move_fn -- ladder.py's use case,
which never needs decision-value recording, unlike instrumented_play.py's
own (separate) recording-factory construction.
"""

from parchis.agents import heuristic
from parchis.az import search as az_search
from parchis.az.agent import NetEvaluator, make_search_agent_factory
from parchis.az.selfplay import random_factory
from parchis.visualization import checkpoint_loading


def parse_spec(spec_str):
    """'checkpoint:<run_dir>[:depth=N]' | 'heuristic:tuned|default' | 'random'
    -> (kind, params, label). Raises ValueError (not argparse-specific --
    CLI callers wrap this themselves) on a malformed spec."""
    parts = spec_str.split(':')
    kind = parts[0]

    if kind == 'checkpoint':
        if len(parts) < 2:
            raise ValueError(
                f"{spec_str!r}: 'checkpoint' needs a run_dir, e.g. checkpoint:runs/my_run"
            )
        run_dir = parts[1]
        depth = None
        for extra in parts[2:]:
            if extra.startswith('depth='):
                depth = int(extra.split('=', 1)[1])
        numpy_net, _num_players, cfg = checkpoint_loading.load_agent_numpy_net(run_dir)
        if depth is None:
            depth = cfg.get('base_depth', az_search.DEFAULT_DEPTH)
        evaluator = NetEvaluator(numpy_net)
        label = f"checkpoint:{run_dir} (search depth={depth})"
        return 'search', (evaluator, depth), label

    if kind == 'heuristic':
        which = parts[1] if len(parts) > 1 else 'tuned'
        if which == 'tuned':
            weights = heuristic.TUNED_WEIGHTS
        elif which == 'default':
            weights = heuristic.DEFAULT_WEIGHTS
        else:
            raise ValueError(
                f"{spec_str!r}: 'heuristic' expects 'tuned' or 'default', got {which!r}"
            )
        return 'heuristic', weights, f"heuristic:{which}"

    if kind == 'random':
        return 'random', None, 'random'

    raise ValueError(
        f"{spec_str!r}: unknown agent kind {kind!r} "
        f"(expected 'checkpoint', 'heuristic', or 'random')"
    )


def build_factory(kind, params):
    """(kind, params) (as returned by parse_spec) -> a plain arena-style
    factory(game, seat, roll_box) -> choose_move_fn. No decision-value
    recording -- see parchis.visualization.instrumented_play for the
    recording variant used by the human-facing replay."""
    if kind == 'search':
        evaluator, depth = params
        return make_search_agent_factory(evaluator, depth)
    if kind == 'heuristic':
        return heuristic.make_heuristic_agent_factory(params)
    if kind == 'random':
        return random_factory
    raise ValueError(f"Unknown agent kind {kind!r} (expected 'search', 'heuristic', or 'random')")
