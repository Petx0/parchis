"""
Load a trained checkpoint's NumpyAZNet from a run directory, for the
visualization CLI (parchis/visualization/play_instrumented_game.py) -- NOT
a new loading mechanism, just parchis.az.champion_pool.load_numpy_net
wired up with the shape info every run directory already carries in its
own config.json, so the CLI doesn't need the user to type in
num_players/hidden_sizes by hand.
"""

import json
from pathlib import Path

from parchis.az import encoding
from parchis.az.champion_pool import load_numpy_net

# Checkpoint filenames this project actually produces, in lookup order.
# "model.pt" is written by parchis.az.train.save_checkpoint (Phase 0-2
# bootstrap runs); "champion.pt" is round_loop.py's own name for a Phase 3
# run's current champion, e.g. useful for pointing at an IN-PROGRESS run
# directory rather than a checkpoint copy filed under the project's own
# runs/<name>_champion/ (see docs/AGENT_REBUILD_PLAN.md's Phase 3 section).
_CHECKPOINT_FILENAMES = ("model.pt", "champion.pt")


def _find_checkpoint_file(run_dir):
    run_dir = Path(run_dir)
    for name in _CHECKPOINT_FILENAMES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No checkpoint file ({', '.join(_CHECKPOINT_FILENAMES)}) found in {run_dir}"
    )


def load_run_config(run_dir):
    """Returns the run directory's config.json as a dict."""
    config_path = Path(run_dir) / "config.json"
    with open(config_path) as f:
        return json.load(f)


def load_agent_numpy_net(run_dir):
    """Loads a run directory's checkpoint into a NumpyAZNet, reading
    num_players/hidden_sizes from that same run's config.json (every
    checkpoint in one lineage shares one shape -- see
    champion_pool.load_numpy_net's own docstring). Returns
    (numpy_net, num_players, config_dict)."""
    cfg = load_run_config(run_dir)
    num_players = cfg["num_players"]
    hidden_sizes = tuple(cfg["hidden_sizes"])
    input_size = encoding.encoding_size(num_players)
    model_path = _find_checkpoint_file(run_dir)
    numpy_net = load_numpy_net(model_path, input_size, num_players, hidden_sizes)
    return numpy_net, num_players, cfg
