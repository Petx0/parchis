"""Experiment registry (docs/EXPERIMENT_REGISTRY.md, PILA-7).

Every run directory carries one `meta.json` run record: git SHA, full config, seeds, checkpoints
(with sha256) and every evaluation result played on its checkpoints. Training
(parchis.az.round_loop) and evaluation (scripts/v2/h2h.py, scripts/v2/ladder.py,
scripts/eval/protocol.py) write it automatically; `python -m parchis.registry compare` renders all
runs side by side.
"""
from parchis.registry.record import (  # noqa: F401
    SCHEMA, META_NAME, REPO, begin_run, update_run, record_eval, find_run_dir, load_meta, save_meta,
    allocate_training_base, apply_seed_base, file_sha256, git_state, rel,
)
