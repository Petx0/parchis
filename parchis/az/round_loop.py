"""
Phase 3's continuous self-play loop (docs/AGENT_REBUILD_PLAN.md Part 3
Phase 3, "the main event, continuous"). Each round:

  1. Generate: parchis.az.selfplay.generate_round_games with the current
     champion (root exploration: temperature anneal + Dirichlet noise),
     written to disk as shards (parchis.az.champion_pool's pool, one
     seat always the champion).
  2. Replay buffer: this round's shards + the previous
     (replay_window_rounds - 1) rounds' -- a recency window, not
     unbounded accumulation (the specific mistake docs/SEARCH_MCTS.md
     documents).
  3. Train: parchis.az.train.bootstrap_train_sharded, WARM-STARTED from
     the current champion's weights (init_state_dict), for a small capped
     number of epochs -- nudging, not re-learning from scratch.
  4. Promote: parchis.evaluation.duplicate.play_duplicate_match, candidate
     vs. champion, >= promotion_n_pairs duplicate pairs, BOTH sides always
     evaluated at base_depth (see the escalation-confound fix below);
     promoted only if the Wilson lower bound clears 50%.
  5. Escalate: after `escalate_after_failures` consecutive non-promotions
     at base_depth, the NEXT round alone GENERATES at escalation_depth
     instead (expert iteration -- stronger training data), then reverts.

Every round is checkpointed under runs/<run_name>/rounds/round_NNNN/,
finishing with a done.json sentinel written only once every step above
succeeded -- run_continuous resumes at the first round WITHOUT that
sentinel, so a crash mid-round costs at most one round's work, never
silently corrupts or skips one.

Bug found & fixed (2026-08-27, after a full 40-round run): the promotion
match originally evaluated BOTH candidate and champion at whatever depth
that round GENERATED at -- so an escalated round's promotion test pitted
the candidate against a champion that was ALSO newly searching deeper.
That confounds two different questions: "did training on depth-2-
generated data make the candidate's underlying net better" (what
escalation is actually for) vs. "is the old champion also playing better
right now" (an artifact of the test itself, unrelated to the candidate).
Across the full run this was a real cost, not a theoretical one: all 9
escalated rounds failed to promote under the confounded test, consuming
~79% of the run's total wall-clock time for zero promotions, while the 31
base-depth rounds (evaluated the same depth they generated at, so never
confounded) promoted 3 times. Fixed by evaluating promotion ALWAYS at
base_depth, decoupling "what depth generated this round's data" from
"what depth the promotion match itself runs at" -- a genuine improvement
in the candidate's value/policy function should now show up as a win
without the champion also getting a search-time boost to compensate.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from parchis.az import champion_pool, config as config_module, encoding, selfplay
from parchis.az import train as train_module
from parchis.az.agent import NetEvaluator, make_search_agent_factory
from parchis.az.net import AZNet, NumpyAZNet
from parchis.evaluation import duplicate

ROUND_DIR_PREFIX = "round_"
_ROUND_DIR_FMT = ROUND_DIR_PREFIX + "{:04d}"
_FRESH_META = {'round': -1, 'promotions': 0, 'consecutive_failures': 0}


def _round_dir(run_dir, round_num):
    return Path(run_dir) / "rounds" / _ROUND_DIR_FMT.format(round_num)


def _is_round_complete(round_dir):
    return (Path(round_dir) / "done.json").exists()


def find_resume_round(run_dir):
    """The next round number to run: 0 for a fresh run_dir, or (the
    highest COMPLETED round's number + 1) otherwise -- resumability
    (module docstring). A round directory that exists but lacks
    done.json (an interrupted run) is treated as not-yet-done and will be
    regenerated from scratch."""
    rounds_dir = Path(run_dir) / "rounds"
    if not rounds_dir.exists():
        return 0
    completed = [
        int(p.name[len(ROUND_DIR_PREFIX):]) for p in rounds_dir.iterdir()
        if p.is_dir() and p.name.startswith(ROUND_DIR_PREFIX) and _is_round_complete(p)
    ]
    return (max(completed) + 1) if completed else 0


def _champion_paths(run_dir):
    run_dir = Path(run_dir)
    return run_dir / "champion.pt", run_dir / "champion_meta.json", run_dir / "promoted_history.json"


def load_champion_state(run_dir, cfg):
    """Returns (state_dict, meta). Loads runs/<run_name>/{champion.pt,
    champion_meta.json} if they exist; otherwise a FRESH random-init
    AZNet of cfg's own shape, with meta={'round': -1, 'promotions': 0,
    'consecutive_failures': 0} ("no round has run yet"). Phase 3 is
    expected to actually be seeded from a Phase 2 bootstrap checkpoint in
    practice (see run_continuous's `initial_champion_state_dict`), but
    this fallback keeps round_loop.py itself independently correct/
    testable without one."""
    champion_path, meta_path, _history_path = _champion_paths(run_dir)
    if champion_path.exists():
        state_dict = torch.load(champion_path, map_location="cpu")
        with open(meta_path) as f:
            meta = json.load(f)
        return state_dict, meta
    input_size = encoding.encoding_size(cfg.num_players)
    model = AZNet(input_size, cfg.num_players, hidden_sizes=cfg.hidden_sizes)
    return model.state_dict(), dict(_FRESH_META)


def _save_champion_state(run_dir, state_dict, meta):
    champion_path, meta_path, _history_path = _champion_paths(run_dir)
    torch.save(state_dict, champion_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _gather_replay_buffer_shards(run_dir, round_num, this_round_shard_paths, replay_window_rounds):
    """This round's own shards + every shard from the previous
    (replay_window_rounds - 1) rounds (oldest rounds outside the window
    are simply never read again -- the recency window is enforced by
    never looking further back, not by deleting anything on disk)."""
    buffer_paths = list(this_round_shard_paths)
    oldest_included = max(0, round_num - replay_window_rounds + 1)
    for prior_round in range(round_num - 1, oldest_included - 1, -1):
        prior_shards_dir = _round_dir(run_dir, prior_round) / "shards"
        if prior_shards_dir.exists():
            buffer_paths.extend(str(p) for p in sorted(prior_shards_dir.glob("shard_*.npz")))
    return buffer_paths


def run_round(round_num, champion_state, meta, promoted_history, cfg, run_dir):
    """Runs ONE self-play round to completion (see module docstring for
    the 5 steps). Returns (new_champion_state, new_meta,
    new_promoted_history) -- the champion only actually changes if this
    round promoted; otherwise all three are returned unchanged (by value,
    not mutated in place) except meta['consecutive_failures'].
    """
    run_dir = Path(run_dir)
    round_dir = _round_dir(run_dir, round_num)
    shards_dir = round_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    escalate = meta['consecutive_failures'] >= cfg.escalate_after_failures
    depth = cfg.escalation_depth if escalate else cfg.base_depth

    input_size = encoding.encoding_size(cfg.num_players)
    champion_model = AZNet(input_size, cfg.num_players, hidden_sizes=cfg.hidden_sizes)
    champion_model.load_state_dict(champion_state)
    champion_model.eval()
    champion_numpy_net = NumpyAZNet.from_torch(champion_model)
    promoted_numpy_nets = [
        champion_pool.load_numpy_net(p, input_size, cfg.num_players, cfg.hidden_sizes)
        for p in promoted_history
    ]

    # --- 1. Generate (in shards, matching Phase 2's on-disk shape) ---
    n_shards = max(1, cfg.n_games_per_round // cfg.games_per_shard)
    games_per_shard = cfg.n_games_per_round // n_shards
    this_round_shard_paths = []
    for shard_i in range(n_shards):
        seed = cfg.generation_seed + round_num * 1000 + shard_i
        examples, stats = selfplay.generate_round_games(
            champion_numpy_net, promoted_numpy_nets, n_games=games_per_shard,
            num_players=cfg.num_players, max_turns=cfg.max_turns, depth=depth, seed=seed,
            lam=cfg.lam, tau_target=cfg.tau_target, tau_start=cfg.tau_start, tau_end=cfg.tau_end,
            anneal_plies=cfg.anneal_plies, dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
        )
        X, policy_targets, value_targets = selfplay.round_examples_to_arrays(examples, cfg.num_players)
        shard_path = shards_dir / f"shard_{shard_i:03d}.npz"
        np.savez(shard_path, X=X, policy_targets=policy_targets, value_targets=value_targets)
        this_round_shard_paths.append(str(shard_path))
        print(f"round {round_num} shard {shard_i}/{n_shards}: {stats['n_games']} games, "
              f"{stats['n_recorded_decisions']} recorded decisions "
              f"({stats['n_recorded_decisions'] / max(stats['n_total_plies'], 1):.1%} of plies), "
              f"depth={depth}", flush=True)

    # --- 2. Replay buffer ---
    buffer_shard_paths = _gather_replay_buffer_shards(
        run_dir, round_num, this_round_shard_paths, cfg.replay_window_rounds,
    )
    train_paths, val_paths = train_module.split_shards_train_val(
        buffer_shard_paths, val_frac=cfg.val_frac, seed=cfg.train_seed,
    )
    print(f"round {round_num}: replay buffer = {len(buffer_shard_paths)} shards "
          f"({len(train_paths)} train / {len(val_paths)} val)", flush=True)

    # --- 3. Train (warm-started from the current champion) ---
    candidate_model, history = train_module.bootstrap_train_sharded(
        train_paths, val_paths, num_players=cfg.num_players, hidden_sizes=cfg.hidden_sizes,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
        max_epochs=cfg.warm_start_max_epochs, patience=cfg.warm_start_patience,
        value_loss_weight=cfg.value_loss_weight, seed=cfg.train_seed, log_every=1,
        init_state_dict=champion_state,
    )
    with open(round_dir / "metrics.jsonl", "w") as f:
        for entry in history:
            f.write(json.dumps(entry) + "\n")
    candidate_path = round_dir / "candidate.pt"
    torch.save(candidate_model.state_dict(), candidate_path)

    # --- 4. Promote? ---
    # Evaluated at base_depth ALWAYS, regardless of what depth this round
    # generated at -- NOT `depth` (bug fixed 2026-08-27, see module
    # docstring: an escalated round's promotion match used to run BOTH
    # sides at escalation_depth, which confounded "did the deeper-search-
    # generated training data make the candidate's underlying net better"
    # with "is the old champion also newly searching deeper right now".
    # All 9 escalations in the first 40-round run failed to promote under
    # the old (confounded) test, while contributing ~79% of that run's
    # wall-clock time -- this decouples the two questions by holding the
    # comparison's OWN search depth fixed at base_depth no matter which
    # depth produced the data, so a genuine training-data improvement can
    # actually show up as a win.
    eval_depth = cfg.base_depth
    candidate_numpy_net = NumpyAZNet.from_torch(candidate_model)
    candidate_factory = make_search_agent_factory(NetEvaluator(candidate_numpy_net), depth=eval_depth)
    champion_factory = make_search_agent_factory(NetEvaluator(champion_numpy_net), depth=eval_depth)
    result = duplicate.play_duplicate_match(
        candidate_factory, champion_factory, n_pairs=cfg.promotion_n_pairs,
        num_players=cfg.num_players, max_turns=cfg.max_turns,
        seed=cfg.promotion_seed + round_num,
    )
    lower, upper = result['win_rate_a_ci']
    # bool(...): wilson_score_interval's lower/upper are numpy.float64
    # (via scipy.stats.norm.ppf) -- float64 quietly subclasses Python's
    # own float (so json.dump handles lower/upper directly just fine),
    # but numpy.bool_ does NOT subclass Python's bool, so the bare
    # comparison's result would fail json.dump below without this cast.
    promoted = bool(lower >= 0.5)
    with open(round_dir / "promotion_result.json", "w") as f:
        json.dump({
            'win_rate_a': result['win_rate_a'], 'win_rate_a_ci': [lower, upper],
            'wins_a': result['wins_a'], 'n_games': result['n_games'],
            'pair_record': result['pair_record'], 'promoted': promoted,
            'generation_depth': depth, 'eval_depth': eval_depth,
        }, f, indent=2)
    print(f"round {round_num}: win_rate_a={result['win_rate_a']:.4f} CI=[{lower:.4f}, {upper:.4f}] "
          f"generation_depth={depth} eval_depth={eval_depth} -> "
          f"{'PROMOTED' if promoted else 'not promoted'}", flush=True)

    # --- 5. Update champion / promoted history / escalation state ---
    new_meta = dict(meta)
    new_meta['round'] = round_num
    new_promoted_history = list(promoted_history)
    if promoted:
        new_state = candidate_model.state_dict()
        new_promoted_history = champion_pool.append_promoted(new_promoted_history, str(candidate_path))
        new_meta['promotions'] = meta.get('promotions', 0) + 1
        new_meta['consecutive_failures'] = 0
    else:
        new_state = champion_state
        # An escalated round "uses up" the failure streak regardless of
        # its own outcome (round_loop.py's own documented choice -- see
        # module docstring / SelfPlayRoundConfig.escalate_after_failures):
        # gives base_depth another full run of attempts before escalating
        # again, rather than re-escalating immediately on the very next
        # round if the escalated attempt ALSO failed to promote.
        new_meta['consecutive_failures'] = 0 if escalate else meta['consecutive_failures'] + 1

    _save_champion_state(run_dir, new_state, new_meta)
    champion_pool.save_promoted_history(_champion_paths(run_dir)[2], new_promoted_history)
    with open(round_dir / "done.json", "w") as f:
        json.dump({'round': round_num, 'promoted': promoted}, f)

    return new_state, new_meta, new_promoted_history


def run_continuous(cfg, runs_dir=config_module.DEFAULT_RUNS_DIR, max_rounds=None,
                    initial_champion_state_dict=None):
    """Runs rounds back-to-back until `max_rounds` TOTAL rounds have
    completed (None = run until stopped). Resumable: re-invoking this
    with the same cfg.run_name (and the same runs_dir) picks up at
    find_resume_round(run_dir), never redoing an already-completed round.

    `initial_champion_state_dict`: used ONLY the very first time this
    run_name is started (no champion.pt on disk yet) -- e.g. a Phase 2
    bootstrap checkpoint's state_dict, so Phase 3 continues training an
    already-reasonable net. Ignored on every later call (the on-disk
    champion.pt always wins once it exists) and ignored entirely if a
    fresh run_dir is intentionally started without one (falls back to
    load_champion_state's own random-init default).

    Returns: (final champion_state, final meta, final promoted_history).
    """
    run_dir = Path(runs_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(runs_dir=runs_dir)

    champion_path, meta_path, history_path = _champion_paths(run_dir)
    if not champion_path.exists() and initial_champion_state_dict is not None:
        _save_champion_state(run_dir, initial_champion_state_dict, dict(_FRESH_META))

    champion_state, meta = load_champion_state(run_dir, cfg)
    promoted_history = champion_pool.load_promoted_history(history_path)
    round_num = find_resume_round(run_dir)
    print(f"round_loop starting at round {round_num} (champion last updated at round "
          f"{meta['round']}, {meta['promotions']} total promotions, "
          f"{len(promoted_history)} promoted checkpoints in pool, "
          f"{meta['consecutive_failures']} consecutive failures)", flush=True)

    while max_rounds is None or round_num < max_rounds:
        start = time.perf_counter()
        champion_state, meta, promoted_history = run_round(
            round_num, champion_state, meta, promoted_history, cfg, run_dir,
        )
        elapsed = time.perf_counter() - start
        print(f"round {round_num} finished in {elapsed:.1f}s ({meta['promotions']} promotions so far)",
              flush=True)
        round_num += 1

    return champion_state, meta, promoted_history
