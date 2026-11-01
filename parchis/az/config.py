"""
Run configuration (docs/AGENT_REBUILD_PLAN.md Part 3): one dataclass,
serialized to runs/<name>/config.json at the start of every run, so a
checkpoint or metrics.jsonl can always be traced back to exactly what
produced it.

Two run shapes, one per phase: BootstrapConfig (Phase 2, the one-time
supervised bootstrap) and SelfPlayRoundConfig (Phase 3, the continuous
self-play loop) -- following the same "one dataclass per run, always
saved" pattern.
"""

import dataclasses
import json
from pathlib import Path

from parchis.az import targets

DEFAULT_RUNS_DIR = "runs"


@dataclasses.dataclass
class BootstrapConfig:
    """Phase 2 bootstrap: generate a fixed dataset from the hand-built
    pool, then supervised-train the value/policy heads on it once."""

    run_name: str
    num_players: int = 2

    # Generation (parchis.az.selfplay.generate_games)
    n_games: int = 20_000
    max_turns: int = 500
    generation_seed: int = 0
    noisy_epsilon: float = 0.15

    # Game-level split (parchis.az.train.split_by_game)
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac is implicitly 1 - train_frac - val_frac
    split_seed: int = 0

    # Net (parchis.az.net.AZNet)
    hidden_sizes: tuple = (256, 256)

    # Optimization (docs/AGENT_REBUILD_PLAN.md Part 4's table)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 50
    early_stopping_patience: int = 5
    value_loss_weight: float = 1.0
    train_seed: int = 0

    def save(self, runs_dir=DEFAULT_RUNS_DIR):
        """Writes runs/<run_name>/config.json. Returns the run directory."""
        run_dir = Path(runs_dir) / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        return run_dir

    @classmethod
    def load(cls, run_name, runs_dir=DEFAULT_RUNS_DIR):
        path = Path(runs_dir) / run_name / "config.json"
        with open(path) as f:
            data = json.load(f)
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)


@dataclasses.dataclass
class SelfPlayRoundConfig:
    """Phase 3's continuous self-play loop (docs/AGENT_REBUILD_PLAN.md
    Part 3 Phase 3): each round generates games with the current champion
    (parchis.az.selfplay.generate_round_games), warm-start retrains on a
    recency-windowed replay buffer (parchis.az.round_loop), and promotes
    the result only on a CI-confirmed win over the champion
    (parchis.evaluation.duplicate). See parchis/az/round_loop.py for how
    these fields are actually used."""

    run_name: str
    num_players: int = 2

    # Generation (parchis.az.selfplay.generate_round_games)
    n_games_per_round: int = 50_000
    games_per_shard: int = 10_000
    max_turns: int = 500
    base_depth: int = 1
    lam: float = targets.DEFAULT_LAMBDA
    tau_target: float = targets.DEFAULT_TAU_TARGET
    tau_start: float = targets.DEFAULT_TAU_START
    tau_end: float = targets.DEFAULT_TAU_END
    anneal_plies: int = targets.DEFAULT_ANNEAL_PLIES
    dirichlet_alpha: float = targets.DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = targets.DEFAULT_DIRICHLET_EPSILON
    generation_seed: int = 0

    # Replay buffer: this round's shards + the previous (replay_window_rounds - 1)
    # rounds' -- a recency window, not unbounded accumulation (Part 3 Phase 3 /
    # docs/SEARCH_MCTS.md's documented past mistake).
    replay_window_rounds: int = 3

    # Warm-start training (parchis.az.train.bootstrap_train_sharded).
    # hidden_sizes MUST match whatever champion.pt was actually saved with
    # (round_loop.py has no way to detect a mismatch other than a shape
    # error at load time).
    hidden_sizes: tuple = (256, 256)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4096
    # Deliberately much smaller than BootstrapConfig's 40/6 -- warm-started
    # from already-reasonable weights on a small recent-rounds buffer, this
    # only needs to nudge the net, not re-learn it from scratch (Part 3
    # Phase 3: "Cap warm-start epochs").
    warm_start_max_epochs: int = 5
    warm_start_patience: int = 2
    value_loss_weight: float = 2.0
    val_frac: float = 0.1
    train_seed: int = 0

    # Promotion (Part 3 Phase 3: "only on a CI-confirmed win ... on >= 600
    # duplicate pairs").
    promotion_n_pairs: int = 600
    promotion_seed: int = 0

    # Escalation: after `escalate_after_failures` CONSECUTIVE non-
    # promotions at base_depth, the next round generates (and evaluates
    # promotion) at escalation_depth instead, for exactly that one round --
    # "expert iteration -- stronger data than the current net can produce
    # on its own". consecutive_failures resets to 0 after an escalated
    # round regardless of whether IT promotes, giving base_depth
    # `escalate_after_failures` more attempts before escalating again
    # (not specified numerically by the plan; this is round_loop.py's own
    # documented choice -- see its module docstring).
    #
    # enable_escalation defaults True to preserve existing configs' exact
    # historical behavior on load (dataclass field defaults fill in
    # missing keys). It exists because the lineage's actual record is 0
    # promotions from 16 escalated rounds across 68 rounds, while
    # consuming ~79% of the first 40 rounds' wall-clock (see
    # docs/AZ_DESIGN.md and .claude/plans/twinkly-marinating-hinton.md) --
    # set False to skip escalation entirely and keep training at
    # base_depth regardless of consecutive_failures.
    escalate_after_failures: int = 3
    escalation_depth: int = 2
    enable_escalation: bool = True

    # Rollout-refined value targets (Phase 2.2, parchis.az.rollouts):
    # value_target_mode="root_value" (default, unchanged historical
    # behavior) always uses search's own root_value as z_value's
    # bootstrap term. "rollout" additionally spends rollout_n heuristic
    # rollouts (parchis.az.rollouts.estimate_rollout_value) on a random
    # rollout_target_fraction of recorded decisions per round, replacing
    # root_value with the rollout estimate as the bootstrap term for
    # exactly those decisions -- an A/B-able alternative (compare a few
    # rounds of each mode via the ladder) rather than a change applied
    # irreversibly to every decision, which would multiply generation
    # cost by rollout_n. See .claude/plans/twinkly-marinating-hinton.md
    # Phase 1.4 for the diagnostic that motivated this (root_value found
    # to systematically overestimate the mover's own win probability by
    # ~2.9 points relative to an independent rollout estimate, p<0.0001).
    value_target_mode: str = "root_value"
    rollout_target_fraction: float = 0.05
    rollout_n: int = 24

    # Playout-depth randomization (Phase 2.3, parchis.az.selfplay): a
    # random deep_fraction of recorded decisions per round use search at
    # deep_depth instead of the round's fixed generation depth --
    # expert-iteration-style extra-strength data injected cheaply, without
    # escalation's proven-bad cost profile of paying for depth on EVERY
    # decision of a round (0/16 promotions, ~79% of early wall-clock; see
    # escalate_after_failures' comment above and
    # .claude/plans/twinkly-marinating-hinton.md). Added after Phases
    # 2.1/3.1/2.2/4.1 combined (65 rounds) showed no detectable strength
    # improvement -- the cheapest untried lever from the original plan.
    # Default 0.0 (off) preserves existing configs' behavior on load; the
    # promotion-gate evaluation is unaffected either way (it always runs
    # at base_depth via a separate code path in round_loop.py).
    deep_fraction: float = 0.0
    deep_depth: int = 2

    # Auxiliary head (Phase 4.1, parchis.az.net.AZNet.aux_head): predicts
    # whether each of the mover's own pieces finishes by game end, free
    # supervision from games already generated (see
    # parchis.az.selfplay.generate_round_games' module docstring). Added
    # after escalation retirement + pool broadening + rollout-refined
    # targets ran 50 rounds combined with no detectable strength
    # improvement (docs/AZ_DESIGN.md's "Strength-improvement plan" entry).
    # Default 0.0 (off) preserves existing configs' behavior on load; the
    # aux head still EXISTS in the network either way (AZNet always has
    # one now) but contributes nothing to the loss until this is raised.
    aux_loss_weight: float = 0.0

    # Promotion gate (docs/V2_PLAN.md C4): True (default) = historical
    # behavior, the generator only changes on a CI-confirmed win. False =
    # the latest candidate ALWAYS becomes the generator (AlphaZero/KataGo
    # style); the candidate-vs-previous match still runs, as telemetry only,
    # and promoted_history is NOT appended (every candidate is already in
    # recent_history; appending both would double-weight the last four
    # nets in the opponent pool -- docs/V2_PLAN.md §10.1 item 5).
    gate_enabled: bool = True

    # Opponent pool (docs/V2_PLAN.md §10.2). pool_random_anchor=True
    # (default, historical) keeps the uniformly-random mover as a pool
    # member; False drops it (V2.0's round 0 drew it for 1/3 of opponent
    # seats -- data about beating a random mover, not about strong play).
    # pool_heuristic_fraction=None (default, historical) samples the
    # opponent seat uniformly over {nets, anchors}; a float in [0, 1] is the
    # probability that the opponent seat is the tuned heuristic, the rest
    # uniform over the other entries -- a fixed small share that does not
    # depend on how many nets the histories currently hold.
    pool_random_anchor: bool = True
    pool_heuristic_fraction: float = None

    # Afterstate rows (docs/V2_PLAN.md C3, parchis.az.selfplay): True also
    # records, for every non-winning decision, the depth-1 search leaf of
    # the move actually played (pre-roll, from the next-to-decide seat's
    # perspective) as a value/aux-only training row with policy weight 0.
    # Makes search's leaf states in-distribution for the value head (V2.0
    # measured a 1.6pp pre-roll/post-roll inconsistency against a 1.3pp
    # median move gap). Roughly doubles rows per round. Default False
    # preserves existing configs' behavior on load.
    afterstate_rows: bool = False

    # Generation workers (docs/V2_PLAN.md C6(a), parchis.az.round_loop).
    # 1 (default, historical): every shard is generated in-process, one
    # after another. N > 1: up to N shards at a time in spawned worker
    # processes. Each shard keeps its own seed, so the shards a round
    # produces are byte-identical to the sequential run -- only the
    # wall-clock changes. Set games_per_shard = n_games_per_round / N so
    # every worker gets exactly one shard.
    generation_workers: int = 1

    # Deep targets (docs/V2_PLAN.md C2, parchis.az.selfplay). Requires
    # deep_fraction > 0. Every decision still plays its own search's move;
    # what changes is the TRAINING TARGET each row carries: a deep decision
    # (searched at deep_depth) keeps the blended value target
    # (1-lam)*outcome + lam*root and a policy target with weight 1; a cheap
    # decision (base_depth) gets the outcome alone (lam = 0 -- the depth-1
    # root is the max-biased quantity being removed) and policy weight 0.
    # Afterstate rows (afterstate_rows=True) follow their decision: the
    # deep search's chance-resolved value of the played move as bootstrap,
    # or the outcome alone. KataGo's playout-cap randomisation. Default
    # False preserves existing configs' behavior on load.
    deep_targets: bool = False

    def save(self, runs_dir=DEFAULT_RUNS_DIR):
        """Writes runs/<run_name>/config.json. Returns the run directory."""
        run_dir = Path(runs_dir) / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)
        return run_dir

    @classmethod
    def load(cls, run_name, runs_dir=DEFAULT_RUNS_DIR):
        path = Path(runs_dir) / run_name / "config.json"
        with open(path) as f:
            data = json.load(f)
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
        return cls(**data)
