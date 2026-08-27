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
    escalate_after_failures: int = 3
    escalation_depth: int = 2

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
