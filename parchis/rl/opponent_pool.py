"""
Opponent-pool sampling for self-play training.

Pure functions used by ParchisSelfPlayEnv (which pool member to use for the
next episode) and by SelfPlayCallback (parchis/training/train_selfplay.py,
which maintains the pool of past checkpoints and picks a sampling strategy).
Mirrors the parchis/rl/rewards.py pattern: pure functions, named constants,
a VALID_X/DEFAULT_X pair for the enum-like choice, ValueError on invalid
input.
"""
import math

VALID_POOL_SAMPLING_STRATEGIES = ("uniform", "recency", "win_rate")
DEFAULT_POOL_SAMPLING_STRATEGY = "uniform"
DEFAULT_POOL_SIZE = 5
DEFAULT_WIN_RATE_EPSILON = 0.01


def compute_recency_weights(n):
    """Linear-rank weights for a pool ordered oldest->newest: [1, 2, ..., n].
    Index n-1 (the newest/last element) gets the highest weight.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [float(i + 1) for i in range(n)]


def compute_win_rate_weights(win_rates, epsilon=DEFAULT_WIN_RATE_EPSILON):
    """weight_i = max(1.0 - win_rates[i], epsilon) -- hard-example-mining:
    favors opponents the current training model is weakest against, with an
    epsilon floor so a 100%-beaten opponent never reaches zero sampling
    probability.
    """
    if not win_rates:
        raise ValueError("win_rates must not be empty")
    for wr in win_rates:
        if wr < 0.0 or wr > 1.0:
            raise ValueError(f"win_rates must be within [0, 1], got {wr}")
    return [max(1.0 - wr, epsilon) for wr in win_rates]


def sample_pool_index(weights, rng):
    """Sample an index proportional to weights.

    rng: a random.Random instance -- NEVER the bare `random` module (dice
    rolls draw from that shared global stream; using it here would perturb
    seeded reproducibility of the game itself, not just opponent selection).
    """
    if not weights:
        raise ValueError("weights must not be empty")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    threshold = rng.uniform(0.0, total)
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if cumulative >= threshold:
            return i
    return len(weights) - 1


def pool_diversity_entropy(counts):
    """Normalized Shannon entropy in [0, 1] over a DENSE list of
    per-pool-member selection counts (callers must include zero-count
    members -- a sparse/filtered list silently hides low-diversity signal,
    the exact thing this metric exists to catch). Returns 1.0 for the
    degenerate len(counts) <= 1 case.
    """
    if len(counts) <= 1:
        return 1.0
    total = sum(counts)
    if total <= 0:
        return 1.0
    entropy = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counts))
