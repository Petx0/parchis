"""
Statistics helpers for evaluation: confidence intervals on win rates, and
the shared Phase 4 KPI-aggregation formula for evaluate_model/evaluate_agent.

Pure functions, no I/O or side effects -- mirrors the parchis/rl/rewards.py /
parchis/rl/opponent_pool.py pattern (named constants, ValueError on invalid
input). Introduced in docs/RL_DESIGN_REVIEW.md Phase 4: every win-rate
number reported anywhere in this codebase before this module was a bare
point estimate, indistinguishable from noise at small sample sizes.
"""

import math

import numpy as np
from scipy import stats as _scipy_stats

DEFAULT_CONFIDENCE = 0.95


def wilson_score_interval(wins, n, confidence=DEFAULT_CONFIDENCE):
    """
    Wilson score interval for a Bernoulli proportion (win rate).

    Preferred over the normal approximation (wins/n +/- z*sqrt(p(1-p)/n))
    because that approximation is unreliable exactly where evaluation runs
    live -- small n, or p near 0/1 -- and can produce bounds outside
    [0, 1]. Wilson's interval stays valid there.

    Args:
        wins: Number of wins observed (0 <= wins <= n).
        n: Number of trials (games). Must be >= 1.
        confidence: Confidence level, e.g. 0.95 for a 95% interval.

    Returns:
        tuple(float, float): (lower, upper), both within [0.0, 1.0].
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if wins < 0 or wins > n:
        raise ValueError(f"wins must be within [0, n], got wins={wins}, n={n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be within (0, 1), got {confidence}")

    # wins == 0 / wins == n are mathematically exact degenerate cases
    # (lower == 0.0 / upper == 1.0 respectively) -- special-cased rather
    # than left to the general formula below, whose sqrt() can round to a
    # tiny nonzero epsilon there (e.g. 5e-17), which would then fail a
    # strict lower <= win_rate <= upper sanity check at exactly p_hat=0.
    if wins == 0:
        lower = 0.0
    if wins == n:
        upper = 1.0

    z = _scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = wins / n
    z2 = z * z

    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n))

    if wins != 0:
        lower = max(0.0, (center - margin) / denominator)
    if wins != n:
        upper = min(1.0, (center + margin) / denominator)
    return lower, upper


def mean_confidence_interval(values, confidence=DEFAULT_CONFIDENCE):
    """
    t-distribution confidence interval on a sample mean (e.g. win_rate
    across N random seeds of the same config).

    Uses Student's t- rather than the normal distribution because
    multi-seed sweeps run as few as 2-3 seeds, where the normal
    approximation understates the interval width substantially.

    Args:
        values: Sample of >= 2 floats (e.g. one win_rate per seed).
        confidence: Confidence level, e.g. 0.95 for a 95% interval.

    Returns:
        tuple(float, float): (lower, upper).
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"values must have at least 2 elements, got {n}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be within (0, 1), got {confidence}")

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std_err = math.sqrt(variance / n)

    if std_err == 0.0:
        return mean, mean

    t_crit = _scipy_stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    margin = t_crit * std_err
    return mean - margin, mean + margin


def intervals_overlap(interval_a, interval_b):
    """
    True if two (lower, upper) intervals overlap at all.

    Used to decide whether a "best config" call is statistically
    supported: if the top config's CI overlaps the runner-up's, they are
    not distinguishable at this sample size and neither should be called
    "best" outright.
    """
    a_lower, a_upper = interval_a
    b_lower, b_upper = interval_b
    return a_lower <= b_upper and b_lower <= a_upper


def breakdown_win_rates(wins_by_key, games_by_key, confidence=DEFAULT_CONFIDENCE):
    """
    Per-key (e.g. per-seat or per-color) win rate + Wilson CI, from dense
    win/game counts covering the same set of keys.

    Args:
        wins_by_key: dict {key: wins}
        games_by_key: dict {key: games}, same keys as wins_by_key.
        confidence: Confidence level passed through to wilson_score_interval.

    Returns:
        dict {key: {'win_rate': float, 'n': int, 'ci': (float, float)}},
        skipping any key with zero games (undefined win rate).
    """
    breakdown = {}
    for key, n in games_by_key.items():
        if n <= 0:
            continue
        wins = wins_by_key.get(key, 0)
        breakdown[key] = {
            'win_rate': wins / n,
            'n': n,
            'ci': wilson_score_interval(wins, n, confidence=confidence),
        }
    return breakdown


def rank_by_mean_with_ci(entries):
    """
    Rank (label, mean, ci) entries by mean, descending, and determine
    whether the top entry is statistically distinguishable from the
    runner-up.

    Used by experiment_alpha_comparison.py / experiment_grid.py's
    multi-seed "best config" selection (docs/RL_DESIGN_REVIEW.md Phase 4):
    a raw point-estimate max() over win_rate_mean is indistinguishable
    from seed noise, so "best" should only be claimed outright when the
    top config's confidence interval doesn't overlap the runner-up's.

    Args:
        entries: list of (label, mean, ci_or_None) tuples. ci is a
            (lower, upper) tuple from mean_confidence_interval, or None if
            not enough samples existed to compute one (e.g. a single seed).

    Returns:
        tuple(list, bool): (entries sorted by mean descending, whether the
        top entry is statistically confirmed best -- True when there's
        only one entry, False whenever fewer than 2 entries would compare,
        or either of the top two lacks a CI, or their CIs overlap).
    """
    if not entries:
        return [], False
    ranked = sorted(entries, key=lambda e: e[1], reverse=True)
    if len(ranked) < 2:
        return ranked, True
    best_ci, runner_up_ci = ranked[0][2], ranked[1][2]
    if best_ci is None or runner_up_ci is None:
        return ranked, False
    return ranked, not intervals_overlap(best_ci, runner_up_ci)


def aggregate_phase4_stats(
    wins, n_episodes,
    wins_by_seat, games_by_seat,
    wins_by_color, games_by_color,
    captures_by_agent, captures_against_agent,
    legal_moves_counts, bonus_chain_lengths,
    three_sixes_penalty_count,
    confidence=DEFAULT_CONFIDENCE,
):
    """
    Build the Phase 4 KPI/CI additions to an evaluation stats dict.

    Shared by evaluate_model (parchis/training/common.py) and
    evaluate_agent (parchis/evaluation/evaluate.py) so these two
    near-duplicate evaluation loops -- which have drifted before, see
    docs/CODE_REVIEW.md -- don't drift on this formula too. Each loop
    still accumulates its own raw per-episode/per-step data (that part
    isn't shared, since the two loops' existing structures differ
    slightly); this function only does the final aggregation.

    Args:
        wins, n_episodes: overall win count / episode count.
        wins_by_seat, games_by_seat: dense dicts {seat_idx: count}.
        wins_by_color, games_by_color: dense dicts {color: count}.
        captures_by_agent, captures_against_agent: one entry per episode,
            each episode's total summed across that episode's turn cycles
            (ParchisEnv only surfaces these per turn cycle, not per step).
        legal_moves_counts: flat list, one entry per environment step
            across all episodes (ParchisEnv surfaces this on every step).
        bonus_chain_lengths: flat list, one entry per completed bonus
            chain across all episodes.
        three_sixes_penalty_count: total occurrences across all episodes.
        confidence: confidence level for win_rate_ci / the per-seat and
            per-color breakdown CIs.

    Returns:
        dict, meant to be merged into the caller's existing stats dict:
        win_rate_ci, win_rate_by_seat, win_rate_by_color, capture_rate,
        capture_rate_against, three_sixes_penalty_rate, and (only if the
        corresponding list is non-empty) mean_legal_moves_count /
        std_legal_moves_count, mean_bonus_chain_length /
        std_bonus_chain_length.
    """
    result = {
        'win_rate_ci': wilson_score_interval(wins, n_episodes, confidence=confidence),
        'win_rate_by_seat': breakdown_win_rates(wins_by_seat, games_by_seat, confidence=confidence),
        'win_rate_by_color': breakdown_win_rates(wins_by_color, games_by_color, confidence=confidence),
        'capture_rate': sum(captures_by_agent) / n_episodes if n_episodes else 0.0,
        'capture_rate_against': sum(captures_against_agent) / n_episodes if n_episodes else 0.0,
        'three_sixes_penalty_rate': three_sixes_penalty_count / n_episodes if n_episodes else 0.0,
    }
    if legal_moves_counts:
        result['mean_legal_moves_count'] = float(np.mean(legal_moves_counts))
        result['std_legal_moves_count'] = float(np.std(legal_moves_counts))
    if bonus_chain_lengths:
        result['mean_bonus_chain_length'] = float(np.mean(bonus_chain_lengths))
        result['std_bonus_chain_length'] = float(np.std(bonus_chain_lengths))
    return result
