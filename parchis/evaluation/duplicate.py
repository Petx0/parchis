"""
Common-random-number (duplicate) match evaluation
(docs/AGENT_REBUILD_PLAN.md §5.1 / Phase 0 item 3).

arena.play_match uses an independent random seed per game -- separating a
genuine 53% from 50% at 95% confidence needs ~4,300 such games (§1.7).
Dice games have a standard fix: play the SAME dice seed more than once,
rotating which seat each agent occupies, and score the group -- this
cancels most of the dice luck a plain independent-seed match carries,
since both games see the identical sequence of rolls and only differ in
who was holding which seat when they landed.

At num_players=2 this is exactly the classic backgammon-style duplicate
PAIR (seats swapped). At num_players>2 it generalizes to rotating the
tested agent through every seat on the same seed (the Phase 4 protocol
docs/AGENT_REBUILD_PLAN.md previews) -- built once, generally, here rather
than as a 2-seat special case plus a separate later rewrite.
"""

import random
import statistics

from parchis.evaluation import arena
from parchis.evaluation import stats as eval_stats


def play_duplicate_group(agent_a_factory, agent_b_factory, num_players=2,
                          max_turns=arena.DEFAULT_MAX_TURNS, seed=None):
    """
    Play `num_players` games all sharing the SAME dice seed, rotating which
    seat `agent_a_factory` occupies (0, 1, ..., num_players-1) with
    `agent_b_factory` filling every other seat each time.

    Returns:
        dict: {'a_wins': int (0..num_players), 'n_games': num_players,
        'winners': list, one entry per rotation -- the seat index that won
        that particular game (as occupied at the time, not translated to
        "a" or "b"), or None if max_turns was hit with no winner}.
    """
    winners = []
    a_wins = 0
    for a_seat in range(num_players):
        agent_factories = {
            seat: (agent_a_factory if seat == a_seat else agent_b_factory)
            for seat in range(num_players)
        }
        winner_seat = arena.play_one_game(agent_factories, num_players=num_players,
                                           max_turns=max_turns, seed=seed)
        winners.append(winner_seat)
        if winner_seat == a_seat:
            a_wins += 1
    return {'a_wins': a_wins, 'n_games': num_players, 'winners': winners}


def play_duplicate_match(agent_a_factory, agent_b_factory, n_pairs, num_players=2,
                          max_turns=arena.DEFAULT_MAX_TURNS, seed=42,
                          confidence=eval_stats.DEFAULT_CONFIDENCE):
    """
    Play `n_pairs` independent duplicate groups (play_duplicate_group),
    each on its own seed drawn from a seeded RNG, and pool every individual
    game into one Wilson-CI'd win rate for agent_a -- the protocol
    docs/AGENT_REBUILD_PLAN.md Part 3 item 10's gate and §5.1 call for.

    The Wilson interval here treats each of the n_pairs*num_players games
    as one Bernoulli trial, same formula as arena.play_match -- it does NOT
    itself know the games are correlated in pairs. That's fine: CRN's
    benefit shows up as a TIGHTER interval for the same game budget because
    the underlying win-count distribution has less spread, not because the
    interval formula changes (see measure_variance_reduction, which
    quantifies exactly how much tighter).

    Returns:
        dict: wins_a, n_games (== n_pairs * num_players), win_rate_a,
        win_rate_a_ci, groups (raw play_duplicate_group results), and
        pair_record ({'a_better', 'split', 'b_better'} counts of groups by
        whether agent_a won more/equal/fewer than half its group's games --
        exact win/split/loss semantics only at num_players=2).
    """
    rng = random.Random(seed)
    groups = []
    wins_a = 0
    pair_record = {'a_better': 0, 'split': 0, 'b_better': 0}

    for _ in range(n_pairs):
        group_seed = rng.randrange(2**31)
        group = play_duplicate_group(agent_a_factory, agent_b_factory,
                                      num_players=num_players, max_turns=max_turns,
                                      seed=group_seed)
        groups.append(group)
        wins_a += group['a_wins']
        half = num_players / 2.0
        if group['a_wins'] > half:
            pair_record['a_better'] += 1
        elif group['a_wins'] < half:
            pair_record['b_better'] += 1
        else:
            pair_record['split'] += 1

    n_games = n_pairs * num_players
    win_rate_a = wins_a / n_games
    ci = eval_stats.wilson_score_interval(wins_a, n_games, confidence=confidence)

    return {
        'wins_a': wins_a,
        'n_games': n_games,
        'win_rate_a': win_rate_a,
        'win_rate_a_ci': ci,
        'groups': groups,
        'pair_record': pair_record,
    }


def measure_variance_reduction(agent_a_factory, agent_b_factory, n_pairs=100,
                                num_players=2, max_turns=arena.DEFAULT_MAX_TURNS,
                                repeats=24, seed=7):
    """
    Quantify the variance reduction duplicate/CRN matches buy over plain
    independent-seed matches (docs/AGENT_REBUILD_PLAN.md §5.1).

    A single Wilson interval computed from one aggregate (wins, n) count
    CANNOT tell correlated (duplicate-paired) trials from independent
    ones apart -- the formula only ever sees the final tally, not how it
    was generated, so it treats both methods identically and understates
    duplicate pairing's real benefit. The actual quantity "variance
    reduction" refers to is how much the win_rate ESTIMATE ITSELF varies
    from one full evaluation run to the next -- so this runs the whole
    procedure `repeats` times independently (a fresh top-level seed each
    time) under both protocols, over the identical per-repeat game budget
    (n_pairs * num_players games), and compares the empirical standard
    deviation of the resulting win_rate_a across repeats.

    Note on the degenerate a_factory is b_factory case: if both factories
    are literally the same policy, every duplicate GROUP cancels exactly
    (whichever seat wins under a shared seed wins regardless of which
    label is "a", so a_wins is always exactly 1 per group, deterministically,
    for any num_players) -- so duplicate_std comes out as a genuine 0.0
    across repeats (confirmed empirically, not assumed), and the
    independent method still shows its usual sampling spread. A real
    property of CRN, not a useful finite multiplier though -- pass two
    distinct-but-comparable agents (e.g. two heuristic weight vectors) for
    a measurement that reflects what a real Phase 1+ gate actually sees.

    Returns:
        dict: repeats, n_games_per_repeat, duplicate_mean_win_rate,
        duplicate_std, independent_mean_win_rate, independent_std,
        effective_n_multiplier ((independent_std / duplicate_std)**2, since
        standard error scales ~1/sqrt(n) -- approximates how many
        independent-seed games it would take to match the duplicate
        method's precision per game actually played; float('inf') if
        duplicate_std is exactly 0).
    """
    rng = random.Random(seed)
    n_games = n_pairs * num_players

    dup_rates = []
    ind_rates = []
    for _ in range(repeats):
        dup_result = play_duplicate_match(
            agent_a_factory, agent_b_factory, n_pairs=n_pairs, num_players=num_players,
            max_turns=max_turns, seed=rng.randrange(2**31),
        )
        dup_rates.append(dup_result['win_rate_a'])

        ind_result = arena.play_match(
            agent_a_factory, agent_b_factory, n_games=n_games, num_players=num_players,
            max_turns=max_turns, seed=rng.randrange(2**31),
        )
        ind_rates.append(ind_result['win_rate_a'])

    dup_std = statistics.pstdev(dup_rates)
    ind_std = statistics.pstdev(ind_rates)

    return {
        'repeats': repeats,
        'n_games_per_repeat': n_games,
        'duplicate_mean_win_rate': statistics.mean(dup_rates),
        'duplicate_std': dup_std,
        'independent_mean_win_rate': statistics.mean(ind_rates),
        'independent_std': ind_std,
        'effective_n_multiplier': (ind_std / dup_std) ** 2 if dup_std > 0 else float('inf'),
    }
