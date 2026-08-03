"""
Elo rating updates for the checkpoint round-robin ladder.

Pure functions, no I/O or side effects -- mirrors the parchis/rl/rewards.py /
parchis/rl/opponent_pool.py pattern (named constants, ValueError on invalid
input). Introduced in docs/RL_DESIGN_REVIEW.md Phase 4, backing
parchis/evaluation/elo_ladder.py: "is checkpoint N+1 actually stronger" was
previously only inferable from a noisy, moving-target self-play win rate.
"""

DEFAULT_INITIAL_RATING = 1200.0
DEFAULT_K_FACTOR = 32.0


def expected_score(rating_a, rating_b):
    """
    Standard logistic Elo expectation: the probability A beats B, given
    their current ratings.

    Returns:
        float in (0.0, 1.0).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(rating_a, rating_b, score_a, k_factor=DEFAULT_K_FACTOR):
    """
    One Elo update from a single pairing's outcome.

    score_a is the fraction of games A won against B this pairing (e.g.
    win_rate from one evaluate_agent() call across n_games) -- one update
    per *pairing*, not per individual game. A pairing's evaluate_agent()
    call already plays n_games and returns a win_rate; treating that whole
    block as one Elo-update observation keeps the ladder a thin
    orchestration layer over the existing evaluate_agent(), rather than
    requiring per-game-level integration into the RL episode loop.
    Documented simplification vs. "true" sequential per-game Elo --
    acceptable for a *lightweight* ladder per docs/RL_DESIGN_REVIEW.md's
    own framing.

    Args:
        rating_a: A's current rating.
        rating_b: B's current rating.
        score_a: A's actual score against B, in [0.0, 1.0] (e.g. win_rate;
            0.5 counts as neither side winning, matching a 50/50 split).
        k_factor: Maximum rating change per update.

    Returns:
        tuple(float, float): (new_rating_a, new_rating_b).
    """
    if not 0.0 <= score_a <= 1.0:
        raise ValueError(f"score_a must be within [0, 1], got {score_a}")

    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a

    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)
    return new_rating_a, new_rating_b


def round_robin_pairings(participant_ids, rng):
    """
    All unordered pairs among participant_ids, in shuffled order.

    rng: a random.Random instance -- NEVER the bare `random` module (see
    parchis/rl/opponent_pool.py's sample_pool_index for why: the bare
    module is the shared stream Dice.roll() draws from, and this ladder
    may run evaluation episodes against live environments in the same
    process).

    Args:
        participant_ids: List of >= 2 distinct participant identifiers.
        rng: random.Random instance used to shuffle pairing order.

    Returns:
        list of (id_a, id_b) tuples, one per unordered pair.
    """
    if len(participant_ids) < 2:
        raise ValueError(
            f"participant_ids must have at least 2 elements, got {len(participant_ids)}"
        )
    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("participant_ids must not contain duplicates")

    pairs = [
        (participant_ids[i], participant_ids[j])
        for i in range(len(participant_ids))
        for j in range(i + 1, len(participant_ids))
    ]
    rng.shuffle(pairs)
    return pairs
