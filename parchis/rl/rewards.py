"""
Reward computation for ParchisEnv.

Extracted into its own module so the three reward_types, their constants,
and the opponent-weighting term each have exactly one implementation with a
dedicated unit test -- previously this all lived inline in
ParchisEnv.step(), flagged as an open rebuild target in docs/CODE_REVIEW.md
("move the reward function into its own module with named constants and
one unit test per term"). See docs/REWARD_STRUCTURE.md for the full
narrative description.
"""

from parchis.game.board import Board

VALID_REWARD_TYPES = ("progress_delta", "win_loss", "win_loss_shaped")

DEFAULT_OPPONENT_WEIGHT = 0.5

WIN_REWARD = 1.0
LOSS_REWARD = -1.0
WIN_LOSS_SHAPED_MIDGAME_SCALE = 0.1

# "mean": equal weight to every opponent's progress delta this cycle --
# the original (and still default) behavior, diluted to 1/(N-1) of its
# 2-player strength as num_players grows.
# "leader": weight only the single opponent with the highest progress at
# the *start* of the cycle (the biggest rival entering the cycle), so the
# term's strength doesn't dilute as num_players grows. Experiment-only --
# see docs/RL_DESIGN_REVIEW.md phase 2 and experiment_alpha_comparison.py.
VALID_OPPONENT_WEIGHTING_SCHEMES = ("mean", "leader")
DEFAULT_OPPONENT_WEIGHTING = "mean"


def calculate_normalized_progress(player):
    """
    A player's normalized progress (0.0 to 1.0): the average completion
    across their 4 pieces.
    - Piece in base: 0.0
    - Piece on board: position / Board.FINAL_POSITION
    - Piece finished: 1.0

    Args:
        player: The Player to calculate progress for.

    Returns:
        float: Average progress across 4 pieces (range: 0.0 to 1.0)
    """
    total_progress = 0.0
    for piece in player.pieces:
        if piece.finished:
            total_progress += 1.0
        elif piece.in_base:
            total_progress += 0.0
        elif piece.position is not None:
            total_progress += piece.position / Board.FINAL_POSITION

    return total_progress / 4.0


def combine_opponent_deltas(opponent_deltas, opponent_start_progress, weighting=DEFAULT_OPPONENT_WEIGHTING):
    """
    Combine multiple opponents' per-turn-cycle progress deltas into the
    single scalar term the progress_delta reward's opponent-penalty is
    based on.

    Args:
        opponent_deltas: dict {opponent_player_idx: progress_delta_this_cycle}
        opponent_start_progress: dict {opponent_player_idx: progress_at_cycle_start}.
            Only read by weighting="leader", to identify the biggest rival
            *entering* the cycle -- deliberately not influenced by what
            happens during the cycle itself.
        weighting: One of VALID_OPPONENT_WEIGHTING_SCHEMES.
            - "mean" (default): equal weight to every opponent.
            - "leader": weight only the opponent with the highest progress
              at the start of the cycle, regardless of how many opponents
              there are.

    Returns:
        float
    """
    if weighting not in VALID_OPPONENT_WEIGHTING_SCHEMES:
        raise ValueError(
            f"Invalid opponent weighting '{weighting}'. "
            f"Must be one of: {VALID_OPPONENT_WEIGHTING_SCHEMES}"
        )

    if not opponent_deltas:
        return 0.0

    if weighting == "mean":
        return sum(opponent_deltas.values()) / len(opponent_deltas)

    # weighting == "leader"
    leader_idx = max(opponent_start_progress, key=opponent_start_progress.get)
    return opponent_deltas[leader_idx]


def compute_reward(reward_type, my_delta, combined_opponent_delta, opponent_weight, terminated, agent_won):
    """
    Compute the reward for one of VALID_REWARD_TYPES.

    Args:
        reward_type: One of VALID_REWARD_TYPES.
        my_delta: The agent's own progress delta this turn cycle.
        combined_opponent_delta: The already-combined opponent delta term
            (see combine_opponent_deltas).
        opponent_weight: alpha, the weight applied to combined_opponent_delta.
        terminated: Whether the episode ended this cycle.
        agent_won: Whether the agent is the winner (only meaningful when terminated).

    Returns:
        tuple(float, float): (reward, progress_delta). progress_delta is
        returned alongside reward since win_loss_shaped needs it and
        callers may want it for logging.
    """
    if reward_type not in VALID_REWARD_TYPES:
        raise ValueError(
            f"Invalid reward_type '{reward_type}'. Must be one of: {VALID_REWARD_TYPES}"
        )

    progress_delta = my_delta - opponent_weight * combined_opponent_delta

    if reward_type == "progress_delta":
        return progress_delta, progress_delta

    if reward_type == "win_loss":
        if terminated:
            return (WIN_REWARD if agent_won else LOSS_REWARD), progress_delta
        return 0.0, progress_delta

    # reward_type == "win_loss_shaped"
    if terminated:
        return (WIN_REWARD if agent_won else LOSS_REWARD), progress_delta
    return WIN_LOSS_SHAPED_MIDGAME_SCALE * progress_delta, progress_delta
