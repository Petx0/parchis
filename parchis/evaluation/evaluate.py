"""
Evaluation script for Parchís agents.

This script tests a trained model against opponents (either another model or random players)
and provides detailed statistics on performance.

Usage:
    # Test against random opponents (2 players by default)
    python -m parchis.evaluation.evaluate --model ./models/my_model/final_model --n-games 100

    # Test with 4 players
    python -m parchis.evaluation.evaluate --model ./models/my_model/final_model --players 4 --n-games 100

    # Test against another model
    python -m parchis.evaluation.evaluate \\
        --model ./models/my_model/final_model \\
        --opponent ./models/opponent_model/final_model \\
        --n-games 100

    # Test with specific settings
    python -m parchis.evaluation.evaluate \\
        --model ./models/my_model/final_model \\
        --opponent random \\
        --players 2 \\
        --n-games 200 \\
        --deterministic
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO

from parchis.rl.env_selfplay import ParchisSelfPlayEnv
from parchis.training.common import make_env, mask_fn, get_game, get_base_env
from parchis.evaluation import stats as eval_stats


def evaluate_agent(
    agent_model_path,
    opponent_model_path=None,
    n_games=100,
    num_players=2,
    deterministic=True,
    max_steps_per_episode=10000,
    verbose=1
):
    """
    Evaluate an agent against opponents.

    Args:
        agent_model_path: Path to the agent model to evaluate
        opponent_model_path: Path to opponent model (None or "random" for random opponents)
        n_games: Number of games to play
        num_players: Number of players (2-4)
        deterministic: Whether to use deterministic policy for agent
        max_steps_per_episode: Maximum steps per episode (safety limit)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)

    Returns:
        dict: Evaluation statistics
    """
    # Load agent model
    if verbose > 0:
        print("=" * 70)
        print("Parchís Agent Evaluation")
        print("=" * 70)
        print(f"Loading agent model from: {agent_model_path}")

    agent_model = MaskablePPO.load(agent_model_path)

    # Determine opponent type and create environment
    if opponent_model_path is None or opponent_model_path.lower() == "random":
        if verbose > 0:
            print(f"Opponent type: Random players ({num_players} players)")
        env = make_env(num_players=num_players, seed=42)
        opponent_model = None
    else:
        if verbose > 0:
            print(f"Loading opponent model from: {opponent_model_path}")
        opponent_model = MaskablePPO.load(opponent_model_path)

        if verbose > 0:
            print(f"Opponent type: Trained model ({num_players} players)")

        # Create self-play environment with opponent model. mask_fn (shared
        # with the training scripts, see parchis.training.common) already
        # knows how to unwrap a ParchisSelfPlayEnv, so no self-play-specific
        # wrapping logic is needed here.
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor

        env = ParchisSelfPlayEnv(opponent_model=opponent_model, num_players=num_players)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)

    # Evaluation statistics
    wins = 0
    losses = 0
    timeouts = 0
    truncated = 0

    episode_rewards = []
    episode_lengths = []
    episode_final_progress = []
    episode_pieces_finished = []
    episode_pieces_out = []
    win_rewards = []
    loss_rewards = []

    # Phase 4 KPI accumulators (docs/RL_DESIGN_REVIEW.md): per-seat/color
    # win-rate breakdown, capture rate, legal-move-count and
    # bonus-chain-length distributions, three-sixes-penalty rate.
    wins_by_seat, games_by_seat = {}, {}
    wins_by_color, games_by_color = {}, {}
    episode_captures_by_agent = []
    episode_captures_against_agent = []
    legal_moves_counts = []
    bonus_chain_lengths = []
    three_sixes_penalty_count = 0

    if verbose > 0:
        print(f"\nEvaluating for {n_games} games...")
        print(f"Agent policy: {'Deterministic' if deterministic else 'Stochastic'}")
        print(f"Max steps per episode: {max_steps_per_episode:,}")
        print("=" * 70)

    # Run evaluation games
    for game_num in range(n_games):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        final_terminated = False
        final_truncated = False
        episode_capture_total = 0
        episode_capture_against_total = 0
        prev_bonus_chain_count = 0

        while not done and episode_length < max_steps_per_episode:
            # Get action mask (mask_fn handles both plain and self-play envs)
            action_masks = mask_fn(env)

            # Predict action
            action, _ = agent_model.predict(
                obs,
                action_masks=action_masks,
                deterministic=deterministic
            )
            action = int(action)

            # Execute step
            obs, reward, terminated, truncated_flag, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated_flag
            final_terminated = terminated
            final_truncated = truncated_flag

            legal_moves_counts.append(info['legal_moves_count'])
            bonus_chain_count = info['bonus_chain_count']
            if bonus_chain_count == 0 and prev_bonus_chain_count > 0:
                bonus_chain_lengths.append(prev_bonus_chain_count)
            prev_bonus_chain_count = bonus_chain_count
            if 'captures_by_agent' in info:
                episode_capture_total += info['captures_by_agent']
                episode_capture_against_total += info['captures_against_agent']
                if info['three_sixes_penalty']:
                    three_sixes_penalty_count += 1

        # get_game/get_base_env handle both plain and self-play envs;
        # agent_player_idx is the random seat ParchisEnv assigned the
        # learning agent for this episode (see env.reset()).
        agent_seat = get_base_env(env).agent_player_idx
        agent_color = get_game(env).players[agent_seat].color
        games_by_seat[agent_seat] = games_by_seat.get(agent_seat, 0) + 1
        games_by_color[agent_color] = games_by_color.get(agent_color, 0) + 1

        # Determine game outcome
        if episode_length >= max_steps_per_episode and not done:
            status = "TIMEOUT"
            timeouts += 1
        elif final_truncated:
            status = "TRUNCATED"
            truncated += 1
        elif final_terminated:
            agent_player = get_game(env).players[agent_seat]

            if agent_player.has_won():
                status = "WIN"
                wins += 1
                win_rewards.append(episode_reward)
                wins_by_seat[agent_seat] = wins_by_seat.get(agent_seat, 0) + 1
                wins_by_color[agent_color] = wins_by_color.get(agent_color, 0) + 1
            else:
                status = "LOSS"
                losses += 1
                loss_rewards.append(episode_reward)
        else:
            status = "UNKNOWN"

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_captures_by_agent.append(episode_capture_total)
        episode_captures_against_agent.append(episode_capture_against_total)

        # Track additional metrics if available in info
        if 'final_progress' in info:
            episode_final_progress.append(info['final_progress'])
        if 'pieces_finished' in info:
            episode_pieces_finished.append(info['pieces_finished'])
        if 'pieces_out_of_base' in info:
            episode_pieces_out.append(info['pieces_out_of_base'])

        # Print progress
        if verbose == 1 and (game_num + 1) % 10 == 0:
            print(f"  Completed {game_num + 1}/{n_games} games...")
        elif verbose >= 2:
            print(f"Game {game_num + 1}/{n_games}: "
                  f"Reward = {episode_reward:7.2f}, "
                  f"Length = {episode_length:4d}, "
                  f"Status = {status}")

    # Calculate statistics
    stats = {
        'n_games': n_games,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'truncated': truncated,
        'win_rate': wins / n_games,
        'loss_rate': losses / n_games,
        'timeout_rate': timeouts / n_games,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_length': np.min(episode_lengths),
        'max_length': np.max(episode_lengths),
    }

    # Add progress metrics if available
    if len(episode_final_progress) > 0:
        stats['mean_final_progress'] = np.mean(episode_final_progress)
        stats['std_final_progress'] = np.std(episode_final_progress)
        stats['min_final_progress'] = np.min(episode_final_progress)
        stats['max_final_progress'] = np.max(episode_final_progress)

    if len(episode_pieces_finished) > 0:
        stats['mean_pieces_finished'] = np.mean(episode_pieces_finished)
        stats['std_pieces_finished'] = np.std(episode_pieces_finished)

    if len(episode_pieces_out) > 0:
        stats['mean_pieces_out'] = np.mean(episode_pieces_out)
        stats['std_pieces_out'] = np.std(episode_pieces_out)

    # Add conditional statistics
    if len(win_rewards) > 0:
        stats['mean_win_reward'] = np.mean(win_rewards)
        stats['std_win_reward'] = np.std(win_rewards)
    else:
        stats['mean_win_reward'] = 0.0
        stats['std_win_reward'] = 0.0

    if len(loss_rewards) > 0:
        stats['mean_loss_reward'] = np.mean(loss_rewards)
        stats['std_loss_reward'] = np.std(loss_rewards)
    else:
        stats['mean_loss_reward'] = 0.0
        stats['std_loss_reward'] = 0.0

    stats.update(eval_stats.aggregate_phase4_stats(
        wins=wins, n_episodes=n_games,
        wins_by_seat=wins_by_seat, games_by_seat=games_by_seat,
        wins_by_color=wins_by_color, games_by_color=games_by_color,
        captures_by_agent=episode_captures_by_agent,
        captures_against_agent=episode_captures_against_agent,
        legal_moves_counts=legal_moves_counts,
        bonus_chain_lengths=bonus_chain_lengths,
        three_sixes_penalty_count=three_sixes_penalty_count,
    ))

    # Print summary
    if verbose > 0:
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"\nGames played: {n_games}")
        print(f"\nOutcomes:")
        print(f"  Wins:      {wins:4d} ({stats['win_rate']*100:5.1f}%)")
        ci_lower, ci_upper = stats['win_rate_ci']
        print(f"  Win rate 95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"  Losses:    {losses:4d} ({stats['loss_rate']*100:5.1f}%)")
        if timeouts > 0:
            print(f"  Timeouts:  {timeouts:4d} ({stats['timeout_rate']*100:5.1f}%)")
        if truncated > 0:
            print(f"  Truncated: {truncated:4d} ({truncated/n_games*100:5.1f}%)")

        print(f"\nRewards:")
        print(f"  Mean:      {stats['mean_reward']:7.2f} ± {stats['std_reward']:.2f}")
        print(f"  Range:     [{stats['min_reward']:7.2f}, {stats['max_reward']:7.2f}]")
        if wins > 0:
            print(f"  Win avg:   {stats['mean_win_reward']:7.2f} ± {stats['std_win_reward']:.2f}")
        if losses > 0:
            print(f"  Loss avg:  {stats['mean_loss_reward']:7.2f} ± {stats['std_loss_reward']:.2f}")

        print(f"\nEpisode Lengths:")
        print(f"  Mean:      {stats['mean_length']:6.1f} ± {stats['std_length']:.1f}")
        print(f"  Range:     [{stats['min_length']:4.0f}, {stats['max_length']:4.0f}]")

        # Print progress metrics if available
        if 'mean_final_progress' in stats:
            print(f"\nProgress Metrics:")
            print(f"  Final progress:    {stats['mean_final_progress']:.3f} ± {stats['std_final_progress']:.3f}")
            print(f"  Progress range:    [{stats['min_final_progress']:.3f}, {stats['max_final_progress']:.3f}]")

        if 'mean_pieces_finished' in stats:
            print(f"  Pieces finished:   {stats['mean_pieces_finished']:.2f} ± {stats['std_pieces_finished']:.2f}")

        if 'mean_pieces_out' in stats:
            print(f"  Pieces out of base: {stats['mean_pieces_out']:.2f} ± {stats['std_pieces_out']:.2f}")

        print(f"\nSeat/Color Fairness:")
        print(f"  By seat:  "
              + ", ".join(f"seat {k}: {v['win_rate']:.1%} (n={v['n']})"
                           for k, v in sorted(stats['win_rate_by_seat'].items())))
        print(f"  By color: "
              + ", ".join(f"{k}: {v['win_rate']:.1%} (n={v['n']})"
                           for k, v in sorted(stats['win_rate_by_color'].items())))

        print(f"\nGame KPIs:")
        print(f"  Capture rate:      {stats['capture_rate']:.2f}/game "
              f"(against agent: {stats['capture_rate_against']:.2f}/game)")
        if 'mean_legal_moves_count' in stats:
            print(f"  Legal moves/decision: {stats['mean_legal_moves_count']:.2f} "
                  f"± {stats['std_legal_moves_count']:.2f}")
        if 'mean_bonus_chain_length' in stats:
            print(f"  Bonus chain length: {stats['mean_bonus_chain_length']:.2f} "
                  f"± {stats['std_bonus_chain_length']:.2f}")
        print(f"  Three-sixes penalty rate: {stats['three_sixes_penalty_rate']:.2f}/game")

        print("=" * 70)

    env.close()
    return stats


def main():
    """Main evaluation script with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a Parchís agent against opponents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate against random opponents (2 players by default)
  python -m parchis.evaluation.evaluate --model ./models/my_model/final_model --n-games 100

  # Evaluate with 4 players
  python -m parchis.evaluation.evaluate --model ./models/my_model/final_model --players 4 --n-games 100

  # Evaluate against another model
  python -m parchis.evaluation.evaluate \\
      --model ./models/my_model/final_model \\
      --opponent ./models/opponent/final_model \\
      --n-games 100

  # Quick test with stochastic policy
  python -m parchis.evaluation.evaluate \\
      --model ./models/my_model/final_model \\
      --n-games 20 \\
      --no-deterministic
        """
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the agent model to evaluate (without .zip extension)"
    )

    # Optional arguments
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        help="Path to opponent model or 'random' for random opponents (default: random)"
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=100,
        help="Number of games to play (default: 100)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)"
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy instead of deterministic"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode (default: 10,000)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0=silent, 1=progress, 2=detailed (default: 1)"
    )

    args = parser.parse_args()

    # Run evaluation
    stats = evaluate_agent(
        agent_model_path=args.model,
        opponent_model_path=args.opponent if args.opponent.lower() != "random" else None,
        n_games=args.n_games,
        num_players=args.players,
        deterministic=args.deterministic,
        max_steps_per_episode=args.max_steps,
        verbose=args.verbose
    )

    return stats


if __name__ == "__main__":
    main()
