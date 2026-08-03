"""
Shared building blocks for Parchís training scripts: action-mask helper,
environment factory, TensorBoard progress-logging callback, and evaluation
loop.

Every training script imports from here instead of reimplementing these
pieces. Before this module existed, a code review found mask_fn copy-pasted
6+ times, make_env() reimplemented 3 times with mutually exclusive extra
knobs (some scripts could set opponent_weight, others reward_type, none
could set both), ProgressLoggingCallback reimplemented 3 times, and the
evaluation loop reimplemented 3 times -- with real drift as a result (e.g.
a gamma default mismatch between one script's function signature and its
own CLI parser). See docs/CODE_REVIEW.md for the full history.
"""

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from parchis.rl.env import ParchisEnv
from parchis.evaluation import stats as eval_stats


def mask_fn(env):
    """
    Return the current action mask, whatever env is wrapped in. Works for
    both a plain ParchisEnv and a ParchisSelfPlayEnv-wrapped environment
    (which nests the real ParchisEnv under .base_env) -- previously each
    training script that needed self-play support redefined this same
    unwrap-and-check logic itself.
    """
    base = env.unwrapped
    if hasattr(base, 'base_env'):
        base = base.base_env
    return base._get_info()['action_masks']


def get_game(env):
    """Get the underlying Game object, whatever env is wrapped in (see mask_fn)."""
    return get_base_env(env).game


def get_base_env(env):
    """Get the underlying ParchisEnv instance, whatever env is wrapped in (see mask_fn)."""
    base = env.unwrapped
    if hasattr(base, 'base_env'):
        base = base.base_env
    return base


def make_env(num_players=4, opponent_weight=0.5, opponent_weighting="mean",
             reward_type="progress_delta", seed=None):
    """
    Create and wrap a ParchisEnv with action masking and episode monitoring.
    The single environment factory for scripts training against random
    opponents (see train_selfplay.make_selfplay_env for the self-play
    equivalent, which wraps ParchisSelfPlayEnv instead).

    Args:
        num_players: Number of players (2-4)
        opponent_weight: α value for the progress_delta/win_loss_shaped
            reward's opponent-progress term (default 0.5)
        opponent_weighting: How multiple opponents' deltas combine into
            that term -- one of ParchisEnv.VALID_OPPONENT_WEIGHTING_SCHEMES
            (default "mean"). Experiment-only knob, see
            parchis/training/experiment_alpha_comparison.py.
        reward_type: One of ParchisEnv.VALID_REWARD_TYPES (default "progress_delta")
        seed: Random seed

    Returns:
        Wrapped environment ready for training
    """
    env = ParchisEnv(num_players=num_players, reward_type=reward_type,
                      opponent_weight=opponent_weight, opponent_weighting=opponent_weighting)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)

    return env


class ProgressLoggingCallback(BaseCallback):
    """
    Logs final_progress / pieces_finished / pieces_out_of_base / win_rate to
    TensorBoard (as a rolling mean over the last `window` episodes) every
    time an episode ends. The single canonical progress-logging callback.
    """

    def __init__(self, window=100, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.episode_final_progress = []
        self.episode_pieces_finished = []
        self.episode_pieces_out = []
        self.episode_wins = []

    def _on_step(self) -> bool:
        if self.locals.get('dones', [False])[0]:
            info = self.locals['infos'][0]

            if 'final_progress' in info:
                self.episode_final_progress.append(info['final_progress'])
            if 'pieces_finished' in info:
                self.episode_pieces_finished.append(info['pieces_finished'])
            if 'pieces_out_of_base' in info:
                self.episode_pieces_out.append(info['pieces_out_of_base'])
            if 'won' in info:
                self.episode_wins.append(1 if info['won'] else 0)

            if self.episode_final_progress:
                self.logger.record('metrics/final_progress', np.mean(self.episode_final_progress[-self.window:]))
            if self.episode_pieces_finished:
                self.logger.record('metrics/pieces_finished', np.mean(self.episode_pieces_finished[-self.window:]))
            if self.episode_pieces_out:
                self.logger.record('metrics/pieces_out_of_base', np.mean(self.episode_pieces_out[-self.window:]))
            if self.episode_wins:
                self.logger.record('metrics/win_rate', np.mean(self.episode_wins[-self.window:]))

        return True


class FixedOpponentEvalCallback(BaseCallback):
    """
    Periodically evaluates the current in-training model against a FIXED,
    stationary opponent (typically random) and logs the resulting win rate
    to TensorBoard as metrics/win_rate_vs_baseline (and, when available,
    metrics/final_progress_vs_baseline).

    This exists specifically for self-play: train_selfplay.py's opponent
    keeps improving over the course of training, so ProgressLoggingCallback's
    win_rate there (measured against that moving target) conflates "is the
    agent improving" with "is its opponent also improving" -- a checkpoint
    could show a flat or falling win rate purely because the opponent got
    stronger, not because the agent regressed. Holding the evaluation
    opponent fixed for the whole run isolates the first question.

    Deliberately does NOT use stable_baselines3's built-in EvalCallback,
    which has a documented compatibility issue with MaskablePPO that can
    hang training indefinitely (see docs/EVALUATION_FIX.md). Calls
    evaluate_model() directly instead, which already has the safety-timeout
    fix that makes evaluation safe to run mid-training.
    """

    def __init__(self, eval_env, eval_freq=50_000, n_eval_episodes=20, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            stats = evaluate_model(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                verbose=0,
            )
            self.logger.record('metrics/win_rate_vs_baseline', stats['win_rate'])
            if 'mean_final_progress' in stats:
                self.logger.record('metrics/final_progress_vs_baseline', stats['mean_final_progress'])
            if self.verbose > 0:
                print(f"\n[baseline eval @ {self.num_timesteps:,} steps] "
                      f"win_rate_vs_baseline={stats['win_rate']:.2%}")

        return True


def evaluate_model(model, env, n_eval_episodes=10, deterministic=True,
                    max_steps_per_episode=10000, verbose=1):
    """
    Evaluate a trained model against whatever opponents `env` plays
    (random or self-play), tracking win/loss/timeout status, reward,
    episode length, and progress metrics for both the agent and its
    opponents. The single canonical evaluation loop.

    Args:
        model: Trained model with .predict(obs, action_masks=..., deterministic=...)
        env: Environment to evaluate on (already wrapped with action masking)
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        max_steps_per_episode: Maximum steps per episode (safety limit --
            see docs/EVALUATION_FIX.md for why this exists)
        verbose: 0 = silent, 1 = per-episode status lines + summary

    Returns:
        dict: aggregate statistics -- n_episodes, wins, losses, timeouts,
            win_rate, mean_reward, std_reward, mean_length, and (when
            available from info/game state) mean_final_progress,
            mean_pieces_finished, mean_pieces_out, mean_opponent_progress,
            std_opponent_progress
    """
    episode_rewards = []
    episode_lengths = []
    episode_final_progress = []
    episode_pieces_finished = []
    episode_pieces_out = []
    episode_opponent_progress = []
    wins = 0
    losses = 0
    timeouts = 0

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

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        final_terminated = False
        final_truncated = False
        final_info = {}
        episode_capture_total = 0
        episode_capture_against_total = 0
        prev_bonus_chain_count = 0

        while not done and episode_length < max_steps_per_episode:
            action_masks = mask_fn(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            action = int(action)  # Convert numpy array to int
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            final_terminated = terminated
            final_truncated = truncated
            if done:
                final_info = info

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

        base_env = get_base_env(env)
        game = base_env.game
        agent_seat = base_env.agent_player_idx
        agent_color = game.players[agent_seat].color
        games_by_seat[agent_seat] = games_by_seat.get(agent_seat, 0) + 1
        games_by_color[agent_color] = games_by_color.get(agent_color, 0) + 1

        if episode_length >= max_steps_per_episode and not done:
            status = "TIMEOUT"
            timeouts += 1
        elif final_truncated:
            status = "TRUNCATED"
        elif final_terminated:
            agent_player = game.players[base_env.agent_player_idx]
            if agent_player.has_won():
                status = "WIN"
                wins += 1
                wins_by_seat[agent_seat] = wins_by_seat.get(agent_seat, 0) + 1
                wins_by_color[agent_color] = wins_by_color.get(agent_color, 0) + 1
            else:
                status = "LOSS"
                losses += 1
        else:
            status = "UNKNOWN"

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_captures_by_agent.append(episode_capture_total)
        episode_captures_against_agent.append(episode_capture_against_total)

        if 'final_progress' in final_info:
            episode_final_progress.append(final_info['final_progress'])
        if 'pieces_finished' in final_info:
            episode_pieces_finished.append(final_info['pieces_finished'])
        if 'pieces_out_of_base' in final_info:
            episode_pieces_out.append(final_info['pieces_out_of_base'])

        opponent_progress = [
            base_env._calculate_normalized_progress(game.players[i])
            for i in range(len(game.players))
            if i != base_env.agent_player_idx
        ]
        if opponent_progress:
            episode_opponent_progress.append(np.mean(opponent_progress))

        if verbose > 0:
            print(f"Episode {episode + 1}/{n_eval_episodes}: "
                  f"Reward = {episode_reward:.2f}, Length = {episode_length}, Status = {status}")

    stats = {
        'n_episodes': n_eval_episodes,
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'win_rate': wins / n_eval_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
    }
    if episode_final_progress:
        stats['mean_final_progress'] = np.mean(episode_final_progress)
    if episode_pieces_finished:
        stats['mean_pieces_finished'] = np.mean(episode_pieces_finished)
    if episode_pieces_out:
        stats['mean_pieces_out'] = np.mean(episode_pieces_out)
    if episode_opponent_progress:
        stats['mean_opponent_progress'] = np.mean(episode_opponent_progress)
        stats['std_opponent_progress'] = np.std(episode_opponent_progress)

    stats.update(eval_stats.aggregate_phase4_stats(
        wins=wins, n_episodes=n_eval_episodes,
        wins_by_seat=wins_by_seat, games_by_seat=games_by_seat,
        wins_by_color=wins_by_color, games_by_color=games_by_color,
        captures_by_agent=episode_captures_by_agent,
        captures_against_agent=episode_captures_against_agent,
        legal_moves_counts=legal_moves_counts,
        bonus_chain_lengths=bonus_chain_lengths,
        three_sixes_penalty_count=three_sixes_penalty_count,
    ))

    if verbose > 0:
        print(f"\nEvaluation Results:")
        print(f"  Episodes: {n_eval_episodes}")
        print(f"  Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
        print(f"  Mean length: {stats['mean_length']:.2f}")
        print(f"  Wins: {wins}/{n_eval_episodes} ({stats['win_rate']*100:.1f}%)")
        ci_lower, ci_upper = stats['win_rate_ci']
        print(f"  Win rate 95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
        print(f"  Losses: {losses}/{n_eval_episodes} ({losses/n_eval_episodes*100:.1f}%)")
        if timeouts > 0:
            print(f"  Timeouts: {timeouts}/{n_eval_episodes} ({timeouts/n_eval_episodes*100:.1f}%)")
        if 'mean_final_progress' in stats:
            print(f"  Mean final progress: {stats['mean_final_progress']:.3f} (0.0 to 1.0)")
        if 'mean_pieces_finished' in stats:
            print(f"  Mean pieces finished: {stats['mean_pieces_finished']:.2f} (0 to 4)")
        if 'mean_pieces_out' in stats:
            print(f"  Mean pieces out of base: {stats['mean_pieces_out']:.2f} (0 to 4)")
        if 'mean_opponent_progress' in stats:
            print(f"  Mean opponent progress: {stats['mean_opponent_progress']:.3f} "
                  f"+/- {stats['std_opponent_progress']:.3f}")
        print(f"  Win rate by seat: "
              + ", ".join(f"seat {k}: {v['win_rate']:.1%} (n={v['n']})"
                           for k, v in sorted(stats['win_rate_by_seat'].items())))
        print(f"  Win rate by color: "
              + ", ".join(f"{k}: {v['win_rate']:.1%} (n={v['n']})"
                           for k, v in sorted(stats['win_rate_by_color'].items())))
        print(f"  Capture rate: {stats['capture_rate']:.2f}/game "
              f"(against agent: {stats['capture_rate_against']:.2f}/game)")
        if 'mean_legal_moves_count' in stats:
            print(f"  Legal moves per decision: {stats['mean_legal_moves_count']:.2f} "
                  f"+/- {stats['std_legal_moves_count']:.2f}")
        if 'mean_bonus_chain_length' in stats:
            print(f"  Bonus chain length: {stats['mean_bonus_chain_length']:.2f} "
                  f"+/- {stats['std_bonus_chain_length']:.2f}")
        print(f"  Three-sixes penalty rate: {stats['three_sixes_penalty_rate']:.2f}/game")

    return stats
