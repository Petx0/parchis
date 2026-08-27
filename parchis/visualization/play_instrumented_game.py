#!/usr/bin/env python3
"""
Command-line tool: play one game with real trained agents (search
checkpoints, the tuned/default heuristic, or random), instrumented to
capture the network/search's decision-time valuation of the position and
of each candidate move, then immediately replay it with the value panel --
the "watch the agents play and see what they're thinking" entry point
(docs/AGENT_REBUILD_PLAN.md's visualization plan). Mirrors
visualize_game.py's CLI style, but plays a fresh instrumented game instead
of only replaying an existing log.

Examples:
  python -m parchis.visualization.play_instrumented_game \\
      --agent 0=checkpoint:runs/selfplay_2p_v1_champion:depth=1 \\
      --agent 1=heuristic:tuned

  python -m parchis.visualization.play_instrumented_game \\
      --agent 0=checkpoint:runs/selfplay_2p_v1_champion \\
      --num-players 2 --seed 42 --auto --no-replay
"""

import argparse
import sys

from parchis.agents import heuristic
from parchis.az import search as az_search
from parchis.az.agent import NetEvaluator
from parchis.visualization import checkpoint_loading
from parchis.visualization.instrumented_play import play_and_record
from parchis.visualization.visualizer import replay_game_from_log


def _parse_agent_spec(spec_str):
    """'SEAT=SPEC' -> (seat, (kind, params)), where SPEC is one of:
    'checkpoint:<run_dir>[:depth=N]' | 'heuristic:tuned|default' | 'random'.
    kind/params match parchis.visualization.instrumented_play.play_and_record's
    agent_specs contract."""
    if '=' not in spec_str:
        raise argparse.ArgumentTypeError(
            f"--agent must be SEAT=SPEC, got {spec_str!r} (missing '=')"
        )
    seat_str, spec = spec_str.split('=', 1)
    try:
        seat = int(seat_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--agent seat must be an integer, got {seat_str!r}")

    parts = spec.split(':')
    kind = parts[0]

    if kind == 'checkpoint':
        if len(parts) < 2:
            raise argparse.ArgumentTypeError(
                f"--agent {spec_str!r}: 'checkpoint' needs a run_dir, e.g. checkpoint:runs/my_run"
            )
        run_dir = parts[1]
        depth = None
        for extra in parts[2:]:
            if extra.startswith('depth='):
                depth = int(extra.split('=', 1)[1])
        numpy_net, _num_players, cfg = checkpoint_loading.load_agent_numpy_net(run_dir)
        if depth is None:
            depth = cfg.get('base_depth', az_search.DEFAULT_DEPTH)
        evaluator = NetEvaluator(numpy_net)
        label = f"checkpoint:{run_dir} (search depth={depth})"
        return seat, ('search', (evaluator, depth)), label

    if kind == 'heuristic':
        which = parts[1] if len(parts) > 1 else 'tuned'
        if which == 'tuned':
            weights = heuristic.TUNED_WEIGHTS
        elif which == 'default':
            weights = heuristic.DEFAULT_WEIGHTS
        else:
            raise argparse.ArgumentTypeError(
                f"--agent {spec_str!r}: 'heuristic' expects 'tuned' or 'default', got {which!r}"
            )
        return seat, ('heuristic', weights), f"heuristic:{which}"

    if kind == 'random':
        return seat, ('random', None), 'random'

    raise argparse.ArgumentTypeError(
        f"--agent {spec_str!r}: unknown agent kind {kind!r} "
        f"(expected 'checkpoint', 'heuristic', or 'random')"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Play one instrumented game with real trained agents and visualize it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--agent', action='append', default=[], metavar='SEAT=SPEC',
                         help="Repeatable. SPEC: checkpoint:<run_dir>[:depth=N] | "
                              "heuristic:tuned|default | random. A seat with no --agent "
                              "keeps the default random player.")
    parser.add_argument('--num-players', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-turns', type=int, default=500)
    parser.add_argument('--log-dir', default='logs')
    parser.add_argument('--auto', action='store_true', help="Auto-play replay without waiting for ENTER")
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--no-value-panel', action='store_true', help="Suppress the value panel in the replay")
    parser.add_argument('--no-replay', action='store_true', help="Play + log only, skip the replay")

    args = parser.parse_args()

    agent_specs = {}
    agent_labels = []
    for spec_str in args.agent:
        seat, spec, label = _parse_agent_spec(spec_str)
        agent_specs[seat] = spec
        agent_labels.append(f"seat {seat}: {label}")
    for seat in range(args.num_players):
        if seat not in agent_specs:
            agent_labels.append(f"seat {seat}: random (default)")

    print("=" * 60)
    print("PLAYING INSTRUMENTED GAME")
    print("=" * 60)
    for label in agent_labels:
        print(f"  {label}")
    print("=" * 60)

    try:
        log_path, agentinfo_path = play_and_record(
            agent_specs, num_players=args.num_players, max_turns=args.max_turns,
            seed=args.seed, log_dir=args.log_dir,
        )
    except Exception as e:
        print(f"\nError while playing the game: {e}")
        raise

    print(f"✓ Game log saved to: {log_path}")
    print(f"✓ Agent-value data saved to: {agentinfo_path}" if agentinfo_path
          else "  (no seat was instrumented -- no agent-value sidecar produced)")

    if args.no_replay:
        return

    try:
        replay_game_from_log(
            log_path, step_by_step=not args.auto, save_frames=args.save_frames,
            agentinfo_filepath=agentinfo_path,
            show_value_panel=(False if args.no_value_panel else None),
        )
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
