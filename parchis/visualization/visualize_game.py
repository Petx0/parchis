#!/usr/bin/env python3
"""
Command-line tool to visualize Parchís games from log files.
"""

import argparse
import os
import sys
from parchis.visualization.visualizer import replay_game_from_log


def main():
    """Main entry point for game visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize Parchís games from log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s logs/game_20260111_100159_BLUE.json
  %(prog)s logs/game_20260111_100159_BLUE.json --auto
  %(prog)s logs/game_20260111_100159_BLUE.json --save-frames
  %(prog)s --latest
        """
    )

    parser.add_argument(
        'log_file',
        nargs='?',
        help='Path to game log JSON file'
    )

    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-play without waiting for user input between turns'
    )

    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save each frame as an image file'
    )

    parser.add_argument(
        '--latest',
        action='store_true',
        help='Visualize the most recent game log'
    )

    args = parser.parse_args()

    # Handle --latest flag
    if args.latest:
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            print(f"Error: Log directory '{log_dir}' not found")
            sys.exit(1)

        # Find most recent log file
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        if not log_files:
            print(f"Error: No log files found in '{log_dir}'")
            sys.exit(1)

        log_files.sort(reverse=True)  # Most recent first (by filename timestamp)
        args.log_file = os.path.join(log_dir, log_files[0])
        print(f"Using latest log file: {args.log_file}")

    # Validate log file
    if not args.log_file:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.log_file):
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)

    # Run visualization
    try:
        step_by_step = not args.auto
        replay_game_from_log(args.log_file, step_by_step, args.save_frames)
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
    except Exception as e:
        print(f"\nError during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
