#!/usr/bin/env python3
"""
Demo script to showcase the Parchís visualization system.
Plays a short game and then visualizes it.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend if available

from parchis.game import Game
from parchis.utils.logger import GameLogger
from parchis.visualization.visualizer import replay_game_from_log


def main():
    """Run a demo game and visualize it."""
    print("=" * 60)
    print("PARCHÍS VISUALIZATION DEMO")
    print("=" * 60)
    print("\nThis demo will:")
    print("1. Play a short 2-player game")
    print("2. Save the game log")
    print("3. Replay the game with visualization")
    print("=" * 60)

    input("\nPress ENTER to start the demo game...")

    # Play a game
    print("\n[1/3] Playing a 2-player game...")
    logger = GameLogger()
    game = Game(num_players=2, logger=logger)
    logger.log_game_start(game.players, game.starting_player, game.initial_dice_rolls)

    turn_count = 0
    max_turns = 200  # Limit for demo

    while not game.game_over and turn_count < max_turns:
        game.play_turn()
        turn_count += 1

    if game.game_over:
        logger.log_game_end(game.winner, game.turn_number)
        log_path = logger.save_to_file()
        print(f"✓ Game completed in {game.turn_number} turns")
        print(f"✓ Winner: {game.winner.color}")
        print(f"✓ Log saved to: {log_path}")
    else:
        print(f"Game reached turn limit ({max_turns} turns)")
        return

    input("\n[2/3] Press ENTER to start the visualization replay...")

    # Visualize the game
    print("\n[3/3] Starting visualization...")
    print("=" * 60)
    print("CONTROLS:")
    print("- Press ENTER to see the move made in response to each roll")
    print("- Type 'q' and press ENTER to quit")
    print("=" * 60)

    replay_game_from_log(log_path, step_by_step=True, save_frames=False)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print("\nYou can replay this game anytime with:")
    print(f"  python visualize_game.py {log_path}")
    print("\nOr replay it automatically with:")
    print(f"  python visualize_game.py {log_path} --auto")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except ImportError as e:
        if 'matplotlib' in str(e):
            print("\nError: matplotlib is required for visualization.")
            print("Install it with: pip install matplotlib")
        else:
            raise
    except Exception as e:
        print(f"\nError: {e}")
        raise
