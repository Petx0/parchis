#!/usr/bin/env python3
"""
Main entry point for the Parchís game.
"""

import argparse
from parchis.game import Game
from parchis.utils.logger import GameLogger


def play_full_game(num_players=4, verbose=False):
    """
    Play a complete game from start to finish automatically.

    Args:
        num_players: Number of players (2-4)
        verbose: If True, print detailed turn information

    Returns:
        tuple: (winner, log_filepath)
    """
    print(f"\n{'='*60}")
    print(f"Starting Parchís Game - {num_players} Players (Full Auto Mode)")
    print(f"{'='*60}\n")

    # Create logger
    logger = GameLogger()

    # Create and play game
    game = Game(num_players=num_players, logger=logger)

    # Log game start with initial dice rolls
    logger.log_game_start(game.players, game.starting_player, game.initial_dice_rolls)

    if verbose:
        print(game.get_game_state_string())

    while not game.game_over:
        turn_info = game.play_turn()

        if verbose:
            print(game.format_turn_info(turn_info))

    # Game over
    print(f"\n{'='*60}")
    print(f"GAME OVER!")
    print(f"Winner: {game.winner.color}")
    print(f"Total Turns: {game.turn_number}")
    print(f"{'='*60}\n")

    # Log game end
    logger.log_game_end(game.winner, game.turn_number)

    # Save log
    log_filepath = logger.save_to_file()
    print(f"Game log saved to: {log_filepath}\n")

    return game.winner, log_filepath


def play_step_by_step(num_players=4):
    """
    Play a game step-by-step, requiring user input to advance.

    Args:
        num_players: Number of players (2-4)

    Returns:
        tuple: (winner, log_filepath)
    """
    print(f"\n{'='*60}")
    print(f"Starting Parchís Game - {num_players} Players (Step-by-Step Mode)")
    print(f"{'='*60}")
    print("Press ENTER to advance each turn, or 'q' to quit\n")

    # Create logger
    logger = GameLogger()

    # Create game
    game = Game(num_players=num_players, logger=logger)

    # Log game start with initial dice rolls
    logger.log_game_start(game.players, game.starting_player, game.initial_dice_rolls)

    print(game.get_game_state_string())

    while not game.game_over:
        user_input = input("\nPress ENTER for next turn (or 'q' to quit): ").strip().lower()

        if user_input == 'q':
            print("\nGame aborted by user.")
            return None, None

        turn_info = game.play_turn()
        print(game.format_turn_info(turn_info))

        if not game.game_over:
            print(game.get_game_state_string())

    # Game over
    print(f"\n{'='*60}")
    print(f"GAME OVER!")
    print(f"Winner: {game.winner.color}")
    print(f"Total Turns: {game.turn_number}")
    print(f"{'='*60}\n")

    # Log game end
    logger.log_game_end(game.winner, game.turn_number)

    # Save log
    log_filepath = logger.save_to_file()
    print(f"Game log saved to: {log_filepath}\n")

    return game.winner, log_filepath


def main():
    """Main function to run the game."""
    parser = argparse.ArgumentParser(
        description="Parchís (Ludo) Game - Version 1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode full --players 4          # Play full game with 4 players
  %(prog)s --mode step --players 2          # Play step-by-step with 2 players
  %(prog)s --mode full --players 4 -v       # Play full game with verbose output
        """
    )

    parser.add_argument(
        '--mode',
        choices=['full', 'step'],
        default='full',
        help='Game mode: "full" for automatic play, "step" for play-by-play (default: full)'
    )

    parser.add_argument(
        '--players',
        type=int,
        choices=[2, 3, 4],
        default=4,
        help='Number of players (2-4, default: 4)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (only for full mode)'
    )

    args = parser.parse_args()

    try:
        if args.mode == 'full':
            play_full_game(num_players=args.players, verbose=args.verbose)
        else:
            play_step_by_step(num_players=args.players)

    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        raise


if __name__ == "__main__":
    main()
