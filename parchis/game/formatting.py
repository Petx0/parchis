"""
Human-readable text rendering of game/turn state, kept separate from Game so
the engine itself has no presentation concerns.
"""


def format_game_state(game):
    """
    Get a human-readable string of the current game state.

    Args:
        game: The Game instance to describe

    Returns:
        str: Game state description
    """
    lines = [
        f"\n{'='*60}",
        f"Turn {game.turn_number} - Current Player: {game.get_current_player().color}",
        f"{'='*60}",
    ]

    # Player status
    lines.append("\nPlayer Status:")
    for player in game.players:
        in_base = len(player.get_pieces_in_base())
        on_board = len(player.get_pieces_on_board())
        finished = len(player.get_finished_pieces())
        lines.append(f"  {player.color}: Base={in_base}, Board={on_board}, Finished={finished}")

    # Board state
    lines.append(f"\n{game.board.get_board_state()}")

    return "\n".join(lines)


def format_turn_info(turn_info):
    """
    Format a TurnInfo record as a readable string.
    Handles multiple rolls (bonus turns from rolling 6, and capture/finish
    bonus chains).

    Args:
        turn_info: parchis.game.records.TurnInfo

    Returns:
        str: Formatted turn information
    """
    player = turn_info.player
    lines = [
        f"\n--- Turn {turn_info.turn_number} ---",
        f"Player: {player.color}",
    ]

    # Format each roll (includes dice rolls and bonus moves)
    for roll_idx, roll_info in enumerate(turn_info.rolls):
        if roll_info.is_bonus:
            bonus_type = roll_info.bonus_type
            bonus_squares = roll_info.bonus_squares
            if bonus_type == 'capture_bonus':
                lines.append(f"\n  [CAPTURE BONUS - {bonus_squares} squares]")
            else:
                lines.append(f"\n  [FINISH BONUS - {bonus_squares} squares]")
            lines.append(f"  Legal Moves Available: {roll_info.legal_moves_count}")
        else:
            # This is a regular dice roll
            if roll_idx > 0:
                lines.append(f"\n  [Bonus Roll from rolling 6]")
            lines.append(f"  Dice Roll: {roll_info.dice_roll}")
            lines.append(f"  Legal Moves Available: {roll_info.legal_moves_count}")

        if roll_info.chosen_move:
            move_info = roll_info.move_info
            piece = move_info.piece
            old_pos = move_info.old_position
            new_pos = move_info.new_position
            move_type = move_info.move_type

            if move_type == 'enter':
                lines.append(f"  Action: Entered {piece} from BASE to position {new_pos}")
            elif move_type == 'finish':
                lines.append(f"  Action: {piece} moved from {old_pos} to FINISH!")
            else:
                lines.append(f"  Action: Moved {piece} from {old_pos} to {new_pos}")

            if move_info.captured:
                for captured_piece in move_info.captured:
                    lines.append(f"    CAPTURED: {captured_piece} sent back to base!")
        else:
            lines.append("  Action: No legal moves available")

    # Check for three sixes penalty or home entry protection
    if turn_info.three_sixes_penalty:
        lines.append(f"\n  *** THREE SIXES PENALTY: {turn_info.penalty_piece} sent back to base! ***")
    elif turn_info.home_entry_protection:
        lines.append(f"\n  *** THREE SIXES - PROTECTED: Piece entered home column, no penalty ***")

    if turn_info.game_over:
        lines.append(f"\n*** GAME OVER - {turn_info.winner.color} WINS! ***")

    return "\n".join(lines)
