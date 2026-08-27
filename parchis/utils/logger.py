"""
Logger module for recording Parchís games.
"""

import json
import os
from datetime import datetime


class GameLogger:
    """Logs game events and saves complete game history."""

    def __init__(self, log_dir="logs"):
        """
        Initialize the game logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.log_data = {
            # Bumped whenever this dict's shape changes in a way a consumer
            # (parchis/visualization/visualizer.py, agentinfo_io.py) needs
            # to branch on -- see docs/CODE_REVIEW.md's note that this
            # contract previously had no version field at all. Consumers
            # should read it via .get('schema_version', 0) so pre-existing
            # logs on disk (which predate this field) still load.
            'schema_version': 1,
            'metadata': {},
            'turns': [],
            'result': {}
        }

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

    def log_game_start(self, players, starting_player=None, initial_dice_rolls=None):
        """
        Log the start of a game.

        Args:
            players: List of Player objects (in turn order)
            starting_player: The player who won the initial roll
            initial_dice_rolls: List of roll rounds from determine_starting_player
        """
        self.log_data['metadata'] = {
            'start_time': datetime.now().isoformat(),
            'num_players': len(players),
            'players': [{'id': p.player_id, 'color': p.color} for p in players]
        }

        # Log starting player and initial dice rolls
        if starting_player:
            self.log_data['metadata']['starting_player'] = {
                'id': starting_player.player_id,
                'color': starting_player.color
            }

        if initial_dice_rolls:
            # Format initial rolls for logging
            roll_rounds = []
            for round_rolls in initial_dice_rolls:
                round_data = [
                    {'player_id': r['player'].player_id, 'color': r['player'].color, 'roll': r['roll']}
                    for r in round_rolls
                ]
                roll_rounds.append(round_data)
            self.log_data['metadata']['initial_dice_rolls'] = roll_rounds

    def log_turn(self, turn_info):
        """
        Log a single turn (which may include multiple rolls due to bonus turns).

        Args:
            turn_info: parchis.game.records.TurnInfo
        """
        turn_data = {
            'turn_number': turn_info.turn_number,
            'player_id': turn_info.player.player_id,
            'player_color': turn_info.player.color,
            'rolls': [],
            'three_sixes_penalty': turn_info.three_sixes_penalty
        }

        # Log each roll in the turn (includes dice rolls and bonus moves)
        for roll_info in turn_info.rolls:
            if roll_info.is_bonus:
                roll_data = {
                    'bonus_type': roll_info.bonus_type,
                    'bonus_squares': roll_info.bonus_squares,
                    'legal_moves_count': roll_info.legal_moves_count,
                }
            else:
                roll_data = {
                    'dice_roll': roll_info.dice_roll,
                    'legal_moves_count': roll_info.legal_moves_count,
                }

            if roll_info.chosen_move:
                move_info = roll_info.move_info
                piece = move_info.piece
                roll_data['move'] = {
                    'piece_id': piece.piece_id,
                    'piece_color': piece.color,
                    'old_position': move_info.old_position,
                    'new_position': move_info.new_position,
                    'move_type': move_info.move_type,
                    'captured': [
                        {'piece_id': p.piece_id, 'color': p.color}
                        for p in move_info.captured
                    ]
                }
            else:
                roll_data['move'] = None

            turn_data['rolls'].append(roll_data)

        # Log penalty piece if three sixes occurred
        if turn_info.three_sixes_penalty and turn_info.penalty_piece:
            penalty_piece = turn_info.penalty_piece
            turn_data['penalty_piece'] = {
                'piece_id': penalty_piece.piece_id,
                'color': penalty_piece.color
            }

        # Log home entry protection if three sixes rolled but no penalty
        if turn_info.home_entry_protection:
            turn_data['home_entry_protection'] = True

        self.log_data['turns'].append(turn_data)

    def log_game_end(self, winner, total_turns):
        """
        Log the end of a game.

        Args:
            winner: The winning Player object
            total_turns: Total number of turns played
        """
        self.log_data['result'] = {
            'end_time': datetime.now().isoformat(),
            'winner_id': winner.player_id,
            'winner_color': winner.color,
            'total_turns': total_turns
        }

    def save_to_file(self, filename=None):
        """
        Save the game log to a JSON file.

        Args:
            filename: Optional filename. If not provided, generates one with timestamp.

        Returns:
            str: Path to the saved log file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            winner_color = self.log_data['result'].get('winner_color', 'unknown')
            filename = f"game_{timestamp}_{winner_color}.json"

        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        return filepath

    def get_summary(self):
        """
        Get a summary of the logged game.

        Returns:
            str: Game summary
        """
        if not self.log_data['result']:
            return "Game in progress or not started"

        metadata = self.log_data['metadata']
        result = self.log_data['result']

        lines = [
            "Game Summary",
            "=" * 50,
            f"Players: {metadata['num_players']}",
            f"Start Time: {metadata['start_time']}",
            f"End Time: {result['end_time']}",
            f"Total Turns: {result['total_turns']}",
            f"Winner: {result['winner_color']} (Player {result['winner_id']})",
            "=" * 50
        ]

        return "\n".join(lines)

    @staticmethod
    def load_from_file(filepath):
        """
        Load a game log from a JSON file.

        Args:
            filepath: Path to the log file

        Returns:
            dict: The loaded game data
        """
        with open(filepath, 'r') as f:
            return json.load(f)
