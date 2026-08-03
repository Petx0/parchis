"""
Numeric rule constants for Parchís turn/dice mechanics.

Board-layout constants (track size, safe squares, home entry points) live in
parchis.game.board.Board instead, since they describe the board, not turns.
"""

ENTRY_ROLL = 5              # Rolling this value allows entering a piece from base
BONUS_TURN_ROLL = 6         # Rolling this value grants a bonus turn (roll again)
ALL_OUT_BONUS_ROLL = 7      # Effective squares moved on a 6 when all pieces are out of base
THREE_SIXES_LIMIT = 3       # Consecutive 6s that trigger the three-sixes penalty
CAPTURE_BONUS_SQUARES = 20  # Bonus move size after a capture
FINISH_BONUS_SQUARES = 10   # Bonus move size after a piece finishes
