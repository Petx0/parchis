"""
Dice module for Parchís game.
"""

import random


class Dice:
    """Represents a single six-sided die."""

    def __init__(self, sides=6):
        """
        Initialize the dice.

        Args:
            sides: Number of sides on the die (default: 6)
        """
        self.sides = sides

    def roll(self):
        """
        Roll the dice and return the result.

        Returns:
            int: Random number between 1 and sides (inclusive)
        """
        return random.randint(1, self.sides)
