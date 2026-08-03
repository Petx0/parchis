"""
Parchís Reinforcement Learning Module.

This module contains RL environments and wrappers for training agents:
- ParchisEnv: Base Gymnasium environment for Parchís
- ParchisSelfPlayEnv: Self-play environment wrapper
"""

from parchis.rl.env import ParchisEnv
from parchis.rl.env_selfplay import ParchisSelfPlayEnv

__all__ = ['ParchisEnv', 'ParchisSelfPlayEnv']
