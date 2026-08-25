"""
Isolate Python's global `random` module state around a block of code.

Dice.roll() (parchis/game/dice.py) and Player.choose_move()'s random
fallback both call the global `random` module directly, hardcoded --
confirmed by reading both files, and true throughout this codebase (the
opponent_pool.py / pool_seed convention exists specifically to keep
opponent-*selection* randomness on a dedicated random.Random instance,
separate from this same global stream that dice rolls use). That means an
MCTS simulation, which needs to play out many hypothetical dice-roll/random-
opponent-move sequences to evaluate a candidate action, cannot be given a
private RNG without modifying Dice/Player -- out of scope (the plan is
explicitly additive).

isolated_random() achieves the same *effect* -- a simulation's internal
randomness never leaks into or perturbs the real game's own future dice
sequence -- via save/restore of the global module's state instead: each
simulation runs under its own deterministic seed, and the real global state
is restored exactly before returning control to real gameplay.
"""

import random
import zlib
from contextlib import contextmanager


def _to_native_seed(seed):
    """random.seed() only accepts None/int/float/str/bytes/bytearray
    (Python 3.11+) -- callers here often want to combine several parts
    (e.g. a base seed + a simulation index) into one seed, so accept
    anything and deterministically fold it into an int via crc32 (not
    Python's built-in hash(), which is randomized per-process for str/bytes
    unless PYTHONHASHSEED is fixed -- crc32 always gives the same answer
    for the same input, which reproducibility here depends on)."""
    if seed is None or isinstance(seed, (int, float, str, bytes, bytearray)):
        return seed
    return zlib.crc32(repr(seed).encode())


@contextmanager
def isolated_random(seed):
    """Run a block under a fresh, deterministically-seeded global `random`
    state, then restore whatever state was there before -- so code outside
    this block (in particular, the real game's own future Dice.roll() calls)
    never observes any effect from randomness consumed inside it."""
    saved_state = random.getstate()
    try:
        random.seed(_to_native_seed(seed))
        yield
    finally:
        random.setstate(saved_state)
