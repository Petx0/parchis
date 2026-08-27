"""
Tracks, purely by observation, whether each successive Player.choose_move
call is a fresh dice roll or a bonus-chain continuation, plus the acting
seat's own six-streak -- shared by parchis/az/agent.py (search-driven play)
and parchis/az/selfplay.py (data generation), both of which need the exact
same (roll, pending_bonus, consecutive_sixes) context to call
parchis.az.encoding.encode() / parchis.az.search.search() correctly.

Extracted rather than duplicated between those two modules (unlike e.g.
parchis/evaluation/arena.py's roll recorder, deliberately duplicated from
mcts.py's own): both live under parchis/az/, so an import between them is
the natural, expected dependency direction, and this specific logic is
exactly the piece §1.4 flags mcts.py for getting wrong ("with
roll_box['last_roll'] still holding the dice roll, ... mcts.search treats a
bonus decision as a turn-start root") -- worth keeping in exactly one
place, not two that could quietly drift apart.

arena.py's roll_box only updates on genuine Dice.roll() calls, never on a
bonus's fixed 10/20 "roll", so a bonus decision can't be detected by
checking whether roll_box changed. Instead, AFTER a move is chosen (but
before it's applied), this checks -- via Game.would_capture /
new_position == FINAL_POSITION, the SAME checks Game.handle_bonus_moves
itself uses -- whether that move will trigger a bonus, and remembers it for
the NEXT choose_move call.

Bug found & fixed (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3), TWICE:

1st attempt (wrong): a bonus can have ZERO legal moves (e.g. a finish bonus
of +10 that no piece can use right now), in which case
Game._execute_bonus_move (game.py) returns without EVER calling
player.choose_move -- so record_move() never ran to clear
self._pending_bonus for that attempt, and the NEXT real choose_move call
would otherwise still be reported as "continuing that same bonus", already
resolved and wrong -- silently, not a crash. First fix tried to self-heal
this INSIDE context_for(), by re-checking game.get_legal_moves(pending_
bonus['squares']) against the CURRENT board right before trusting a
remembered bonus. That re-check is unsound: board state keeps changing
across every subsequent turn (for this seat and others), so "is a move of
this exact SQUARE COUNT coincidentally legal right now" can come back
true for a totally unrelated reason long after the real bonus was already
silently resolved with nothing played -- confirmed by a real crash in
generate_round_games where a genuinely fresh entry-roll decision (roll=5)
was mis-served a stale finish_bonus=10 context because some ON-BOARD piece
happened to have a legal +10 move by then, for reasons having nothing to
do with the original (long-resolved) finish bonus.

2nd attempt (this one): predict correctly, AT THE MOMENT the bonus would
be set, whether Game._execute_bonus_move will actually find a legal move
for it -- by SIMULATING the exact same computation it will do (apply the
move via snapshot/execute_move/restore, matching search.py's and
heuristic.py's own established "peek at a hypothetical future state, then
undo it" technique) and checking game.get_legal_moves on the resulting
board, at record_move() time, when the true answer is unambiguous. If
that comes back empty, the bonus is never recorded as pending at all,
exactly mirroring what Game._execute_bonus_move will silently do a moment
later for real. No re-checking anything later is needed, and context_for()
goes back to trusting a set self._pending_bonus unconditionally.
"""

from parchis.game.board import Board
from parchis.game.constants import BONUS_TURN_ROLL, CAPTURE_BONUS_SQUARES, FINISH_BONUS_SQUARES


class TurnContextTracker:
    """One instance per game per seat (mirrors roll_box's own per-game
    lifetime in arena.py). Not thread-safe / not reentrant -- call
    context_for() then record_move() strictly alternately, once per
    choose_move() invocation."""

    def __init__(self):
        self._pending_bonus = None
        self._streak = 0

    def context_for(self, roll_box):
        """Call FIRST in choose_move(legal_moves), before deciding the
        move. Returns (roll, pending_bonus, consecutive_sixes) ready to
        pass to parchis.az.search.search() / parchis.az.encoding.encode().
        """
        if self._pending_bonus is not None:
            return None, self._pending_bonus, 0
        roll = roll_box["last_roll"]
        self._streak = self._streak + 1 if roll == BONUS_TURN_ROLL else 0
        return roll, None, min(self._streak, 2)

    def record_move(self, game, move):
        """Call AFTER deciding `move` (a legal (piece, new_position,
        move_type) tuple from the CURRENT, not-yet-mutated `game`, or None
        if there was nothing to choose), to prepare state for the NEXT
        context_for() call.

        Only actually sets a pending bonus if it will genuinely be
        offered as a real decision -- see module docstring: predicts
        Game._execute_bonus_move's own "zero legal moves -> silently
        skip, never call choose_move" behavior by simulating the move
        first, rather than recording a bonus that will never actually
        reach a choose_move call."""
        if move is None:
            self._pending_bonus = None
            return
        piece, new_position, move_type = move
        if game.would_capture(move):
            bonus = {'type': 'capture_bonus', 'squares': CAPTURE_BONUS_SQUARES}
        elif new_position == Board.FINAL_POSITION:
            bonus = {'type': 'finish_bonus', 'squares': FINISH_BONUS_SQUARES}
        else:
            bonus = None

        if bonus is not None:
            mover = game.get_current_player()
            snap = game.snapshot()
            game.execute_move(piece, new_position, move_type)
            has_legal_bonus_move = bool(game.get_legal_moves(mover, bonus['squares']))
            game.restore(snap)
            if not has_legal_bonus_move:
                bonus = None  # Game._execute_bonus_move would silently skip this -- see module docstring

        self._pending_bonus = bonus
