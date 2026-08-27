"""
Shared recording types for "instrumented" agent factories -- the
visualization-only siblings of make_search_agent_factory (parchis/az/agent.py)
and make_heuristic_agent_factory (parchis/agents/heuristic.py) that capture
what the plain factories compute and discard: a search agent's per-move
value breakdown (parchis.az.search.search's move_values/root_value) or a
heuristic agent's per-move scores (heuristic._score_move).

Lives in parchis/agents/, not parchis/az/, matching the codebase's existing
az -> agents import direction (parchis.az.champion_pool already imports
from parchis.agents.heuristic; the reverse has never been true) -- a
recorder shared by both a search-agent sibling (in parchis/az/agent.py) and
a heuristic-agent sibling (in this same package) belongs on the agents side
of that boundary, not the az side.

Never used on any hot path (self-play generation, promotion matches): those
call search.search()/choose_move_with_weights directly or via the plain
(non-recording) factories, and must stay exactly as fast and exactly as
tested as they already are. This module exists purely so a human-facing
replay (parchis/visualization/instrumented_play.py) can see what an agent
was "thinking" at each decision.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionRecord:
    """One decision an instrumented agent made. `kind` says which of the
    two mutually-exclusive value shapes below is populated:
      - "search": root_value (np.ndarray[num_players], this position's own
        value) and move_values ({piece_id: np.ndarray[num_players]}, the
        value of each legal move) -- both straight from
        parchis.az.search.search()'s return.
      - "heuristic": move_scores ({piece_id: float}, _score_move's raw
        linear-combination score) -- a single scalar per move, not a
        per-seat probability vector; the visualizer must not conflate the
        two (see parchis/visualization/visualizer.py's draw_value_panel)."""
    seat: int
    turn_number: int
    decision_index_in_turn: int
    kind: str
    chosen_piece_id: Optional[int]
    root_value: Optional[object] = None   # np.ndarray[num_players], kind="search" only
    move_values: Optional[dict] = None    # {piece_id: np.ndarray[num_players]}, kind="search" only
    move_scores: Optional[dict] = None    # {piece_id: float}, kind="heuristic" only


class DecisionRecorder:
    """Collects DecisionRecords for one seat across one game. One instance
    per (game, seat), mirroring TurnContextTracker's own per-game-per-seat
    lifetime (parchis/az/turn_context.py)."""

    def __init__(self):
        self.records = []
        self._current_turn = None
        self._counter = 0

    def next_index(self, turn_number):
        """Returns this turn's next decision_index_in_turn (0-based),
        resetting the counter whenever `turn_number` changes from the last
        call. Call exactly once per recorded decision, in turn order --
        matches how choose_move is actually invoked, so no external
        bookkeeping of "which roll this is" is needed by callers."""
        if turn_number != self._current_turn:
            self._current_turn = turn_number
            self._counter = 0
        idx = self._counter
        self._counter += 1
        return idx
