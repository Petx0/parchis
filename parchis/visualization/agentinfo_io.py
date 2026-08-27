"""
Sidecar JSON for per-decision agent value data (search's move_values/
root_value, or a heuristic's move_scores) alongside a GameLogger-produced
move log -- see docs/AGENT_REBUILD_PLAN.md's visualization plan for why
this is a SEPARATE file rather than an extra field on GameLogger's own
schema (parchis/utils/logger.py):

- parchis/game/records.py and parchis/utils/logger.py have zero dependents
  outside parchis/game/parchis/utils/parchis/search/mcts.py and no
  dependency on numpy or parchis.az -- their whole design intent is "what
  actually happened," not "what an evaluator thought about it."
- docs/CODE_REVIEW.md already flags that contract as having "no shared
  constants or version field." A new, small, purpose-built, EXPLICITLY
  versioned sidecar fixes that problem for this feature instead of growing
  the existing one; GameLogger itself also gained a schema_version key
  (parchis/utils/logger.py) at the same time, for the same reason.
- A game played with default/random players (or any historical log already
  on disk) produces exactly the same GameLogger output as before -- zero
  regression risk.

File convention: `<log_stem>.agentinfo.json` next to the log file, e.g.
`logs/game_20260827_100159_BLUE.json` -> `logs/game_20260827_100159_BLUE.agentinfo.json`.

Schema (SCHEMA_VERSION=1):
{
  "schema_version": 1,
  "num_players": 2,
  "seats": {
    "0": {
      "agent_label": "champion (search depth=1)",
      "decisions": [
        {"seat": 0, "turn_number": 3, "decision_index_in_turn": 0, "kind": "search",
         "root_value": [0.52, 0.48], "move_values": {"0": [0.55, 0.45], "2": [0.5, 0.5]},
         "chosen_piece_id": 0},
        {"seat": 0, "turn_number": 3, "decision_index_in_turn": 1, "kind": "heuristic",
         "move_scores": {"1": 0.8, "3": -0.2}, "chosen_piece_id": 1}
      ]
    }
  }
}

Indexing: within one seat's turn, `decision_index_in_turn` is "the Nth time
this seat's choose_move actually produced a recorded decision this turn"
(parchis.agents.decision_recorder.DecisionRecorder.next_index). On the
replay side, matching a RollEntry (as logged by GameLogger) to a decision
needs exactly one rule, verified directly against parchis/game/game.py:

- A plain dice roll always calls player.choose_move(legal_moves), even
  when legal_moves is empty (game.py's play_turn) -- but a recording
  factory only ever APPENDS a record when legal_moves was non-empty (same
  early-return shape as the plain, non-recording factories), so an empty
  legal_moves never produces a decision to match against.
- A bonus roll skips choose_move entirely when legal_moves is empty
  (game.py's _execute_bonus_move) -- likewise never a decision.
- The three-consecutive-sixes penalty roll is a plain roll with
  legal_moves_count=0 (game.py appends its RollEntry without calling
  choose_move at all in that one special case) -- already covered by the
  first bullet, no separate case needed.

So the one rule below correctly covers every skip case without having to
name any of them individually.

IMPORTANT -- seat vs. player_id: DecisionRecord.seat (and this sidecar's
top-level "seats" keys) use the SAME "seat" convention as the rest of this
codebase's agent-factory contract (parchis/evaluation/arena.py,
parchis/az/agent.py, parchis/agents/heuristic.py): seat 0 is whoever moves
FIRST in this particular game, seat 1 the next in turn order, etc. -- a
fixed per-game turn-order position. GameLogger's own `player_id` (recorded
in every turn_data dict) is a DIFFERENT, older identity: a fixed
color-slot label assigned once in Player.__init__, BEFORE Game.__init__
rotates its `self.players` list so the dice-determined starting player
ends up at list index 0 (see Game.__init__'s "Rotate players list so
starting player is first"). Those two numberings only coincide when the
dice-determined starter's player_id already happens to be 0 -- true by
chance roughly 1/num_players of the time, false the rest. Conflating them
(matching a DecisionRecord's seat directly against turn_data['player_id'])
was an actual bug caught after this feature shipped: it silently failed to
match almost every decision whenever that chance coincidence didn't hold,
making the value panel look empty/broken for no apparent reason.

The fix: build_seat_by_player_id() below derives the correct
{player_id: seat} mapping directly from the log's own turn order (the
first turn's player_id is seat 0, the next NEW player_id encountered is
seat 1, etc. -- turn order cycles through every seat exactly once before
repeating, so this is exact, not a heuristic) -- decision_for_roll() and
replay_game_from_log()'s per-seat color lookup both go through this
mapping instead of ever comparing player_id and seat directly.
"""

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def agentinfo_path_for(log_filepath):
    """`<log_stem>.agentinfo.json`, next to the log file itself."""
    log_path = Path(log_filepath)
    return log_path.with_suffix("").with_suffix(".agentinfo.json")


def _decision_record_to_dict(record):
    d = {
        "seat": record.seat,
        "turn_number": record.turn_number,
        "decision_index_in_turn": record.decision_index_in_turn,
        "kind": record.kind,
        "chosen_piece_id": record.chosen_piece_id,
    }
    if record.root_value is not None:
        d["root_value"] = np.asarray(record.root_value).tolist()
    if record.move_values is not None:
        d["move_values"] = {
            str(piece_id): np.asarray(v).tolist() for piece_id, v in record.move_values.items()
        }
    if record.move_scores is not None:
        d["move_scores"] = {str(piece_id): float(v) for piece_id, v in record.move_scores.items()}
    return d


def save_agentinfo(recorders_by_seat, agent_labels, log_filepath, num_players):
    """recorders_by_seat: {seat: DecisionRecorder}, only instrumented seats
    (an un-instrumented/random seat is simply absent -- see
    decision_for_roll's graceful-degradation contract below).
    agent_labels: {seat: str}, a human-readable description of that seat's
    agent (shown by the visualizer). Writes `<log_stem>.agentinfo.json` and
    returns its path."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "num_players": num_players,
        "seats": {
            str(seat): {
                "agent_label": agent_labels.get(seat, f"seat {seat}"),
                "decisions": [_decision_record_to_dict(r) for r in recorder.records],
            }
            for seat, recorder in recorders_by_seat.items()
        },
    }
    path = agentinfo_path_for(log_filepath)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def load_agentinfo(log_filepath):
    """Returns the parsed sidecar dict, or None if it doesn't exist -- the
    graceful-degradation entry point every caller should check before
    assuming any agent data is available at all."""
    path = agentinfo_path_for(log_filepath)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_seat_by_player_id(turns):
    """{player_id: seat}, derived from `turns` (log_data['turns']) alone --
    see this module's docstring for why player_id and seat are different
    numberings that must never be compared directly. Turn order visits
    every seat exactly once, in seat order, before repeating (Game cycles
    current_player_idx = (current_player_idx + 1) % num_players starting
    from 0 = the seat-0/starting player), so "seat = order of first
    appearance in turn order" is exact for any game of at least one full
    round -- not a heuristic or an approximation."""
    seat_by_player_id = {}
    for turn_data in turns:
        pid = turn_data["player_id"]
        if pid not in seat_by_player_id:
            seat_by_player_id[pid] = len(seat_by_player_id)
    return seat_by_player_id


def _roll_had_decision(roll_data):
    """True iff this RollEntry (as logged by GameLogger) corresponds to a
    real recorded decision -- see this module's docstring for the full
    derivation. Simplifies to one check: any roll (bonus or plain, for any
    reason including the three-sixes penalty) with legal_moves_count == 0
    never had a decision recorded against it."""
    return roll_data.get("legal_moves_count", 0) > 0


def decision_for_roll(turn_data, roll_idx, agentinfo_data, seat_by_player_id):
    """Returns the DecisionRecord dict (as saved by save_agentinfo) matching
    turn_data['rolls'][roll_idx], or None whenever: there's no sidecar at
    all (agentinfo_data is None), the acting player_id has no known seat
    (shouldn't happen for a well-formed log, see build_seat_by_player_id),
    the acting seat wasn't instrumented, or this specific roll had no
    decision (see _roll_had_decision). Walks every roll of this turn UP TO
    AND INCLUDING roll_idx to compute the right decision_index_in_turn -- a
    single linear counter, since only the turn's own acting player ever has
    a decision on any of its rolls.

    seat_by_player_id: {player_id: seat}, from build_seat_by_player_id --
    REQUIRED, since turn_data['player_id'] must never be compared directly
    against a DecisionRecord's seat (see this module's docstring)."""
    if agentinfo_data is None:
        return None

    seat = seat_by_player_id.get(turn_data["player_id"])
    if seat is None:
        return None
    seat_data = agentinfo_data.get("seats", {}).get(str(seat))
    if seat_data is None:
        return None

    roll_data = turn_data["rolls"][roll_idx]
    if not _roll_had_decision(roll_data):
        return None

    decision_index = sum(
        1 for r in turn_data["rolls"][:roll_idx] if _roll_had_decision(r)
    )
    turn_number = turn_data["turn_number"]
    for decision in seat_data["decisions"]:
        if decision["turn_number"] == turn_number and decision["decision_index_in_turn"] == decision_index:
            return decision
    return None
