"""
CSV loader for the tactical puzzle suite (docs/AGENT_REBUILD_PLAN.md
Part 5.4): turns one hand-authored row into a real, fully-specified `Game`
position plus the decision context an agent needs to answer it -- built by
mutating a real `Game`'s real `Piece`/`Board` objects and validated via the
engine's own real `get_legal_moves`, never a hand-rolled reimplementation
of any rule.

Schema (one row = one puzzle = one decision point):
    puzzle_id, category,
    a_piece_0..a_piece_3 (RED's 4 pieces), b_piece_0..b_piece_3 (YELLOW's),
    turn (A|B), roll (1-6 | capture_bonus | finish_bonus),
    consecutive_sixes (0-2, must be 0 unless roll==6),
    correct_piece_id (0-3, or several 0-3 values separated by '/' when more
    than one move is genuinely correct -- e.g. "2/3"),
    rationale.

Position encoding (same for all 8 piece columns): 0 = in base,
1-68 = main track (shared numbering across colors), 69-75 = that color's
own private home column (the SAME numbers, but a physically different
square per color), 76 = finished (the same physical point -- the hub --
for every color, per Board.FINAL_POSITION).

Colors are fixed: player A is always RED, player B is always YELLOW -- one
of the two pairs Game.__init__ already randomly picks between for a
2-player game. Fixing it removes a whole column, since which specific
color pair is in play never changes any tactic (this whole project's
design -- encoding, search -- is color-invariant throughout).

`correct_piece_id` is deliberately NOT a destination square: the
destination is fully determined by (piece, roll/bonus, board state) under
Parchís rules, so specifying it separately would be redundant and a source
of authoring-typo bugs. The loader computes the real legal moves and
validates EVERY listed correct_piece_id is genuinely one of them -- this
single check, reusing RuleEngine.get_legal_moves (the exact path every
real agent decision goes through), transitively catches bad board setups,
wrong turn/roll/consecutive_sixes combinations, and plain typos, without
reimplementing any rule logic itself.

A puzzle can have more than one genuinely correct answer (e.g. two
equally-safe captures) -- `PuzzleCase.correct_piece_ids` is always a
tuple (never a bare int, even for the single-answer case, so callers
never need two different code paths): `chosen_piece_id in
case.correct_piece_ids` is the one correctness check every consumer
(runner.score_puzzles, visualize_puzzles.render_puzzle) uses. '/' was
chosen as the in-field separator (not ',') specifically so it never
collides with either of the two CSV delimiters _detect_delimiter accepts
(a puzzle author must never need to know or care which delimiter their
own file happens to use when writing a multi-answer cell).

Historical note: RuleEngine.get_legal_moves' home-column occupancy check
used to not filter by color (a puzzle with both a RED and a YELLOW piece
at the same numeric 69-75 slot could have seen a more restrictive
legal_moves set than real Parchís rules produce) -- fixed in rules.py's
_occupancy_count_for_move, see docs/AZ_DESIGN.md's "Home-column
occupancy bug" entry. This loader always deferred to whatever
get_legal_moves computed rather than reimplementing the check itself, so
no loader change was needed when the engine was fixed.
"""

import csv
import glob
import os
from dataclasses import dataclass
from typing import Optional

from parchis.game.board import Board
from parchis.game.constants import CAPTURE_BONUS_SQUARES, FINISH_BONUS_SQUARES
from parchis.game.game import Game

PLAYER_A_COLOR = "RED"
PLAYER_B_COLOR = "YELLOW"
_GAME_BUILD_RETRIES = 20


@dataclass
class PuzzleCase:
    """One fully-specified puzzle: a real Game object (already set up per
    the CSV row) plus the decision context an agent needs to answer it."""
    puzzle_id: str
    category: str
    game: object  # parchis.game.game.Game
    roll: Optional[int]
    pending_bonus: Optional[dict]
    consecutive_sixes: int
    acting_seat: int
    legal_moves: list
    correct_piece_ids: tuple  # tuple[int, ...], sorted, always >= 1 element
    rationale: str


def _place_piece(piece, csv_value, board):
    """Overwrite `piece`'s position/in_base/finished + board registration
    to match one CSV cell, regardless of the piece's current state --
    safe to call unconditionally for all 8 pieces of a freshly-built Game,
    including piece 0 (which Game.__init__ already places ON the board at
    that color's starting square, unlike pieces 1-3 which start in base).

    Piece has no single setter that keeps all three fields consistent on
    its own: mark_finished() sets finished/position but never touches
    in_base (already False by then in normal play, but not necessarily
    true of a piece we're about to relocate FROM finished/base), and
    move_to() never clears a stale finished=True. So each branch below
    sets all three fields explicitly, rather than trusting any one
    existing method to do it completely.
    """
    board.remove_piece(piece)  # no-op if the piece has no current board entry
    if csv_value == 0:
        piece.send_to_base()  # already sets position/in_base/finished consistently
        return
    if csv_value == Board.FINAL_POSITION:
        piece.finished = True
        piece.in_base = False
        piece.position = Board.FINAL_POSITION
        board.add_piece(piece, Board.FINAL_POSITION)
        return
    piece.finished = False
    piece.move_to(csv_value)  # sets position, clears in_base if it was set
    board.add_piece(piece, csv_value)


def _build_two_player_game():
    """A fresh Game(num_players=2), retried (bounded) until it lands on
    the RED/YELLOW color pair -- Game.__init__ picks RED/YELLOW or
    BLUE/GREEN via a bare, unseeded random.choice() with no hook to force
    one or the other. ~2 attempts expected; a failure after
    _GAME_BUILD_RETRIES would mean Game.__init__'s pairing logic changed,
    not ordinary bad luck."""
    for _ in range(_GAME_BUILD_RETRIES):
        game = Game(num_players=2)
        colors = {p.color for p in game.players}
        if colors == {PLAYER_A_COLOR, PLAYER_B_COLOR}:
            return game
    raise RuntimeError(
        f"Could not construct a {PLAYER_A_COLOR}/{PLAYER_B_COLOR} 2-player Game "
        f"in {_GAME_BUILD_RETRIES} attempts -- Game.__init__'s color-pairing logic "
        f"may have changed."
    )


def _puzzle_id_of(row):
    return (row.get("puzzle_id") or "?").strip()


def _parse_positions(row, prefix):
    values = []
    puzzle_id = _puzzle_id_of(row)
    for i in range(4):
        key = f"{prefix}_piece_{i}"
        raw = row.get(key)
        try:
            v = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"puzzle {puzzle_id}: {key} must be an integer, got {raw!r}")
        if not (0 <= v <= Board.FINAL_POSITION):
            raise ValueError(f"puzzle {puzzle_id}: {key}={v} out of range [0, {Board.FINAL_POSITION}]")
        values.append(v)
    return values


def _parse_roll(row):
    puzzle_id = _puzzle_id_of(row)
    raw = (row.get("roll") or "").strip()
    if raw == "capture_bonus":
        return None, {"type": "capture_bonus", "squares": CAPTURE_BONUS_SQUARES}
    if raw == "finish_bonus":
        return None, {"type": "finish_bonus", "squares": FINISH_BONUS_SQUARES}
    try:
        roll = int(raw)
    except ValueError:
        raise ValueError(
            f"puzzle {puzzle_id}: roll must be 1-6, 'capture_bonus', or 'finish_bonus', "
            f"got {raw!r}"
        )
    if not (1 <= roll <= 6):
        raise ValueError(f"puzzle {puzzle_id}: roll={roll} out of range [1, 6]")
    return roll, None


def load_puzzle_row(row):
    """One csv.DictReader row (dict of strings) -> PuzzleCase. Raises
    ValueError, naming the puzzle_id, on any validation failure."""
    puzzle_id = _puzzle_id_of(row)
    if not puzzle_id or puzzle_id == "?":
        raise ValueError(f"puzzle row missing puzzle_id: {row}")
    category = (row.get("category") or "").strip()

    a_positions = _parse_positions(row, "a")
    b_positions = _parse_positions(row, "b")

    turn = (row.get("turn") or "").strip().upper()
    if turn not in ("A", "B"):
        raise ValueError(f"puzzle {puzzle_id}: turn must be 'A' or 'B', got {row.get('turn')!r}")

    roll, pending_bonus = _parse_roll(row)

    try:
        consecutive_sixes = int(row.get("consecutive_sixes"))
    except (TypeError, ValueError):
        raise ValueError(f"puzzle {puzzle_id}: consecutive_sixes must be an integer")
    if consecutive_sixes not in (0, 1, 2):
        raise ValueError(f"puzzle {puzzle_id}: consecutive_sixes={consecutive_sixes} must be 0, 1, or 2")
    if roll != 6 and consecutive_sixes != 0:
        raise ValueError(
            f"puzzle {puzzle_id}: consecutive_sixes={consecutive_sixes} but roll="
            f"{row.get('roll')!r} is not 6 -- the real engine only ever reports a "
            f"nonzero streak on a fresh roll of 6 (TurnContextTracker.context_for "
            f"resets it to 0 on any other roll), so this combination can never occur "
            f"in real play"
        )

    game = _build_two_player_game()
    red = next(p for p in game.players if p.color == PLAYER_A_COLOR)
    yellow = next(p for p in game.players if p.color == PLAYER_B_COLOR)
    for i in range(4):
        _place_piece(red.pieces[i], a_positions[i], game.board)
        _place_piece(yellow.pieces[i], b_positions[i], game.board)

    acting_player = red if turn == "A" else yellow
    acting_seat = game.players.index(acting_player)
    # search.search()/encoding.encode() read the mover off Game state
    # (current_player_idx), not a parameter -- see search.py's own
    # `player = game.get_current_player(); mover_seat = game.current_player_idx`.
    # turn_number is deliberately left at Game()'s default 0: a documented
    # simplification (a trained net's leaf evaluations see turn_number=0
    # regardless of how late-game the position looks), not fixed here --
    # extending the schema with an optional turn_number column later would
    # be a small, backward-compatible loader change if this ever matters.
    game.current_player_idx = acting_seat

    effective_value = roll if pending_bonus is None else pending_bonus["squares"]
    legal_moves = game.get_legal_moves(acting_player, effective_value)

    raw_correct = (row.get("correct_piece_id") or "").strip()
    try:
        correct_piece_ids = tuple(sorted({int(x) for x in raw_correct.split("/")}))
    except ValueError:
        raise ValueError(
            f"puzzle {puzzle_id}: correct_piece_id must be one or more integers "
            f"separated by '/' (e.g. '2' or '2/3'), got {raw_correct!r}"
        )
    out_of_range = [pid for pid in correct_piece_ids if not (0 <= pid <= 3)]
    if out_of_range:
        raise ValueError(f"puzzle {puzzle_id}: correct_piece_id(s) {out_of_range} must be 0-3")

    legal_ids = {m[0].piece_id for m in legal_moves}
    illegal = [pid for pid in correct_piece_ids if pid not in legal_ids]
    if illegal:
        raise ValueError(
            f"puzzle {puzzle_id}: correct_piece_id(s) {illegal} not among the "
            f"actual legal moves for this position (legal piece_ids: {sorted(legal_ids)}) -- "
            f"check the board setup, turn, roll, and consecutive_sixes"
        )

    rationale = (row.get("rationale") or "").strip()

    return PuzzleCase(
        puzzle_id=puzzle_id, category=category, game=game, roll=roll,
        pending_bonus=pending_bonus, consecutive_sixes=consecutive_sixes,
        acting_seat=acting_seat, legal_moves=legal_moves,
        correct_piece_ids=correct_piece_ids, rationale=rationale,
    )


def _csv_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.csv")))
    return [path]


_DELIMITER_CANDIDATES = (",", ";")


def _detect_delimiter(header_line, filepath):
    """','  is the documented schema delimiter (this is a CSV suite), but
    spreadsheet software exports ';'-delimited "CSV" by default in many
    locales (anywhere ',' is the decimal separator) -- a real puzzle file
    authored in a spreadsheet and saved as CSV hit exactly this. Detected
    by checking which candidate actually splits the header into a first
    column literally named 'puzzle_id' -- not a byte-count heuristic
    (csv.Sniffer would get confused by a comma legitimately appearing
    inside a semicolon-delimited file's own rationale text, which real
    puzzles do)."""
    for delimiter in _DELIMITER_CANDIDATES:
        if header_line.split(delimiter)[0].strip() == "puzzle_id":
            return delimiter
    raise ValueError(
        f"{filepath}: couldn't detect the CSV delimiter -- expected the header's "
        f"first column to be 'puzzle_id' using ',' or ';' as the separator, got "
        f"header line: {header_line!r}"
    )


def load_puzzles(path):
    """path: a single .csv file, or a directory of them (globbed, sorted
    for deterministic ordering across a filesystem). Returns
    list[PuzzleCase]. Validates puzzle_id uniqueness GLOBALLY across every
    row from every file, not just within one file.

    Each file is opened with encoding='utf-8-sig' (strips a leading UTF-8
    BOM if present -- a no-op otherwise -- another common byte-for-byte
    artifact of the same spreadsheet-export path _detect_delimiter's own
    docstring describes) and its delimiter auto-detected independently
    (different puzzle files may use different delimiters; each is only
    ever read with its own)."""
    cases = []
    seen_in = {}
    for filepath in _csv_files(path):
        with open(filepath, encoding="utf-8-sig", newline="") as f:
            header_line = f.readline()
            delimiter = _detect_delimiter(header_line, filepath)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                case = load_puzzle_row(row)
                if case.puzzle_id in seen_in:
                    raise ValueError(
                        f"duplicate puzzle_id {case.puzzle_id!r} in {filepath} "
                        f"(already seen in {seen_in[case.puzzle_id]})"
                    )
                seen_in[case.puzzle_id] = filepath
                cases.append(case)
    if not cases:
        raise ValueError(f"no puzzles found at {path!r}")
    return cases
