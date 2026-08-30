"""
Canonical, path-relative observation encoding (docs/AGENT_REBUILD_PLAN.md
§2.1) -- replaces the absolute-square encoding parchis/rl/env.py uses for
this package's own value/policy network.

Everything is expressed from the perspective of one *observer seat*, along
that seat's own 72-position path. §1.2 establishes why this is possible at
all: starting squares are exactly 17 apart, every colour's own
start-to-home-entry distance is 63 steps, and every colour's safe squares
and other-starts land at the same relative offsets -- so a single relative
geometry serves every colour, and the network only ever has to learn it
once. The one part this relative frame cannot express is each seat's
private home column (69-76 are reused, unshared, numbers per colour), which
is why it gets its own separate per-seat block below rather than folding
into the 68-slot shared track.

encode() is a pure function of a Game -- no ParchisEnv instance, no
puppeting (parchis/search/state_view.py's ObservationAdapter trick does not
apply here): callers pass the roll/pending-bonus/six-streak state
explicitly, exactly the three pieces of turn state Game itself does not
track on the instance (Game.play_turn() keeps them as local variables --
see game.py's docstring).

Layout (fixed block order), for N = game.num_players:
    own_pieces    : 4 x NUM_OWN_PIECE_FEATURES   (40, independent of N)
    track         : N x 68                        (shared-track occupancy)
    home_columns  : N x 8                         (each seat's private lane)
    per_seat      : N x NUM_PER_SEAT_SCALARS       (6 per seat)
    turn_context  : NUM_TURN_CONTEXT_FEATURES      (12, independent of N)

Every N-wide block uses the SAME seat ordering: index 0 is the observer's
own seat, index k is the seat k turns after the observer in play order
(`_ordered_seats`) -- a fixed geometry independent of whose turn it
actually is right now (that's what "canonical" buys: the network sees one
stable layout per observer, never one that reshuffles turn to turn). Turn
state -- whose decision this actually is, right now -- lives only in the
turn_context block's own scalar, not in the block ordering.
"""

import numpy as np

from parchis.game.board import Board
from parchis.game.constants import BONUS_TURN_ROLL

NUM_OWN_PIECE_FEATURES = 10
NUM_PER_SEAT_SCALARS = 6
NUM_TURN_CONTEXT_FEATURES = 12
TRACK_SLOTS = Board.MAIN_TRACK_SIZE       # 68
HOME_COLUMN_SLOTS = Board.HOME_COLUMN_SIZE  # 8

# Own-piece feature order (see _own_piece_features for definitions).
OWN_PIECE_FEATURE_NAMES = (
    "in_base", "finished", "s_normalized", "steps_to_home_entry_normalized",
    "steps_to_finish_normalized", "in_home_column", "on_safe_square",
    "is_stacked_with_own", "capture_threat_score", "blockade_forced_on_6",
)
# Per-seat scalar order (see _per_seat_scalars for definitions).
PER_SEAT_SCALAR_NAMES = (
    "pieces_in_base", "pieces_on_board", "pieces_finished",
    "mean_path_progress", "max_piece_progress", "own_blockades_on_track",
)

_PATH_LENGTH = 71  # s ranges [0, 71]; 63 = own home-entry square (§1.2)


def own_piece_block_size():
    return 4 * NUM_OWN_PIECE_FEATURES


def track_block_size(num_players):
    return num_players * TRACK_SLOTS


def home_column_block_size(num_players):
    return num_players * HOME_COLUMN_SLOTS


def per_seat_scalar_block_size(num_players):
    return num_players * NUM_PER_SEAT_SCALARS


def turn_context_block_size():
    return NUM_TURN_CONTEXT_FEATURES


def encoding_size(num_players):
    """Total encode() output length for a given player count. 2p=216 is a
    coincidence of this exact feature list, not a rounded target (the plan's
    own §2.1 estimate -- "2p ~220, 4p ~430" -- was an approximation; the
    actual totals this module produces are close but not identical, since
    none of the ~10 "own piece" features scale with N. See docs/AZ_DESIGN.md."""
    return (own_piece_block_size() + track_block_size(num_players)
            + home_column_block_size(num_players) + per_seat_scalar_block_size(num_players)
            + turn_context_block_size())


def block_offsets(num_players):
    """{block_name: start_index} in the fixed layout order, plus 'total'."""
    own = 0
    track = own + own_piece_block_size()
    home = track + track_block_size(num_players)
    per_seat = home + home_column_block_size(num_players)
    turn = per_seat + per_seat_scalar_block_size(num_players)
    total = turn + turn_context_block_size()
    return {
        "own_pieces": own, "track": track, "home_columns": home,
        "per_seat": per_seat, "turn_context": turn, "total": total,
    }


def _ordered_seats(observer_seat, num_players):
    """[observer_seat, next seat in play order, ...] -- a fixed rotation
    independent of game.current_player_idx, so the same physical seat
    always lands in the same channel for a given observer regardless of
    whose turn it actually is right now."""
    return [(observer_seat + k) % num_players for k in range(num_players)]


def _relative_track_index(abs_pos, observer_start):
    """j in [0, 67]: a shared-main-track absolute position, relative to the
    OBSERVER's own start (used for every seat's pieces in the track block,
    not just the observer's own -- this is what makes the geometry
    canonical: opponent pieces/home-entries land at fixed relative offsets
    no matter which colour the observer is)."""
    return (abs_pos - observer_start) % Board.MAIN_TRACK_SIZE


def _own_path_step(piece, observer_start):
    """s in [0, 71] along the OBSERVER's own 72-position path (§1.2): 0 =
    own start, 0..63 shared track (63 = own home-entry square, falls out of
    the same modular formula with no special-casing), 64..71 = the private
    home column (absolute 69..76). Caller handles in_base separately (no
    path position while off the board)."""
    if piece.finished:
        return _PATH_LENGTH  # 71
    if piece.position >= Board.HOME_COLUMN_START:
        return 64 + (piece.position - Board.HOME_COLUMN_START)
    return _relative_track_index(piece.position, observer_start)


def _capture_threat_scores(game, observer_player):
    """Exact port of parchis/rl/env.py::ParchisEnv._capture_threat_scores'
    algorithm to a plain function of (game, player) -- no ParchisEnv
    instance, matching this module's "pure function of a Game" contract
    (§2.1). Duplicated rather than imported: importing ParchisEnv into
    parchis/az/ (a package meant to eventually replace the PPO+MCTS stack
    ParchisEnv was built for) would be an unwanted dependency direction,
    and the original is a bound method that needs a live env instance to
    call. See that method's docstring for the full rule-exactness
    rationale and its two documented, deliberately-accepted scope limits
    (no same-piece bonus-chain continuation; double threats from different
    opponents are summed, not deduplicated).

    Returns:
        dict[int, int]: piece_id -> raw hit count (0..6 per opponent,
        summed across opponents), matching the original's contract exactly
        (not yet divided by 6 or clipped -- callers do that).
    """
    from parchis.game.constants import CAPTURE_BONUS_SQUARES, FINISH_BONUS_SQUARES

    hit_counts = {piece.piece_id: 0 for piece in observer_player.pieces}

    for opponent in game.players:
        if opponent is observer_player:
            continue

        moves_by_roll = {v: game.get_legal_moves(opponent, v) for v in range(1, 7)}
        direct_targets_by_roll = {
            v: {p for m in moves for p in game.would_capture(m)}
            for v, moves in moves_by_roll.items()
        }
        capture_rolls = {v for v, targets in direct_targets_by_roll.items() if targets}
        finish_rolls = {
            v for v, moves in moves_by_roll.items()
            if any(move_type == 'finish' for (_p, _pos, move_type) in moves)
        }

        bonus20_targets = set()
        if capture_rolls:
            bonus20_moves = game.get_legal_moves(opponent, CAPTURE_BONUS_SQUARES)
            bonus20_targets = {p for m in bonus20_moves for p in game.would_capture(m)}
        bonus10_targets = set()
        if finish_rolls:
            bonus10_moves = game.get_legal_moves(opponent, FINISH_BONUS_SQUARES)
            bonus10_targets = {p for m in bonus10_moves for p in game.would_capture(m)}

        for piece in observer_player.pieces:
            for v in range(1, 7):
                hit = (
                    piece in direct_targets_by_roll[v]
                    or (v in capture_rolls and piece in bonus20_targets)
                    or (v in finish_rolls and piece in bonus10_targets)
                )
                if hit:
                    hit_counts[piece.piece_id] += 1

    return hit_counts


def _own_piece_features(game, observer_player, observer_start):
    """(4, NUM_OWN_PIECE_FEATURES) array, row-indexed strictly by piece_id
    (fixed slot, matching parchis/rl/env.py's own convention and this
    project's Discrete(4) action space) -- see OWN_PIECE_FEATURE_NAMES."""
    features = np.zeros((4, NUM_OWN_PIECE_FEATURES), dtype=np.float32)
    blockades = game.get_blockades()
    legal_on_6 = game.get_legal_moves(observer_player, BONUS_TURN_ROLL)
    forced_on_6_ids = {p.piece_id for p, _np, _mt in legal_on_6}
    threat_hits = _capture_threat_scores(game, observer_player)

    for piece in observer_player.pieces:
        row = features[piece.piece_id]
        if piece.in_base:
            row[0] = 1.0  # in_base
            row[2] = 0.0  # s_normalized
            row[3] = 1.0  # steps_to_home_entry_normalized (max remaining)
            row[4] = 1.0  # steps_to_finish_normalized (max remaining)
            continue

        s = _own_path_step(piece, observer_start)
        row[1] = 1.0 if piece.finished else 0.0
        row[2] = s / _PATH_LENGTH
        row[3] = max(0.0, (63 - s)) / 63.0
        row[4] = (_PATH_LENGTH - s) / _PATH_LENGTH
        row[5] = 1.0 if (piece.finished or piece.position >= Board.HOME_COLUMN_START) else 0.0

        # on_safe_square: mirrors env.py's exact convention (in_base/finished
        # pieces are NOT flagged here either -- finished-ness is its own
        # separate feature above).
        if not piece.finished and (piece.position in Board.SAFE_SQUARES
                                    or piece.position >= Board.HOME_COLUMN_START):
            row[6] = 1.0

        if not piece.finished:
            row[7] = 1.0 if any(
                other is not piece and not other.in_base and not other.finished
                and other.position == piece.position
                for other in observer_player.pieces
            ) else 0.0

        row[8] = min(threat_hits[piece.piece_id] / 6.0, 1.0)

        if not piece.finished and piece.position in blockades:
            row[9] = 1.0 if piece.piece_id in forced_on_6_ids else 0.0

    return features


def _relative_piece_progress(piece, owner_start):
    """Progress along THIS PIECE's own colour's path, in [0, 1] -- unlike
    parchis.rl.rewards.calculate_normalized_progress (which divides the
    RAW ABSOLUTE position by 76, only actually monotonic-with-progress for
    a colour whose path never wraps the 1-68 boundary before reaching home,
    i.e. Yellow), this uses the same relative path-step transform as the
    own-piece block (_own_path_step), so it stays correct -- and, crucially
    for this module, colour-invariant -- for every colour."""
    if piece.finished:
        return 1.0
    if piece.in_base:
        return 0.0
    return _own_path_step(piece, owner_start) / _PATH_LENGTH


def _per_seat_scalars(player, owner_start):
    """(NUM_PER_SEAT_SCALARS,) for one seat -- see PER_SEAT_SCALAR_NAMES."""
    scalars = np.zeros(NUM_PER_SEAT_SCALARS, dtype=np.float32)
    scalars[0] = len(player.get_pieces_in_base()) / 4.0
    scalars[1] = len(player.get_pieces_on_board()) / 4.0
    scalars[2] = len(player.get_finished_pieces()) / 4.0
    piece_progresses = [_relative_piece_progress(p, owner_start) for p in player.pieces]
    scalars[3] = sum(piece_progresses) / 4.0
    scalars[4] = max(piece_progresses)

    own_blockade_squares = sum(
        1 for sq in Board.SAFE_SQUARES
        if len(player_pieces_at := [p for p in player.pieces
                                     if not p.in_base and not p.finished and p.position == sq]) == 2
    )
    scalars[5] = own_blockade_squares / 2.0
    return scalars


def encode(game, observer_seat, roll=None, pending_bonus=None, consecutive_sixes=0):
    """
    Encode `game` from `observer_seat`'s perspective as a flat float32
    array of length encoding_size(game.num_players).

    Args:
        game: a parchis.game.Game (read-only: never mutated).
        observer_seat: index into game.players. roll/pending_bonus/
            consecutive_sixes below are interpreted as describing THIS
            seat's own current decision (typically game.current_player_idx
            -- see module docstring -- but deliberately decoupled for
            testing, e.g. colour-invariance, and for any future use that
            wants a non-acting seat's own encoding).
        roll: the current die roll (1-6), or None if no fresh roll is
            pending (e.g. mid-bonus-chain).
        pending_bonus: None, or {'type': 'capture_bonus'|'finish_bonus',
            'squares': 20|10} -- same shape ParchisEnv uses.
        consecutive_sixes: the observer's own current six-streak (0-2).

    Returns:
        np.ndarray, shape (encoding_size(game.num_players),), dtype float32.
    """
    num_players = game.num_players
    offsets = block_offsets(num_players)
    out = np.zeros(offsets["total"], dtype=np.float32)

    observer_player = game.players[observer_seat]
    observer_start = Board.STARTING_POSITIONS[observer_player.color]
    ordered_seats = _ordered_seats(observer_seat, num_players)

    # --- own_pieces: 4 x NUM_OWN_PIECE_FEATURES ---
    own_features = _own_piece_features(game, observer_player, observer_start)
    out[offsets["own_pieces"]:offsets["track"]] = own_features.reshape(-1)

    # --- track: num_players x 68, channel order = ordered_seats ---
    track = out[offsets["track"]:offsets["home_columns"]].reshape(num_players, TRACK_SLOTS)
    for channel, seat in enumerate(ordered_seats):
        player = game.players[seat]
        for piece in player.pieces:
            if piece.in_base or piece.finished or piece.position >= Board.HOME_COLUMN_START:
                continue
            j = _relative_track_index(piece.position, observer_start)
            track[channel, j] += 0.5

    # --- home_columns: num_players x 8, same channel order ---
    home = out[offsets["home_columns"]:offsets["per_seat"]].reshape(num_players, HOME_COLUMN_SLOTS)
    for channel, seat in enumerate(ordered_seats):
        player = game.players[seat]
        for piece in player.pieces:
            # Finished pieces excluded (mirrors parchis/rl/env.py's own
            # board-state convention exactly): up to 4 same-colour pieces
            # can legally occupy square 76 at once (no MAX_PIECES_PER_SQUARE
            # cap there -- see RuleEngine.get_legal_moves), which the
            # 0/0.5/1.0-capped-at-2 scale below can't represent, and
            # finished-piece COUNT is already carried by per_seat scalars
            # and the own-piece block's own `finished` flag.
            if piece.in_base or piece.finished or piece.position < Board.HOME_COLUMN_START:
                continue
            idx = piece.position - Board.HOME_COLUMN_START
            home[channel, idx] += 0.5

    # --- per_seat: num_players x NUM_PER_SEAT_SCALARS, same channel order ---
    per_seat = out[offsets["per_seat"]:offsets["turn_context"]].reshape(num_players, NUM_PER_SEAT_SCALARS)
    for channel, seat in enumerate(ordered_seats):
        seat_player = game.players[seat]
        seat_start = Board.STARTING_POSITIONS[seat_player.color]
        per_seat[channel] = _per_seat_scalars(seat_player, seat_start)

    # --- turn_context: NUM_TURN_CONTEXT_FEATURES ---
    turn = out[offsets["turn_context"]:offsets["total"]]
    if roll is not None:
        if roll < 6:
            turn[roll - 1] = 1.0
        else:
            if len(observer_player.get_pieces_in_base()) > 0:
                turn[5] = 1.0  # is_dice_6_normal
            else:
                turn[6] = 1.0  # is_dice_6_no_base (effective_roll=7)
    if pending_bonus is not None:
        if pending_bonus['type'] == 'finish_bonus':
            turn[7] = 1.0
        elif pending_bonus['type'] == 'capture_bonus':
            turn[8] = 1.0
    turn[9] = consecutive_sixes / 3.0  # THREE_SIXES_LIMIT, matching ParchisEnv's own scale
    turn[10] = ((game.current_player_idx - observer_seat) % num_players) / num_players
    turn[11] = min(game.turn_number / 300.0, 1.0)

    return out
