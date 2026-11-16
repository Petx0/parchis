"""PILA-10 diagnostic: how often does the turn state the search/encoding model
diverge from the rules, and does it change moves?

Two rule facts are checked against parchis/az/search.py and encoding.py:

1. Six-again after a bonus. docs/RULES.md and Game.play_turn: a roll of 6 earns a
   reroll even when the move triggers a capture/finish bonus (keep_rolling =
   dice_roll == 6, after the bonus chain). search.py decides six-again with
   `pending_bonus is None and roll == 6`; the bonus decision is entered with
   roll=None, so after the bonus the search passes the turn to the opponent.
   TurnContextTracker also hands bonus decisions consecutive_sixes=0, so the
   encoding cannot tell "reroll owed" from "turn ends".
2. Three-sixes penalty target. After the second six, the piece moved on that
   roll is at risk (1/6). The encoding has the streak but not which piece.

The script plays S1-vs-S1 games at depth 1 (one process, one BLAS thread),
counts the affected decisions, and at each 6-roll decision with a capturing or
finishing move compares depth-2 search as shipped against a patched copy that
carries the reroll through the bonus chain (patch lives only in this script).
"""
import argparse, inspect, json, os, random, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from parchis.az import search, encoding
from parchis.az.agent import NetEvaluator
from parchis.az.champion_pool import load_numpy_net
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation.arena import play_one_game
from parchis.game.board import Board

orig = {n: getattr(search, n) for n in ("_expand_decision", "_resolve_turn_continuation", "afterstate_context")}


def build_patched():
    ns = search.__dict__
    patched = {}
    src = inspect.getsource(orig["_expand_decision"])
    src = src.replace("node = _decision_value(\n                game, None, {'type': 'capture_bonus'",
                      "node = _decision_value(\n                game, roll, {'type': 'capture_bonus'")
    src = src.replace("node = _decision_value(\n                game, None, {'type': 'finish_bonus'",
                      "node = _decision_value(\n                game, roll, {'type': 'finish_bonus'")
    assert src.count("game, roll, {'type'") == 2, "patch anchor for _expand_decision not found"
    for name, s in (("_expand_decision", src),
                    ("_resolve_turn_continuation", inspect.getsource(orig["_resolve_turn_continuation"])),
                    ("afterstate_context", inspect.getsource(orig["afterstate_context"]))):
        s2 = s.replace("six_again = (pending_bonus is None and roll == BONUS_TURN_ROLL)",
                       "six_again = (roll == BONUS_TURN_ROLL)")
        if name != "_expand_decision":
            assert s2 != s, name
        loc = {}
        exec(s2, ns, loc)
        patched[name] = loc[name]
    return patched


def use(fns):
    for n, f in fns.items():
        setattr(search, n, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/soups_2026-11-16/s1/model.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=91_000_000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    net = load_numpy_net(a.model, encoding.encoding_size(2), 2, (256, 256))
    ev = NetEvaluator(net)
    patched = build_patched()
    c = dict(decisions=0, roll6_decisions=0, roll6_bonus_move_decisions=0, bonus_decisions=0,
             bonus_after_six=0, streak2_decisions=0, d2_disagree=0, d2_checked=0,
             d1_disagree_bonus_root=0, d2_disagree_bonus_root=0, bonus_root_checked=0)
    gaps, valgaps = [], []
    cur_game_log = []
    t0 = time.time()

    def factory(game, seat, roll_box):
        tr = TurnContextTracker()
        state = {"last_roll_six": False}

        def choose(legal):
            if not legal:
                tr.record_move(game, None)
                return None
            roll, pb, k = tr.context_for(roll_box)
            c["decisions"] += 1
            if pb is None:
                state["last_roll_six"] = (roll == 6)
            if roll == 6:
                c["roll6_decisions"] += 1
            if k == 2 and pb is None:
                c["streak2_decisions"] += 1
            if pb is not None:
                c["bonus_decisions"] += 1
                if state["last_roll_six"]:
                    c["bonus_after_six"] += 1
                    # root bonus decision: shipped (roll=None) vs patched (roll=6 carried)
                    if len(legal) > 1:
                        use(orig)
                        m1, mv1, _ = search.search(game, None, pb, k, depth=2, evaluator=ev)
                        use(patched)
                        m2, mv2, _ = search.search(game, 6, pb, k, depth=2, evaluator=ev)
                        use(orig)
                        c["bonus_root_checked"] += 1
                        c["d2_disagree_bonus_root"] += int(m1[0].piece_id != m2[0].piece_id)
            if roll == 6 and pb is None:
                bonus_moves = [m for m in legal if game.would_capture(m) or m[1] == Board.FINAL_POSITION]
                if bonus_moves:
                    c["roll6_bonus_move_decisions"] += 1
                    if len(legal) > 1:
                        use(orig)
                        m1, mv1, _ = search.search(game, roll, None, k, depth=2, evaluator=ev)
                        use(patched)
                        m2, mv2, _ = search.search(game, roll, None, k, depth=2, evaluator=ev)
                        use(orig)
                        c["d2_checked"] += 1
                        c["d2_disagree"] += int(m1[0].piece_id != m2[0].piece_id)
                        me = game.current_player_idx
                        for bm in bonus_moves:
                            pid = bm[0].piece_id
                            valgaps.append(float(mv2[pid][me] - mv1[pid][me]))
            move, _, rv = search.search(game, roll, pb, k, depth=1, evaluator=ev)
            cur_game_log.append((game.turn_number, game.current_player_idx, float(rv[game.current_player_idx])))
            tr.record_move(game, move)
            return move
        return choose

    phase_rows = []  # (turn, value for mover, mover won)
    lengths = []
    for g in range(a.games):
        cur_game_log.clear()
        w = play_one_game({0: factory, 1: factory}, seed=a.seed0 + g)
        lengths.append(max(t for t, _, _ in cur_game_log))
        for t, seat, v in cur_game_log:
            phase_rows.append((t, v, 1.0 if w == seat else 0.0))
    # Outcome predictability by game phase: Brier of the depth-1 root value
    # (mover's win probability) against the realised result, by fraction of game elapsed.
    arr = np.array(phase_rows)
    L = np.median(lengths)
    c["median_game_turns"] = float(L)
    edges = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, np.inf]  # fraction of the median game length
    phase = []
    tfrac = arr[:, 0] / L
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tfrac >= lo) & (tfrac < hi)
        if m.sum() == 0:
            continue
        v, z = arr[m, 1], arr[m, 2]
        brier = float(np.mean((v - z) ** 2))
        base = float(np.mean((z.mean() - z) ** 2))
        phase.append(dict(turn_frac=f"[{lo},{hi})", n=int(m.sum()), brier=round(brier, 4),
                          base_brier=round(base, 4), resolved_share=round(1 - brier / base, 3),
                          mean_abs_v_minus_half=round(float(np.mean(np.abs(v - 0.5))), 3)))
    c["phase"] = phase
    c["games"] = a.games
    c["seeds"] = f"{a.seed0}+g, g<{a.games}"
    if valgaps:
        c["capture_or_finish_on_6_value_shift_pp_mean"] = round(100 * float(np.mean(valgaps)), 2)
        c["capture_or_finish_on_6_value_shift_pp_median"] = round(100 * float(np.median(valgaps)), 2)
        c["n_value_shift"] = len(valgaps)
    c["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(c, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(c, f, indent=2)


if __name__ == "__main__":
    main()
