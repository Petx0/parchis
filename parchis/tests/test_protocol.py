"""Unit tests for the champion-promotion rule (parchis/evaluation/protocol.py, docs/EVAL_PROTOCOL.md)."""
import pytest

from parchis.evaluation import protocol as P


def _res(rate, n, draws=0):
    return P.summarise(round(rate * n), draws, n)


def _results(decisive=0.52, heur=0.73, champ_heur=0.73, rand=0.97, prev=0.53, draws=0):
    return {"decisive": _res(decisive, 20000, draws), "heuristic": _res(heur, 2000),
            "champion_heuristic": _res(champ_heur, 2000), "random": _res(rand, 2000),
            "previous": _res(prev, 2000)}


def test_block_seeds_are_disjoint_and_below_training_base():
    a = P.block_seeds(0, 0)
    b = P.block_seeds(0, 100)
    assert a == list(range(5_000_000, 5_000_008)) and not set(a) & set(b)
    assert P.block_seeds(P.MAX_BLOCKS - 1, 900)[-1] < 10_000_000
    with pytest.raises(ValueError):
        P.block_seeds(P.MAX_BLOCKS, 0)
    with pytest.raises(ValueError):
        P.block_seeds(0, 995)


def test_match_offsets_fit_and_only_heuristic_pair_shares_seeds():
    offs = {name: off for name, off, _ in P.MATCHES}
    assert offs["heuristic"] == offs["champion_heuristic"]
    others = [o for n, o in offs.items() if n != "champion_heuristic"]
    assert len(set(others)) == len(others)


def test_next_block_never_reuses_and_merges_registries():
    assert P.next_block([], []) == 0
    home = [{"kind": "protocol", "block": 3}]
    repo = [{"kind": "protocol", "block": 5}, {"kind": "legacy_eval"}]
    assert P.next_block(home, repo) == 6


def test_pass_needs_lower_bound_above_half():
    assert P.verdict(_results(decisive=0.52))["verdict"] == "PASS"
    # 50.6% over 20,000 games: lower bound ~49.9% -> FAIL
    assert P.verdict(_results(decisive=0.506))["verdict"] == "FAIL"
    assert P.verdict(_results(decisive=0.50))["verdict"] == "FAIL"


def test_guards():
    assert P.verdict(_results(heur=0.69, champ_heur=0.73))["verdict"] == "FAIL"
    assert P.verdict(_results(heur=0.71, champ_heur=0.73))["verdict"] == "PASS"
    assert P.verdict(_results(rand=0.94))["verdict"] == "FAIL"
    # previous champion is context only
    assert P.verdict(_results(prev=0.40))["verdict"] == "PASS"


def test_best_of_n_needs_confirmation():
    assert P.verdict(_results(), selection="best-of-4")["verdict"] == "PROVISIONAL"
    assert P.verdict(_results(), selection="best-of-4", confirmation_of="x.json")["verdict"] == "PASS"
    assert P.verdict(_results(decisive=0.50), selection="best-of-4")["verdict"] == "FAIL"


def test_draws_count_half_and_too_many_invalidate():
    s = P.summarise(10, 2, 20)
    assert s["win_rate"] == pytest.approx(0.55)
    assert P.verdict(_results(draws=100))["verdict"] == "INVALID"
