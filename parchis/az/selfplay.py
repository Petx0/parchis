"""
Game generation + training-target construction: plays games from a pool of
agents, recording every decision as a training example.

Two generation modes, added in two different phases:
  - generate_games (docs/AGENT_REBUILD_PLAN.md Part 3 item 11, Phase 2):
    a FIXED pool {tuned heuristic, ε-noisy heuristic, random}, recording
    (encoding, chosen move, final outcome) for a one-time supervised
    value/policy bootstrap.
  - generate_round_games (Part 3 Phase 3): the CURRENT CHAMPION net,
    self-play against a pool of {champion, last 4 promoted, tuned
    heuristic, random} (parchis.az.champion_pool), with root-exploration
    (temperature anneal + Dirichlet noise, parchis.az.targets) and search-
    derived (root_value, policy_target) targets -- see its own docstring.

Both reuse parchis/evaluation/arena.py's existing
factory(game, seat, roll_box) convention and Game.play_turn() (via
arena.play_one_game) rather than a custom loop -- the SAME already-tested
six-again/bonus/three-sixes mechanics every other evaluation/generation
path in this codebase relies on. Each recorded seat's factory is wrapped
with a recording layer that uses parchis.az.turn_context.TurnContextTracker
(the same bonus-vs-fresh-roll detection parchis/az/agent.py uses) to build
the exact (roll, pending_bonus, consecutive_sixes) context
parchis.az.encoding.encode / parchis.az.search.search need, BEFORE
resolving that decision's actual move.
"""

import random

import numpy as np

from parchis.agents import heuristic
from parchis.az import champion_pool, encoding, rollouts, search, targets
from parchis.az.agent import NetEvaluator
from parchis.az.turn_context import TurnContextTracker
from parchis.evaluation import arena


def random_factory(game, seat, roll_box):
    """arena-style factory playing uniformly random legal moves. Public
    (not `_`-prefixed): reused as-is by parchis.az.champion_pool's Phase 3
    pool, not just default_pool_factories below."""
    player = game.players[seat]
    return lambda legal_moves: player.__class__.choose_move(player, legal_moves)


def default_pool_factories(noisy_seed=None):
    """The pool docs/AGENT_REBUILD_PLAN.md Part 3 item 11 calls for:
    {tuned heuristic, ε-noisy heuristic, random}. A fresh tuple each call
    (the noisy member owns a private RNG -- see
    parchis.agents.heuristic.make_epsilon_noisy_heuristic_agent_factory)."""
    return (
        heuristic.make_heuristic_agent_factory(heuristic.TUNED_WEIGHTS),
        heuristic.make_epsilon_noisy_heuristic_agent_factory(
            heuristic.TUNED_WEIGHTS, seed=noisy_seed,
        ),
        random_factory,
    )


def _make_recording_factory(base_factory, examples):
    """Wraps an arena-style factory so every decision it makes is appended
    to the shared `examples` list as {'encoding', 'chosen_piece_id',
    'mover_seat'} -- 'outcome' is filled in by the caller once the whole
    game concludes (a single game's true outcome isn't known until then)."""

    def factory(game, seat, roll_box):
        base_choose_move = base_factory(game, seat, roll_box)
        tracker = TurnContextTracker()

        def choose_move(legal_moves):
            if not legal_moves:
                tracker.record_move(game, None)
                return base_choose_move(legal_moves)

            roll, pending_bonus, consecutive_sixes = tracker.context_for(roll_box)
            obs = encoding.encode(game, seat, roll=roll, pending_bonus=pending_bonus,
                                   consecutive_sixes=consecutive_sixes)
            move = base_choose_move(legal_moves)
            tracker.record_move(game, move)
            if move is not None:
                examples.append({
                    'encoding': obs,
                    'chosen_piece_id': move[0].piece_id,
                    'mover_seat': seat,
                })
            return move

        return choose_move

    return factory


def generate_games(pool_factories, n_games, num_players=2, max_turns=arena.DEFAULT_MAX_TURNS,
                    seed=None):
    """
    Play `n_games`, sampling one pool member per SEAT per game, uniformly
    and independently (mirrors parchis.rl.opponent_pool's own "one member
    per seat per episode" convention -- different seats can hold different
    pool members within the same game).

    Records every decision as a training example, backfilling 'outcome'
    once each game concludes: a win/draw vector expressed in THAT
    DECISION'S OWN mover-relative seat order -- index 0 is always "did
    THIS decision's own mover go on to win" (1.0, 0.0, or 1/num_players
    each for a truncated game -- Part 4's "never silently 0"), index k is
    the seat k turns after the mover in play order, exactly
    parchis.az.encoding's own `_ordered_seats(mover_seat, ...)` convention.
    This must match the encoding's own channel order: the net is trained
    to map "an encoding centered on whoever is deciding" to "that same
    decider's own win probability first", and a target expressed in
    absolute seat order instead would train it on a meaningless,
    inconsistent mapping (confirmed the hard way -- see
    docs/AZ_DESIGN.md's Phase 2 entry). Each example also carries
    'game_index' (0-based, in generation order) -- decisions from the same
    game are highly correlated (they share, or nearly share, the same
    outcome), so any train/validation/test split must be done at the GAME
    level, never by shuffling individual decisions.

    Returns:
        tuple(list[dict], dict): (examples, stats).
        examples: [{'encoding': np.ndarray, 'chosen_piece_id': int,
                    'mover_seat': int, 'game_index': int,
                    'outcome': np.ndarray[num_players]}, ...]  (outcome in
                    mover-relative order, per decision -- see above)
        stats: {'n_games', 'n_decisions', 'n_truncated',
                'n_by_winner_seat': {seat: count}} ('n_by_winner_seat' is
                the one place seats stay ABSOLUTE -- it's a fairness/sanity
                stat about the game, not a per-decision training target).
    """
    rng = random.Random(seed)
    examples = []
    recording_pool = [_make_recording_factory(f, examples) for f in pool_factories]
    stats = {
        'n_games': 0, 'n_decisions': 0, 'n_truncated': 0,
        'n_by_winner_seat': {seat: 0 for seat in range(num_players)},
    }

    for game_index in range(n_games):
        mark = len(examples)
        agent_factories = {seat: rng.choice(recording_pool) for seat in range(num_players)}
        game_seed = rng.randrange(2**31)
        winner_seat = arena.play_one_game(
            agent_factories, num_players=num_players, max_turns=max_turns, seed=game_seed,
        )

        if winner_seat is not None:
            absolute_outcome = np.zeros(num_players, dtype=np.float32)
            absolute_outcome[winner_seat] = 1.0
            stats['n_by_winner_seat'][winner_seat] += 1
        else:
            absolute_outcome = np.full(num_players, 1.0 / num_players, dtype=np.float32)
            stats['n_truncated'] += 1

        for example in examples[mark:]:
            # Remap into THIS example's own mover-relative order (each
            # decision in a multi-seat game can have a different mover, so
            # this can't be hoisted out of the loop) -- inverse of
            # parchis.az.agent.NetEvaluator's own relative->absolute
            # np.roll(probs, observer_seat), matching encode()'s
            # _ordered_seats(observer_seat, N) = [(observer_seat+k) % N ...].
            example['outcome'] = np.roll(absolute_outcome, -example['mover_seat'])
            example['game_index'] = game_index

        stats['n_games'] += 1
        stats['n_decisions'] += len(examples) - mark

    return examples, stats


def examples_to_arrays(examples, num_players):
    """Stack a list of per-decision example dicts (generate_games' output)
    into dense arrays for training: (X, policy_targets, value_targets).
    X: (n, encoding_size) float32. policy_targets: (n,) int64 piece_id
    (0-3). value_targets: (n, num_players) float32."""
    n = len(examples)
    input_size = encoding.encoding_size(num_players)
    X = np.empty((n, input_size), dtype=np.float32)
    policy_targets = np.empty(n, dtype=np.int64)
    value_targets = np.empty((n, num_players), dtype=np.float32)
    for i, ex in enumerate(examples):
        X[i] = ex['encoding']
        policy_targets[i] = ex['chosen_piece_id']
        value_targets[i] = ex['outcome']
    return X, policy_targets, value_targets


# --- Phase 3: self-play round generation (docs/AGENT_REBUILD_PLAN.md Part 3 Phase 3) ---


def _make_search_recording_factory(numpy_net, examples, game_index_box, depth, ply_box,
                                    dirichlet_rng, tau_start, tau_end, anneal_plies,
                                    tau_target, dirichlet_alpha, dirichlet_epsilon,
                                    rollout_target_fraction=0.0, rollout_n=24,
                                    rollout_rng=None, max_turns=arena.DEFAULT_MAX_TURNS):
    """One seat's factory for generate_round_games, when that seat is
    occupied by a search-capable pool member (the champion or a promoted
    net) this game: every real decision runs search.search() directly
    (NOT parchis.az.agent.make_search_agent_factory's opaque greedy
    wrapper) so the move actually PLAYED can be sampled from an
    exploration distribution while still recording the search's own
    un-perturbed (root_value, move_values) for target construction.

    `ply_box`/`game_index_box` are single-key dicts SHARED across every
    seat's factory within one game (mutable cells, not closures over a
    fresh int -- every seat's decision must advance the SAME ply clock,
    since "the first ~15 plies" means the game's own ply count, not this
    seat's own decision count).

    `rollout_target_fraction`/`rollout_n`/`rollout_rng` (Phase 2.2,
    parchis.az.rollouts): when > 0, a `rollout_target_fraction` random
    subset of this factory's OWN recorded decisions also get an
    independent rollout-based value estimate (stored as
    'rollout_value', else None) -- see rollouts.py's own module
    docstring for why this is deliberately sampled rather than applied to
    every decision. `rollout_target_fraction=0.0` (the default) never
    touches `rollout_rng`, preserving byte-identical behavior to before
    this parameter existed for any caller that doesn't pass it."""
    evaluator = NetEvaluator(numpy_net)

    def factory(game, seat, roll_box):
        tracker = TurnContextTracker()

        def choose_move(legal_moves):
            ply = ply_box['ply']
            ply_box['ply'] += 1
            if not legal_moves:
                tracker.record_move(game, None)
                return None

            roll, pending_bonus, consecutive_sixes = tracker.context_for(roll_box)
            obs = encoding.encode(game, seat, roll=roll, pending_bonus=pending_bonus,
                                   consecutive_sixes=consecutive_sixes)
            _greedy_move, move_values, root_value_absolute = search.search(
                game, roll=roll, pending_bonus=pending_bonus,
                consecutive_sixes=consecutive_sixes, depth=depth, evaluator=evaluator,
            )

            tau = targets.anneal_temperature(ply, tau_start=tau_start, tau_end=tau_end,
                                              anneal_plies=anneal_plies)
            probs, legal_piece_ids = targets.dirichlet_mixed_probs(
                move_values, seat, tau, dirichlet_rng,
                alpha=dirichlet_alpha, epsilon=dirichlet_epsilon,
            )
            chosen_piece_id = targets.sample_move_piece_id(probs, legal_piece_ids, dirichlet_rng)
            move = next(m for m in legal_moves if m[0].piece_id == chosen_piece_id)

            # move_values/root_value_absolute are in ABSOLUTE seat order
            # (search.py's own contract); remap to mover-relative order
            # HERE, at recording time, matching encode()'s own convention
            # -- exactly generate_games' hard-won fix (see module
            # docstring), applied from the start this time rather than
            # discovered as a bug later.
            rollout_value = None
            if rollout_target_fraction > 0 and rollout_rng.random() < rollout_target_fraction:
                # search.search() never mutates `game` (its own guarantee),
                # so it's still safe to snapshot AFTER calling it, at the
                # same pre-move state `obs` was itself encoded from.
                snap = game.snapshot()
                rollout_value = rollouts.estimate_rollout_value(
                    game, snap, mover_seat=seat, n_rollouts=rollout_n,
                    rng=rollout_rng, max_turns=max_turns,
                )

            examples.append({
                'encoding': obs,
                'root_value': np.roll(root_value_absolute, -seat),
                'rollout_value': rollout_value,
                'policy_target': targets.policy_target_from_move_values(
                    move_values, seat, tau_target=tau_target,
                ),
                'chosen_piece_id': chosen_piece_id,
                'mover_seat': seat,
                'game_index': game_index_box['index'],
            })
            tracker.record_move(game, move)
            return move

        return choose_move

    return factory


def _make_ply_counting_factory(base_factory, ply_box):
    """Wraps a non-recordable anchor factory (tuned heuristic, random) so
    its decisions still advance the shared per-game ply clock -- see
    _make_search_recording_factory's docstring for why that clock is
    shared across every seat, not just recorded ones."""

    def factory(game, seat, roll_box):
        base_choose_move = base_factory(game, seat, roll_box)

        def choose_move(legal_moves):
            ply_box['ply'] += 1
            return base_choose_move(legal_moves)

        return choose_move

    return factory


def generate_round_games(champion_numpy_net, promoted_numpy_nets, n_games, num_players=2,
                          max_turns=arena.DEFAULT_MAX_TURNS, depth=1, seed=None,
                          lam=targets.DEFAULT_LAMBDA, tau_target=targets.DEFAULT_TAU_TARGET,
                          tau_start=targets.DEFAULT_TAU_START, tau_end=targets.DEFAULT_TAU_END,
                          anneal_plies=targets.DEFAULT_ANNEAL_PLIES,
                          dirichlet_alpha=targets.DEFAULT_DIRICHLET_ALPHA,
                          dirichlet_epsilon=targets.DEFAULT_DIRICHLET_EPSILON,
                          recent_numpy_nets=(), tuned_weights=None,
                          rollout_target_fraction=0.0, rollout_n=24):
    """
    Phase 3 self-play generation (Part 3 Phase 3's "Generate" bullet): the
    CURRENT CHAMPION (`champion_numpy_net`) always occupies one seat per
    game (chosen uniformly at random, mirroring arena.play_match's own
    "randomize which seat" fairness convention -- guarantees every game
    yields at least one recorded seat). Every OTHER seat samples
    independently from parchis.az.champion_pool.build_pool's pool
    (champion again, `promoted_numpy_nets`, `recent_numpy_nets`, tuned
    heuristic, random).

    Every decision made by a search-capable seat (the guaranteed champion
    seat, or a sampled opponent seat that also turned out to be net-backed)
    is recorded: the encoding, the move actually PLAYED (sampled from
    softmax(move values / annealed tau) mixed with Dirichlet root noise --
    exploration, not search.search()'s own greedy argmax), and the
    search's own root_value + per-move breakdown at that decision,
    immediately turned into (root_value, policy_target) in mover-relative
    order (parchis.az.targets). A heuristic/random seat's decisions are
    never recorded (no root_value to build a target from -- see
    parchis.az.champion_pool's module docstring) but still consume the
    shared per-game ply clock the temperature anneal reads from.

    `value_target` is filled in once each game concludes (mirrors
    generate_games' own backfill pattern): blend_value_target(outcome,
    root_value, lam), with `outcome` computed in the SAME mover-relative
    order as generate_games' own (np.roll(absolute_outcome,
    -mover_seat)).

    `rollout_target_fraction`/`rollout_n` (Phase 2.2, parchis.az.rollouts):
    forwarded to _make_search_recording_factory -- see its own docstring.
    0.0 (default) never spends any rollout compute.

    Returns:
        tuple(list[dict], dict): (examples, stats).
        examples: [{'encoding', 'root_value' (mover-relative),
            'rollout_value' (mover-relative, or None -- only set for the
            rollout_target_fraction of decisions actually sampled),
            'policy_target' (length-4 float32, from the UN-perturbed move
            values -- NOT what selected chosen_piece_id below),
            'chosen_piece_id' (diagnostic: the exploration-sampled move
            actually played), 'mover_seat', 'game_index', 'value_target'
            (mover-relative, filled in on backfill, from rollout_value
            when set else root_value), 'aux_target' (Phase 4.1, filled in
            on backfill: length-4 float32, 0.0/1.0 per own piece_id --
            did that piece finish by game end, from the SAME game's own
            arena.play_one_game(return_piece_status=True) call, free --
            no extra generation cost, unlike rollout_value)}, ...].
        stats: {'n_games', 'n_recorded_decisions', 'n_total_plies'
            (every real choose_move call across every seat, recorded or
            not -- what the ply clock actually counted), 'n_truncated',
            'n_unrecorded_games' (games where even the guaranteed champion
            seat recorded nothing -- expected to be ~0 in real games;
            measured, not assumed, since a tiny max_turns test fixture can
            end before the champion seat's first turn), 'n_by_winner_seat'}.
    """
    rng = random.Random(seed)
    dirichlet_rng = np.random.default_rng(seed)
    # Independent stream from `rng` (same seed value, different Random
    # instance -- matching dirichlet_rng's own precedent just above): only
    # ever consumed when rollout_target_fraction > 0 (short-circuit
    # evaluation in the recording factory), so existing callers that don't
    # pass rollout_target_fraction see byte-identical output to before
    # this parameter existed.
    rollout_rng = random.Random(seed)
    examples = []
    nets, anchor_factories = champion_pool.build_pool(
        champion_numpy_net, promoted_numpy_nets, recent_numpy_nets=recent_numpy_nets,
        tuned_weights=tuned_weights,
    )
    # Every seat NOT holding the guaranteed champion samples uniformly from
    # this combined pool -- ('net', i) again lets champion-vs-champion (or
    # champion-vs-promoted) self-play arise, not just champion-vs-anchor.
    pool_entries = [('net', i) for i in range(len(nets))] + \
                   [('anchor', i) for i in range(len(anchor_factories))]

    stats = {
        'n_games': 0, 'n_recorded_decisions': 0, 'n_total_plies': 0, 'n_truncated': 0,
        'n_unrecorded_games': 0,
        'n_by_winner_seat': {seat: 0 for seat in range(num_players)},
    }

    for game_index in range(n_games):
        mark = len(examples)
        ply_box = {'ply': 0}
        game_index_box = {'index': game_index}

        champion_seat = rng.randrange(num_players)
        seat_kinds = {champion_seat: ('net', 0)}
        for seat in range(num_players):
            if seat != champion_seat:
                seat_kinds[seat] = rng.choice(pool_entries)

        agent_factories = {}
        for seat, (kind, idx) in seat_kinds.items():
            if kind == 'net':
                agent_factories[seat] = _make_search_recording_factory(
                    nets[idx], examples, game_index_box, depth, ply_box, dirichlet_rng,
                    tau_start, tau_end, anneal_plies, tau_target, dirichlet_alpha, dirichlet_epsilon,
                    rollout_target_fraction=rollout_target_fraction, rollout_n=rollout_n,
                    rollout_rng=rollout_rng, max_turns=max_turns,
                )
            else:
                agent_factories[seat] = _make_ply_counting_factory(anchor_factories[idx], ply_box)

        game_seed = rng.randrange(2**31)
        winner_seat, piece_status = arena.play_one_game(
            agent_factories, num_players=num_players, max_turns=max_turns, seed=game_seed,
            return_piece_status=True,
        )

        if winner_seat is not None:
            absolute_outcome = np.zeros(num_players, dtype=np.float32)
            absolute_outcome[winner_seat] = 1.0
            stats['n_by_winner_seat'][winner_seat] += 1
        else:
            absolute_outcome = np.full(num_players, 1.0 / num_players, dtype=np.float32)
            stats['n_truncated'] += 1

        n_new = len(examples) - mark
        for example in examples[mark:]:
            outcome_relative = np.roll(absolute_outcome, -example['mover_seat'])
            # Use the rollout-refined estimate as the bootstrap term when
            # this decision was sampled for one (Phase 2.2); otherwise
            # fall back to root_value, exactly as before this existed.
            bootstrap_value = example['rollout_value']
            if bootstrap_value is None:
                bootstrap_value = example['root_value']
            example['value_target'] = targets.blend_value_target(
                outcome_relative, bootstrap_value, lam=lam,
            ).astype(np.float32)
            # Phase 4.1 aux target: did THIS decision's own mover's own 4
            # pieces (piece_id-indexed, matching encoding._own_piece_features'
            # convention -- never seat-rotated, since this is about "my own
            # pieces" regardless of which seat I am) finish by game end?
            # Free from this same game -- no extra generation cost.
            example['aux_target'] = np.array(
                piece_status[example['mover_seat']], dtype=np.float32,
            )

        stats['n_games'] += 1
        stats['n_recorded_decisions'] += n_new
        stats['n_total_plies'] += ply_box['ply']
        if n_new == 0:
            stats['n_unrecorded_games'] += 1

    return examples, stats


def round_examples_to_arrays(examples, num_players):
    """Stack generate_round_games' output into dense arrays for training:
    (X, policy_targets, value_targets, aux_targets). Unlike
    examples_to_arrays (Phase 2), policy_targets here is a SOFT (n, 4)
    float32 distribution (§2.3's z_policy), not a hard (n,) int64 class
    index -- parchis.az.train's _forward_losses already handles either
    shape correctly, since torch.nn.functional.cross_entropy dispatches
    on the target's own dtype/shape, so no training-code change is needed,
    only this different packing. aux_targets (Phase 4.1) is (n, 4)
    float32, one 0.0/1.0 per own piece_id -- see generate_round_games'
    own docstring for how it's computed."""
    n = len(examples)
    input_size = encoding.encoding_size(num_players)
    X = np.empty((n, input_size), dtype=np.float32)
    policy_targets = np.empty((n, 4), dtype=np.float32)
    value_targets = np.empty((n, num_players), dtype=np.float32)
    aux_targets = np.empty((n, 4), dtype=np.float32)
    for i, ex in enumerate(examples):
        X[i] = ex['encoding']
        policy_targets[i] = ex['policy_target']
        value_targets[i] = ex['value_target']
        aux_targets[i] = ex['aux_target']
    return X, policy_targets, value_targets, aux_targets
