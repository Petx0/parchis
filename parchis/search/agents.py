"""
Player.choose_move-compatible adapters for the arena (parchis/evaluation/arena.py):
wrap a trained MaskablePPO model, with or without MCTS search on top, or
plain randomness, as a factory(game, seat, roll_box) -> choose_move_fn.

Why a factory rather than a ready-made choose_move_fn: Player.choose_move(legal_moves)
never receives the live `game` object or the roll that produced legal_moves
(see mcts.py's module docstring for the same constraint) -- each agent
needs to close over the SPECIFIC game instance it's playing in this real
match, and over a shared `roll_box` (arena.py installs one roll-recorder
per game, mirroring mcts.py's own _install_roll_recorder) to recover the
actual dice roll for observation construction, which choose_move's own
signature can't provide.
"""

from parchis.search import mcts
from parchis.search.state_view import ObservationAdapter
from parchis.search.network_eval import make_network_evaluate_fn


def make_plain_ppo_agent_factory(model, num_players, deterministic=True):
    """No search -- model.predict() directly, exactly like every other
    trained-checkpoint usage this session (train_ppo.py, evaluate.py, ...).
    Returns factory(game, seat, roll_box) -> choose_move_fn."""
    def factory(game, seat, roll_box):
        adapter = ObservationAdapter(num_players=num_players)

        def choose_move(legal_moves):
            if not legal_moves:
                return None
            obs = adapter.observation(game, seat, current_dice_roll=roll_box["last_roll"],
                                       pending_bonus=None, consecutive_sixes=0)
            mask = adapter.action_mask(legal_moves)
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action = int(action)
            for piece, new_pos, move_type in legal_moves:
                if piece.piece_id == action:
                    return (piece, new_pos, move_type)
            return legal_moves[0]  # shouldn't happen given the mask; defensive fallback

        return choose_move

    return factory


def make_mcts_ppo_agent_factory(model, num_players, n_simulations=50,
                                 c_puct=mcts.DEFAULT_C_PUCT,
                                 max_depth=mcts.DEFAULT_MAX_DEPTH):
    """MCTS search using `model`'s outputs as priors/leaf-value (Phase B --
    the existing, already-trained checkpoint, no retraining). Returns
    factory(game, seat, roll_box) -> choose_move_fn."""
    def factory(game, seat, roll_box):
        evaluate_fn = make_network_evaluate_fn(model, num_players=num_players)
        call_count = {"n": 0}

        def choose_move(legal_moves):
            if not legal_moves:
                return None
            call_count["n"] += 1
            move, _root = mcts.search(
                game, agent_seat=seat, legal_moves=legal_moves,
                dice_roll=roll_box["last_roll"], n_simulations=n_simulations,
                evaluate_fn=evaluate_fn, c_puct=c_puct, max_depth=max_depth,
                rng_seed=(id(game), seat, call_count["n"]),
            )
            return move

        return choose_move

    return factory
