"""
PUCT-guided Monte Carlo Tree Search over parchis.game.Game.

Scope (see the plan's "Search scope" decision): tree nodes are the
searching agent's own TURN-STARTING decisions only (a fresh, non-bonus
roll). Six-again rerolls and bonus-chain continuations within that same
turn, and every opponent's whole turn, are resolved via a fixed policy
while simulating between two tree nodes -- not separately searched. This
keeps the tree tractable and reuses Game.play_turn()'s already-correct
six-again/three-sixes/bonus-chain mechanics verbatim instead of
reimplementing any of it.

Value semantics: because only the agent's own decisions are ever tree
nodes (never an opponent's), there is no adversarial min/max framing here
-- value is always "expected outcome for the searching agent" (+1 win,
-1 loss, 0 draw/cutoff), and backpropagation is a plain running average
along the visited path, no perspective flipping.

How a turn-starting decision is captured without reimplementing
Game.play_turn(): the searching agent's Player.choose_move is temporarily
overridden to raise _AgentDecisionPoint(legal_moves) the moment
play_turn() calls it for that turn's first (non-bonus) roll -- before any
move has been applied and before next_player() has run, so aborting via
exception at that exact point leaves the simulated Game in a well-defined
state. (game.turn_number is incremented one step early by the aborted
play_turn() call -- a harmless bookkeeping artifact; nothing downstream
reads it for legality or scoring.)
"""

import copy
import math

DEFAULT_C_PUCT = 1.4
DEFAULT_MAX_SIMULATED_TURNS = 300  # safety cap, mirrors evaluate_model's max_steps_per_episode
# Tree nodes are the agent's own successive TURNS (see module docstring) --
# with only 2-4 actions per node, a handful of simulations is enough to
# reach real depth fast. Left unbounded, a modest simulation budget (tens)
# ends up spending most of its budget many turns downstream of the root
# instead of resolving the actual decision at hand (confirmed empirically:
# an unbounded tree let a clearly-worse move end up with MORE visits than
# a clearly-better one, because by depth 8-9 "who's ahead" had long since
# stopped reflecting the root's own choice). Beyond this depth, a node is
# treated as a leaf -- evaluate_fn's value is used as-is, not refined
# further -- exactly the standard "bounded lookahead + leaf evaluation"
# shape a value network is meant to enable.
DEFAULT_MAX_DEPTH = 3


class _AgentDecisionPoint(Exception):
    """Raised by the searching agent's overridden choose_move to hand a
    fresh turn-starting decision's legal_moves back to _play_forward,
    interrupting the in-progress Game.play_turn() call at exactly that
    point (no move applied yet)."""
    def __init__(self, legal_moves):
        super().__init__()
        self.legal_moves = legal_moves


def _install_pause_on_first_call(player):
    def pause(legal_moves):
        raise _AgentDecisionPoint(legal_moves)
    player.choose_move = pause


def _install_fixed_policy(game, agent_seat, opponent_choose_move):
    """Every non-agent seat's choose_move is fixed for the whole simulated
    future -- these are never tree nodes (see module docstring).

    Compares by object identity against game.players[agent_seat], NOT
    player.player_id -- Game.__init__ rotates self.players so the
    dice-determined starting player becomes list index 0, but
    Player.player_id is assigned before that rotation and never updated,
    so it can differ from its own object's list index. agent_seat is
    always a list index (matching ParchisEnv.agent_player_idx's own
    convention -- see parchis/rl/env.py), so identity/index comparisons
    against game.players[agent_seat] are the only correct check here."""
    agent_player = game.players[agent_seat]
    for player in game.players:
        if player is not agent_player:
            player.choose_move = (lambda legal_moves, p=player: opponent_choose_move(p, legal_moves))


def _install_roll_recorder(game):
    """Wrap game.dice.roll to record the most recent roll value, without
    changing its behavior. Game.play_turn() never passes the roll itself
    to choose_move() (only the resulting legal_moves) -- this recovers it
    for observation purposes (the dice-onehot feature) without touching
    Dice."""
    box = {"last_roll": None}
    original_roll = game.dice.roll

    def recording_roll():
        value = original_roll()
        box["last_roll"] = value
        return value

    game.dice.roll = recording_roll
    return box


def default_random_opponent_policy(player, legal_moves):
    """Fixed policy for opponent turns: the plain default Player.choose_move
    (uniform random). Calls the CLASS method explicitly (Player.choose_move,
    not player.choose_move) -- by the time this runs, `player.choose_move`
    has already been overridden (see _install_fixed_policy) to route
    through this very function, so going through the instance attribute
    here would recurse forever."""
    from parchis.game.player import Player
    return Player.choose_move(player, legal_moves)


def _legal_action_ids(legal_moves):
    return [piece.piece_id for piece, _new_pos, _move_type in legal_moves]


def _play_forward(game, agent_seat, roll_box, max_turns):
    """Advance `game` (mutated in place) turn by turn via Game.play_turn()
    until the agent's next fresh decision, the game ends, or `max_turns`
    is exceeded.

    Returns:
        ('terminal', agent_won)  -- agent_won is True/False, or None if the
            game ended (e.g. all-pieces-finished never happened) without a
            winner (shouldn't normally occur, defensive)
        ('cutoff', None)
        ('decision', (legal_moves, dice_roll))
    """
    for _ in range(max_turns):
        if game.game_over:
            # Identity check, not player_id -- see _install_fixed_policy's
            # docstring for why player_id can't be trusted as a seat index.
            return "terminal", (game.winner is game.players[agent_seat] if game.winner else None)
        if game.current_player_idx == agent_seat:
            _install_pause_on_first_call(game.get_current_player())
        try:
            game.play_turn()
        except _AgentDecisionPoint as e:
            # Remove the one-shot instance-attribute override immediately --
            # this game (and this player object) may be stored on a node
            # and deepcopied again later; a stale override surviving that
            # deepcopy (functions aren't deep-copied, so it would still
            # close over THIS player object, not the copy's) would raise
            # spuriously or worse. Falls back to the class's default method.
            del game.get_current_player().choose_move
            return "decision", (e.legal_moves, roll_box["last_roll"])
    if game.game_over:
        return "terminal", (game.winner is game.players[agent_seat] if game.winner else None)
    return "cutoff", None


class MCTSNode:
    """A tree node = the searching agent facing a specific turn-starting
    decision. Root and every expanded node stores the live `game` (and the
    dice roll that produced its legal_moves) so a later simulation
    revisiting it can deepcopy from exactly that point -- meaning the roll
    that produced a given node's legal actions is sampled once, the first
    time that node is created, and stays fixed on repeat visits (an
    "open-loop" simplification -- see the plan's "Chance nodes" note;
    revisit if this proves too coarse)."""

    __slots__ = ("prior", "N", "W", "Q", "children", "game", "legal_moves",
                 "dice_roll", "expanded", "terminal")

    def __init__(self, prior):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}      # action(piece_id int) -> MCTSNode
        self.game = None
        self.legal_moves = None
        self.dice_roll = None
        self.expanded = False
        self.terminal = False

    def action_ids(self):
        return list(self.children.keys())


def _puct_score(parent_N, child, c_puct):
    exploration = c_puct * child.prior * math.sqrt(parent_N) / (1 + child.N)
    return child.Q + exploration


def _select_action(node, c_puct):
    return max(node.children.items(), key=lambda kv: _puct_score(node.N, kv[1], c_puct))[0]


def make_root(game, agent_seat, legal_moves, dice_roll, priors):
    """Build the root node for a real (not simulated) decision the
    searching agent is actually facing right now -- `game` is the real,
    live Game object; a deepcopy is taken per-simulation, never mutating
    it. `priors`: dict {piece_id: prior_probability} over `legal_moves`'
    action ids."""
    root = MCTSNode(prior=1.0)
    root.game = game
    root.legal_moves = legal_moves
    root.dice_roll = dice_roll
    root.expanded = True
    for action in _legal_action_ids(legal_moves):
        root.children[action] = MCTSNode(prior=priors[action])
    return root


def _expand(parent, action, agent_seat, evaluate_fn, opponent_choose_move, max_turns,
            depth, max_depth):
    """Realize `parent.children[action]` (currently a bare, unexpanded
    MCTSNode) by deepcopying parent.game, applying `action`, and resolving
    forward to the next agent decision (or a terminal/cutoff). Mutates the
    child node in place. Returns the leaf value to backpropagate.

    `depth` is this child's depth (root's children are depth 1); once it
    reaches `max_depth`, the resulting decision node is left childless --
    evaluate_fn's value is used as a bounded-lookahead leaf estimate rather
    than expanding (and simulating) further. See DEFAULT_MAX_DEPTH."""
    child = parent.children[action]

    game_copy = copy.deepcopy(parent.game)
    agent_player = game_copy.players[agent_seat]
    # For the ROOT's own children specifically, parent.game IS the real,
    # live Game -- and its agent player's choose_move may itself be
    # overridden to route through mcts.search() (see parchis/search/agents.py,
    # used by the arena). deepcopy carries that instance-attribute override
    # into game_copy too (functions aren't deep-copied, so it'd still be
    # the exact same closure), and this module's whole design assumes
    # within-turn continuations (bonus chains) fall through to the DEFAULT
    # random policy -- without this reset, a bonus chain here would call
    # back into mcts.search() on an already-simulated copy, recursing
    # without bound (confirmed: this raised RecursionError before the fix).
    if "choose_move" in vars(agent_player):
        del agent_player.choose_move
    _, new_position, move_type = next(m for m in parent.legal_moves if m[0].piece_id == action)
    # `parent.legal_moves`' piece reference is from parent.game (a DIFFERENT
    # object graph than game_copy after deepcopy) -- new_position/move_type
    # are plain values so reusing them from that tuple is fine, but the
    # piece to actually move must be resolved against game_copy's own
    # players, or execute_move would mutate parent.game's real piece object
    # in place (and if parent is the root, that IS the live real game).
    piece = next(p for p in agent_player.pieces if p.piece_id == action)

    _install_fixed_policy(game_copy, agent_seat, opponent_choose_move)
    roll_box = _install_roll_recorder(game_copy)

    # Apply the root/parent-level chosen move directly (this is the ONE
    # decision this expansion is realizing) -- mirrors Game.play_turn()'s
    # own move-execution + bonus-chain handling exactly, since after this
    # point we hand control to Game.play_turn()'s bonus-chain logic by
    # calling it, not by re-deciding this move a second time.
    move_info = game_copy.execute_move(piece, new_position, move_type)

    from parchis.game.records import TurnInfo
    turn_info = TurnInfo(turn_number=game_copy.turn_number, player=agent_player)
    game_copy.handle_bonus_moves(agent_player, move_info, turn_info)

    if agent_player.has_won():
        game_copy.game_over = True
        game_copy.winner = agent_player
        value = 1.0
        child.expanded = True
        child.terminal = True
        parent.children[action] = child
        return value

    # This turn's bonus chain (if any) is done; six-again streak handling
    # and next_player() are exactly what Game.play_turn() would do next
    # for a turn that already had its move executed -- but play_turn()
    # always starts a turn from its own dice.roll(), so instead of
    # re-deriving the six-again state machine here (real duplication risk
    # of a subtly wrong copy), advance the turn boundary with next_player()
    # and let _play_forward's next iteration re-enter through
    # Game.play_turn() normally for whoever's turn is next -- six-again
    # continuation for THIS move (if the roll that led to `move` was a 6)
    # is handled by the fixed policy the same as any other continuation,
    # consistent with this module's documented scope (only the turn's
    # FIRST decision is tree-searched).
    game_copy.next_player()

    status, payload = _play_forward(game_copy, agent_seat, roll_box, max_turns)

    if status == "terminal":
        agent_won = payload
        value = 1.0 if agent_won else (-1.0 if agent_won is not None else 0.0)
        child.expanded = True
        child.terminal = True
    elif status == "cutoff":
        value = 0.0
        child.expanded = True
        child.terminal = True
    else:
        legal_moves, dice_roll = payload
        priors, value = evaluate_fn(game_copy, agent_seat, legal_moves, dice_roll)
        child.game = game_copy
        child.legal_moves = legal_moves
        child.dice_roll = dice_roll
        if depth < max_depth:
            for a in _legal_action_ids(legal_moves):
                child.children[a] = MCTSNode(prior=priors[a])
        # else: depth limit reached -- leave child.children empty, making
        # it a leaf (see docstring); child.Q will just be this evaluate_fn
        # value, never refined by deeper simulation.
        child.expanded = True

    return value


def run_simulations(root, agent_seat, n_simulations, evaluate_fn,
                     opponent_choose_move, c_puct=DEFAULT_C_PUCT,
                     max_turns=DEFAULT_MAX_SIMULATED_TURNS,
                     max_depth=DEFAULT_MAX_DEPTH, rng_seed=None):
    """Run `n_simulations` MCTS simulations from `root` (already built via
    make_root), mutating the tree in place. Each simulation is wrapped in
    isolated_random (a fresh, deterministic seed per simulation, saved/
    restored around it) so the many hypothetical dice rolls/random
    opponent moves it plays out never perturb the real game's own future
    Dice.roll() sequence -- see isolated_random.py. Pass `rng_seed` for a
    fully reproducible search (same seed -> same simulations); omitted,
    each simulation is still isolated from the real game but not
    reproducible run-to-run (falls back to object identity)."""
    from parchis.search.isolated_random import isolated_random

    base_seed = rng_seed if rng_seed is not None else id(root)
    for sim in range(n_simulations):
        with isolated_random(seed=(base_seed, sim)):
            path = [root]
            node = root
            value = None
            depth = 0
            while True:
                if node.terminal or not node.children:
                    # Terminal, or a bounded-depth/no-legal-move leaf:
                    # node.Q already equals its known/settled value (set on
                    # first backprop below, unchanged thereafter since it
                    # has no children left to refine it further).
                    value = node.Q
                    break
                action = _select_action(node, c_puct)
                child = node.children[action]
                depth += 1
                if not child.expanded:
                    value = _expand(node, action, agent_seat, evaluate_fn,
                                     opponent_choose_move, max_turns, depth, max_depth)
                    path.append(child)
                    break
                path.append(child)
                node = child

            for n in path:
                n.N += 1
                n.W += value
                n.Q = n.W / n.N


def visit_counts(root):
    """{action: visit_count} for the root's children -- the search's
    actual output (AlphaZero trains on this distribution, not on Q)."""
    return {action: child.N for action, child in root.children.items()}


def best_action(root):
    counts = visit_counts(root)
    return max(counts.items(), key=lambda kv: kv[1])[0]


def search(game, agent_seat, legal_moves, dice_roll, n_simulations, evaluate_fn,
           opponent_choose_move=default_random_opponent_policy,
           c_puct=DEFAULT_C_PUCT, max_turns=DEFAULT_MAX_SIMULATED_TURNS,
           max_depth=DEFAULT_MAX_DEPTH, rng_seed=None):
    """End-to-end: build the root for this real decision, run the search,
    return (chosen_move_tuple, root) -- `root` is returned too so callers
    can inspect visit_counts()/Q for debugging or for recording AlphaZero
    training targets later (Phase C).

    `game` is the REAL, live Game object -- never mutated (every
    simulation works on its own deepcopy)."""
    if not legal_moves:
        return None, None
    priors, _root_value = evaluate_fn(game, agent_seat, legal_moves, dice_roll)
    root = make_root(game, agent_seat, legal_moves, dice_roll, priors)
    run_simulations(root, agent_seat, n_simulations, evaluate_fn,
                     opponent_choose_move, c_puct=c_puct, max_turns=max_turns,
                     max_depth=max_depth, rng_seed=rng_seed)
    action = best_action(root)
    move = next(m for m in legal_moves if m[0].piece_id == action)
    return move, root
