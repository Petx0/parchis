"""
Value-network + full-expectimax stack (docs/AGENT_REBUILD_PLAN.md Part 2):
the TD-Gammon-shaped replacement for the retired PPO+MCTS architecture.

Purely additive, like parchis/search/ before it: nothing here is imported
by parchis/rl/, parchis/training/, or parchis/search/, and nothing in those
packages is modified beyond the two engine-level fixes Phase 0 already
made (Game.snapshot()/restore(), ParchisEnv._get_observation(perspective_seat=)).
Operates directly on parchis.game.Game -- never parchis.rl.env.ParchisEnv,
no puppeting, no Gym-API step() loop -- since search needs direct
forward-simulation control an env instance doesn't offer.
"""
