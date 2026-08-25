"""
Search-augmented decision-making for Parchís (Phase A/B of the AlphaZero-style
plan -- see ~/.claude/plans/please-read-the-documents-dynamic-simon.md).

Purely additive: nothing here is imported by parchis/rl/, parchis/training/,
or parchis/evaluation/, and nothing in those packages is modified. This
package operates directly on parchis.game.{Game,Board,Player} -- the
deterministic engine -- not on the Gym-API ParchisEnv, since MCTS needs
direct forward-simulation control that env.step() doesn't offer.
"""
