"""
Hand-built (non-learned) agents for Parchís.

parchis/agents/heuristic.py is the absolute strength anchor the AlphaZero-
style rebuild plan calls for (docs/AGENT_REBUILD_PLAN.md §2.4): a linear
feature-weighted move scorer, tuned by CEM, used as the bootstrap opponent
and the pool member that prevents single-lineage collapse during self-play.
"""
