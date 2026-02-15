"""
Lean evaluation framework for Among Us LLM agent simulation.

Provides three core metrics per model:
  - TruthfulQA score (standalone benchmark, independent of gameplay)
  - Deception Elo (from game performance as impostor)
  - Win Rate (impostor and crewmate)
"""
