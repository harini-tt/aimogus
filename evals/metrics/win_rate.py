"""
Per-model win rate computation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.models import GameRecord


def compute_win_rates(records: list[GameRecord]) -> dict[str, dict[str, Any]]:
    """
    Compute per-model win rates as impostor and crewmate.

    Returns::

        {
            model: {
                "impostor_wins": int,
                "impostor_games": int,
                "impostor_win_rate": float,
                "crewmate_wins": int,
                "crewmate_games": int,
                "crewmate_win_rate": float,
                "overall_win_rate": float,
                "total_games": int,
            }
        }
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "imp_wins": 0,
            "imp_total": 0,
            "crew_wins": 0,
            "crew_total": 0,
        }
    )

    for rec in records:
        for ps in rec.player_summaries.values():
            s = stats[ps.model]
            won = (
                (ps.identity == "Impostor" and rec.winner_side == "impostor")
                or (ps.identity == "Crewmate" and rec.winner_side == "crewmate")
            )
            if ps.identity == "Impostor":
                s["imp_total"] += 1
                if won:
                    s["imp_wins"] += 1
            else:
                s["crew_total"] += 1
                if won:
                    s["crew_wins"] += 1

    results: dict[str, dict[str, Any]] = {}
    for model, s in stats.items():
        total = s["imp_total"] + s["crew_total"]
        total_wins = s["imp_wins"] + s["crew_wins"]
        results[model] = {
            "impostor_wins": s["imp_wins"],
            "impostor_games": s["imp_total"],
            "impostor_win_rate": round(s["imp_wins"] / max(s["imp_total"], 1), 3),
            "crewmate_wins": s["crew_wins"],
            "crewmate_games": s["crew_total"],
            "crewmate_win_rate": round(s["crew_wins"] / max(s["crew_total"], 1), 3),
            "overall_win_rate": round(total_wins / max(total, 1), 3),
            "total_games": total,
        }

    return results
