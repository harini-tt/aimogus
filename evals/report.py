"""
Result aggregation, export (CSV/JSON), and summary printing.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from evals.models import GameRecord, MetricResult


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregatedMetric:
    metric_name: str
    model: str
    identity: str
    mean: float
    std: float
    count: int


def aggregate_by_model(
    all_metrics: list[MetricResult],
) -> list[AggregatedMetric]:
    """Group metrics by (name, model, identity) and compute mean/std."""
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for m in all_metrics:
        buckets[(m.metric_name, m.model, m.player_identity)].append(m.value)

    aggregated: list[AggregatedMetric] = []
    for (name, model, identity), values in sorted(buckets.items()):
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
        std = math.sqrt(variance)
        aggregated.append(
            AggregatedMetric(name, model, identity, round(mean, 4), round(std, 4), n)
        )
    return aggregated


# ---------------------------------------------------------------------------
# Win rates
# ---------------------------------------------------------------------------


def win_rates_by_model(records: list[GameRecord]) -> dict[str, dict[str, Any]]:
    """
    Compute per-model win rates as impostor and crewmate.

    Returns ``{model: {impostor_win_rate, crewmate_win_rate, total_games}}``.
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"imp_wins": 0, "imp_total": 0, "crew_wins": 0, "crew_total": 0}
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

    return {
        model: {
            "impostor_win_rate": round(s["imp_wins"] / max(s["imp_total"], 1), 3),
            "crewmate_win_rate": round(s["crew_wins"] / max(s["crew_total"], 1), 3),
            "total_games": s["imp_total"] + s["crew_total"],
        }
        for model, s in stats.items()
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(
    records: list[GameRecord],
    all_metrics: list[MetricResult],
    output_dir: str = "evals/results",
) -> dict[str, str]:
    """Export all metrics to CSV files. Returns ``{name: filepath}``."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # Raw metrics
    raw_path = os.path.join(output_dir, "metrics_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "metric_name", "player_name", "player_identity", "model", "value",
        ])
        for m in all_metrics:
            writer.writerow([m.metric_name, m.player_name, m.player_identity, m.model, m.value])
    paths["metrics_raw"] = raw_path

    # Aggregated
    agg = aggregate_by_model(all_metrics)
    agg_path = os.path.join(output_dir, "metrics_aggregated.csv")
    with open(agg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "model", "identity", "mean", "std", "count"])
        for a in agg:
            writer.writerow([a.metric_name, a.model, a.identity, a.mean, a.std, a.count])
    paths["metrics_aggregated"] = agg_path

    # Win rates
    wr = win_rates_by_model(records)
    wr_path = os.path.join(output_dir, "win_rates.csv")
    with open(wr_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "impostor_win_rate", "crewmate_win_rate", "total_games"])
        for model, rates in wr.items():
            writer.writerow([model, rates["impostor_win_rate"], rates["crewmate_win_rate"], rates["total_games"]])
    paths["win_rates"] = wr_path

    # Game records as JSON
    records_path = os.path.join(output_dir, "game_records.json")
    from dataclasses import asdict
    with open(records_path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2, default=str)
    paths["game_records"] = records_path

    return paths


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------


def print_summary(
    records: list[GameRecord],
    all_metrics: list[MetricResult],
) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*80}")
    print(f"EVAL SUMMARY — {len(records)} games")
    print(f"{'='*80}")

    # Win distribution
    imp_wins = sum(1 for r in records if r.winner_side == "impostor")
    crew_wins = sum(1 for r in records if r.winner_side == "crewmate")
    print(f"\nWin distribution: Impostor {imp_wins} / Crewmate {crew_wins}")

    # Win rates by model
    wr = win_rates_by_model(records)
    print(f"\n{'Model':<40} {'Imp WR':>8} {'Crew WR':>8} {'Games':>6}")
    print("-" * 66)
    for model, rates in sorted(wr.items()):
        print(
            f"{model:<40} {rates['impostor_win_rate']:>7.1%} "
            f"{rates['crewmate_win_rate']:>7.1%} {rates['total_games']:>6}"
        )

    # Aggregated metrics
    agg = aggregate_by_model(all_metrics)
    if agg:
        print(f"\n{'Metric':<40} {'Model':<25} {'Role':<10} {'Mean':>7} {'Std':>7} {'N':>4}")
        print("-" * 98)
        for a in agg:
            print(f"{a.metric_name:<40} {a.model:<25} {a.identity:<10} {a.mean:>7.3f} {a.std:>7.3f} {a.count:>4}")

    print(f"\n{'='*80}\n")
