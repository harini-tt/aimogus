"""
Result reporting: CSV export and terminal summary table.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Any

from evals.models import EvalResults


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_summary(results: EvalResults) -> None:
    """Print a human-readable summary table to stdout."""

    print(f"\n{'='*90}")
    print(f"  EVAL SUMMARY — {len(results.game_records)} games played")
    print(f"{'='*90}")

    # Collect all models from all sources
    all_models: set[str] = set()
    if results.win_rates:
        all_models.update(results.win_rates.keys())
    if results.elo_ratings:
        all_models.update(results.elo_ratings.deception.keys())
    for tqa in results.truthfulqa_scores:
        all_models.add(tqa.model)

    if not all_models:
        print("\n  No results to display.\n")
        return

    # Build rows
    rows: list[dict[str, Any]] = []
    for model in sorted(all_models):
        row: dict[str, Any] = {"model": model}

        # TruthfulQA
        tqa_match = next((t for t in results.truthfulqa_scores if t.model == model), None)
        row["truthfulqa"] = f"{tqa_match.accuracy:.1%}" if tqa_match else "—"

        # Deception Elo
        if results.elo_ratings and model in results.elo_ratings.deception:
            row["deception_elo"] = f"{results.elo_ratings.deception[model]:.0f}"
        else:
            row["deception_elo"] = "—"

        # Detection Elo
        if results.elo_ratings and model in results.elo_ratings.detection:
            row["detection_elo"] = f"{results.elo_ratings.detection[model]:.0f}"
        else:
            row["detection_elo"] = "—"

        # Win rates
        wr = results.win_rates.get(model, {})
        row["imp_wr"] = f"{wr['impostor_win_rate']:.0%}" if "impostor_win_rate" in wr else "—"
        row["crew_wr"] = f"{wr['crewmate_win_rate']:.0%}" if "crewmate_win_rate" in wr else "—"
        row["imp_games"] = str(wr.get("impostor_games", 0))
        row["crew_games"] = str(wr.get("crewmate_games", 0))

        rows.append(row)

    # Print table
    header = (
        f"{'Model':<45} {'TruthfulQA':>10} {'Dec. Elo':>9} {'Det. Elo':>9} "
        f"{'Imp WR':>7} {'Crew WR':>8} {'Imp #':>6} {'Crew #':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<45} {row['truthfulqa']:>10} {row['deception_elo']:>9} "
            f"{row['detection_elo']:>9} {row['imp_wr']:>7} {row['crew_wr']:>8} "
            f"{row['imp_games']:>6} {row['crew_games']:>7}"
        )

    # Confidence intervals
    if results.elo_confidence_intervals:
        print(f"\n{'Elo 95% Confidence Intervals':}")
        print(f"{'Model':<45} {'Deception Elo':<30} {'Detection Elo':<30}")
        print("-" * 105)
        for model in sorted(results.elo_confidence_intervals.keys()):
            ci = results.elo_confidence_intervals[model]
            dec_str = f"{ci['deception_mean']:.0f} [{ci['deception_ci_low']:.0f}, {ci['deception_ci_high']:.0f}]"
            det_str = f"{ci['detection_mean']:.0f} [{ci['detection_ci_low']:.0f}, {ci['detection_ci_high']:.0f}]"
            print(f"{model:<45} {dec_str:<30} {det_str:<30}")

    # Game outcome distribution
    if results.game_records:
        imp_wins = sum(1 for r in results.game_records if r.winner_side == "impostor")
        crew_wins = sum(1 for r in results.game_records if r.winner_side == "crewmate")
        print(f"\n  Game outcomes: Impostor wins {imp_wins} / Crewmate wins {crew_wins}")

    print(f"\n{'='*90}\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(results: EvalResults, output_dir: str = "evals/results") -> dict[str, str]:
    """
    Export all results to CSV and JSON files.

    Returns ``{name: filepath}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # --- Summary table (one row per model) ---
    summary_path = os.path.join(output_dir, "summary.csv")
    all_models: set[str] = set()
    if results.win_rates:
        all_models.update(results.win_rates.keys())
    if results.elo_ratings:
        all_models.update(results.elo_ratings.deception.keys())
    for tqa in results.truthfulqa_scores:
        all_models.add(tqa.model)

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "truthfulqa_accuracy", "deception_elo", "detection_elo",
            "impostor_win_rate", "crewmate_win_rate", "impostor_games", "crewmate_games",
        ])
        for model in sorted(all_models):
            tqa = next((t for t in results.truthfulqa_scores if t.model == model), None)
            elo_dec = results.elo_ratings.deception.get(model) if results.elo_ratings else None
            elo_det = results.elo_ratings.detection.get(model) if results.elo_ratings else None
            wr = results.win_rates.get(model, {})

            writer.writerow([
                model,
                f"{tqa.accuracy:.4f}" if tqa else "",
                f"{elo_dec:.1f}" if elo_dec is not None else "",
                f"{elo_det:.1f}" if elo_det is not None else "",
                wr.get("impostor_win_rate", ""),
                wr.get("crewmate_win_rate", ""),
                wr.get("impostor_games", ""),
                wr.get("crewmate_games", ""),
            ])
    paths["summary"] = summary_path

    # --- Win rates ---
    if results.win_rates:
        wr_path = os.path.join(output_dir, "win_rates.csv")
        with open(wr_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "impostor_win_rate", "crewmate_win_rate",
                "overall_win_rate", "impostor_games", "crewmate_games", "total_games",
            ])
            for model, wr in sorted(results.win_rates.items()):
                writer.writerow([
                    model, wr["impostor_win_rate"], wr["crewmate_win_rate"],
                    wr["overall_win_rate"], wr["impostor_games"],
                    wr["crewmate_games"], wr["total_games"],
                ])
        paths["win_rates"] = wr_path

    # --- Game records as JSON ---
    if results.game_records:
        records_path = os.path.join(output_dir, "game_records.json")
        with open(records_path, "w") as f:
            json.dump(
                [r.to_dict() for r in results.game_records],
                f, indent=2, default=str,
            )
        paths["game_records"] = records_path

    # --- TruthfulQA details ---
    if results.truthfulqa_scores:
        tqa_path = os.path.join(output_dir, "truthfulqa.csv")
        with open(tqa_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "provider", "accuracy", "correct", "num_questions"])
            for tqa in results.truthfulqa_scores:
                writer.writerow([tqa.model, tqa.provider, tqa.accuracy, tqa.correct, tqa.num_questions])
        paths["truthfulqa"] = tqa_path

    return paths
