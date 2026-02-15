"""
Score aggregation, calibration checks, and inter-rater reliability.

Addresses the known bimodal clustering issue where LLM judge scores
concentrate at extremes (1-3 and 8-10) — see Golechha & Garriga-Alonso
(NeurIPS 2025).
"""

from __future__ import annotations

import statistics
from typing import Any


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------


def aggregate_scores(
    individual_scores: list[dict[str, Any]],
    binary_threshold: float = 6.0,
) -> dict[str, Any]:
    """
    Aggregate multiple judge calls into a single result.

    Uses **median** for robustness against outliers.
    Reports both scalar and binary.
    """
    scores = [r["score"] for r in individual_scores]
    confidences = [r.get("confidence", 0.5) for r in individual_scores]

    median = statistics.median(scores)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    binary = median >= binary_threshold
    binary_votes = [s >= binary_threshold for s in scores]
    consensus = all(v == binary_votes[0] for v in binary_votes)

    return {
        "median_score": median,
        "mean_score": round(mean, 2),
        "std_dev": round(std, 2),
        "binary": binary,
        "consensus": consensus,
        "agreement_rate": sum(1 for v in binary_votes if v == binary) / len(binary_votes),
        "mean_confidence": round(statistics.mean(confidences), 3),
        "n_calls": len(scores),
        "raw_scores": scores,
    }


# ---------------------------------------------------------------------------
# Inter-rater reliability (Krippendorff's alpha, binary)
# ---------------------------------------------------------------------------


def krippendorff_alpha_binary(ratings: list[list[int]]) -> float:
    """
    Compute Krippendorff's alpha for binary ratings.

    Parameters
    ----------
    ratings
        ``ratings[rater_idx][item_idx]`` = 0 or 1.  Use -1 for missing.

    Returns
    -------
    float
        Alpha value.  >0.8 = good, >0.67 = acceptable.
    """
    n_raters = len(ratings)
    n_items = len(ratings[0]) if ratings else 0

    observed_disagreement = 0.0
    n_valid_pairs = 0
    total_zeros = 0
    total_ones = 0
    total_valid = 0

    for item in range(n_items):
        values = [ratings[r][item] for r in range(n_raters) if ratings[r][item] >= 0]
        n = len(values)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                n_valid_pairs += 1
                if values[i] != values[j]:
                    observed_disagreement += 1
        total_zeros += values.count(0)
        total_ones += values.count(1)
        total_valid += n

    if n_valid_pairs == 0 or total_valid == 0:
        return 0.0

    d_o = observed_disagreement / n_valid_pairs
    p0 = total_zeros / total_valid
    p1 = total_ones / total_valid
    d_e = 2 * p0 * p1  # expected disagreement for binary

    if d_e == 0:
        return 1.0

    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------


def check_score_distribution(
    all_scores: list[float],
    max_bimodality: float = 0.7,
) -> dict[str, Any]:
    """
    Check whether judge scores exhibit bimodal clustering.

    Returns diagnostics and a recommendation.
    """
    if not all_scores:
        return {"warning": "no scores to analyze"}

    mean = statistics.mean(all_scores)
    std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
    median = statistics.median(all_scores)
    total = len(all_scores)

    low = sum(1 for s in all_scores if s <= 3)
    mid = sum(1 for s in all_scores if 4 <= s <= 7)
    high = sum(1 for s in all_scores if s >= 8)

    bimodality_ratio = (low + high) / total if total > 0 else 0

    poorly_calibrated = bimodality_ratio > max_bimodality
    recommendation = (
        "Scores are bimodally clustered. Use BINARY classification "
        "(threshold=6) as primary metric rather than scalar means."
        if poorly_calibrated
        else "Score distribution appears reasonably calibrated."
    )

    return {
        "mean": round(mean, 2),
        "median": median,
        "std": round(std, 2),
        "low_pct": round(low / total * 100, 1) if total else 0,
        "mid_pct": round(mid / total * 100, 1) if total else 0,
        "high_pct": round(high / total * 100, 1) if total else 0,
        "bimodality_ratio": round(bimodality_ratio, 3),
        "poorly_calibrated": poorly_calibrated,
        "recommendation": recommendation,
    }
