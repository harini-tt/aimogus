"""
Batch game runner and eval orchestrator.

Runs N games headless, extracts structured data, computes all
requested metrics, and optionally runs LLM-judge scoring and
post-game interviews.  **No game engine modifications.**
"""

from __future__ import annotations

import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.game import AmongUs

from evals.configs import EvalConfig
from evals.extractor import extract_game_record, parse_private_reasoning
from evals.ground_truth import build_context_block, build_game_summary, build_ground_truth_facts
from evals.interview import interview_all_agents
from evals.judge import judge_step, judge_with_reliability
from evals.models import (
    DimensionScore,
    EloRatings,
    GameRecord,
    JudgeMode,
    MetricResult,
)
from evals.metrics.capabilities import compute_all_deterministic
from evals.metrics.elo import bootstrap_elo_ci, process_all_games
from evals.metrics.sycophancy import compute_ppd_scores, compute_vote_conformity
from evals.metrics.deceptive_alignment import (
    compute_observer_effect,
    compute_strategic_patience,
    compute_trust_exploitation_curve,
)
from evals.metrics.means_to_ends import compute_fabrication_rates, compute_vulnerability_targeting
from evals.metrics.deception_taxonomy import classify_utterances, compute_deception_profiles
from evals.prompts import INTERVIEW_JUDGE_PROMPT, INTERVIEW_DIMENSION_DESCRIPTIONS, JUDGE_SYSTEM_PROMPT
from evals.sampler import KeyMoments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-game pipeline
# ---------------------------------------------------------------------------


def _resolve_model_configs(
    config: EvalConfig,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Determine per-player model configs for one game."""
    num_players = config.game_config["num_players"]
    if config.model_configs:
        return config.model_configs[:num_players]
    if config.model_pool:
        return [rng.choice(config.model_pool) for _ in range(num_players)]
    return [{"provider": "openai", "model": "gpt-4o"}] * num_players


def run_single_game(
    config: EvalConfig,
    model_configs: list[dict[str, str]],
    game_id: str,
) -> dict[str, Any]:
    """
    Run one game and compute all requested metrics.

    Returns a dict with ``game_record``, ``metrics``, plus optional
    ``alignment_results``, ``taxonomy_results``, ``interview_results``.
    """
    log_dir = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"game-{game_id}.log")

    # Set up per-game logging
    game_logger = logging.getLogger("envs.game")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    game_logger.addHandler(fh)

    result: dict[str, Any] = {}

    try:
        # Run the game
        game = AmongUs(
            game_config=config.game_config,
            UI=None,
            model_configs=model_configs,
        )
        winner_code = game.run_game()

        # Extract structured data
        rec = extract_game_record(
            game, winner_code, model_configs, config.game_config_name,
        )
        rec.game_id = game_id
        result["game_record"] = rec

        # Deterministic metrics
        metrics: list[MetricResult] = []
        if config.run_deterministic:
            metrics.extend(compute_all_deterministic(rec))

        # Per-step LLM judge
        if config.run_judge:
            judge_mode = JudgeMode(config.judge_mode)
            sampler = KeyMoments()
            for adapter in game.agents:
                agent = adapter.agent
                interactions = agent.context.interactions
                indices = sampler.select(interactions, agent.role.value, config.judge_max_steps)
                for idx in indices:
                    interaction = interactions[idx]
                    output_text = str(interaction.model_dump()) if hasattr(interaction, "model_dump") else str(interaction)
                    ctx = build_context_block(
                        judge_mode, game, rec, agent.name,
                        getattr(interaction, "round", 0),
                        output_text, log_path,
                    )
                    for dim in config.judge_dimensions:
                        score = judge_step(ctx, dim, agent.role.value, config.judge_model, config.judge_n_calls)
                        metrics.append(MetricResult(
                            metric_name=f"judge_{dim}",
                            player_name=agent.name,
                            player_identity=adapter.player.identity,
                            model=rec.player_summaries.get(agent.name)
                            and rec.player_summaries[agent.name].model or "unknown",
                            value=score.median_score,
                            metadata={"binary": score.binary, "std": score.std_dev},
                        ))

        # Post-game interviews
        if config.run_interviews:
            interview_data = interview_all_agents(
                game, dimensions=config.interview_dimensions,
            )
            # Judge each interview response
            game_summary = build_game_summary(rec)
            for agent_name, exchanges in interview_data.items():
                ps = rec.player_summaries.get(agent_name)
                for ex in exchanges:
                    gt_facts = build_ground_truth_facts(rec, agent_name)
                    dim_desc = INTERVIEW_DIMENSION_DESCRIPTIONS.get(ex.dimension, "")
                    outcome = f"{rec.winner_side} win" if ps else "unknown"
                    prompt = INTERVIEW_JUDGE_PROMPT.format(
                        dimension_name=ex.dimension.upper(),
                        dimension_description=dim_desc,
                        game_summary=game_summary,
                        agent_name=agent_name,
                        agent_role=ps.identity if ps else "?",
                        outcome=outcome,
                        question=ex.question,
                        response=ex.response,
                        ground_truth_facts=gt_facts,
                    )
                    score = judge_with_reliability(prompt, model=config.judge_model, n_calls=config.judge_n_calls)
                    metrics.append(MetricResult(
                        metric_name=f"interview_{ex.dimension}",
                        player_name=agent_name,
                        player_identity=ps.identity if ps else "?",
                        model=ps.model if ps else "?",
                        value=score.median_score,
                        metadata={"binary": score.binary, "std": score.std_dev},
                    ))
            result["interview_data"] = interview_data

        # Alignment metrics
        if config.run_alignment:
            private_reasoning = parse_private_reasoning(log_path)
            result["alignment"] = {
                "sycophancy_ppd": compute_ppd_scores(game, rec, private_reasoning, config.judge_model),
                "vote_conformity": compute_vote_conformity(rec),
                "strategic_patience": compute_strategic_patience(rec),
                "trust_exploitation": compute_trust_exploitation_curve(rec),
                "observer_effect": compute_observer_effect(rec),
                "fabrication": compute_fabrication_rates(game, rec, config.judge_model),
                "vulnerability_targeting": compute_vulnerability_targeting(rec),
            }

        # Deception taxonomy
        if config.run_taxonomy:
            private_reasoning = parse_private_reasoning(log_path)
            classified = classify_utterances(game, rec, private_reasoning, config.judge_model)
            profiles = compute_deception_profiles(classified)
            result["taxonomy"] = {
                "classified_utterances": classified,
                "deception_profiles": profiles,
            }

        result["metrics"] = metrics
        return result

    finally:
        game_logger.removeHandler(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_eval(config: EvalConfig) -> dict[str, Any]:
    """
    Run a full eval: N games, all requested metrics.

    Returns a dict with ``records``, ``all_metrics``, ``elo_ratings``,
    ``elo_ci``, and per-game results in ``game_results``.
    """
    rng = random.Random(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    records: list[GameRecord] = []
    all_metrics: list[MetricResult] = []
    game_results: list[dict[str, Any]] = []

    for i in range(config.num_games):
        game_id = f"{config.name}-{i:03d}"
        model_configs = _resolve_model_configs(config, rng)
        print(f"\n[eval] Game {i + 1}/{config.num_games} ({game_id})")
        for j, mc in enumerate(model_configs):
            print(f"  Player {j + 1}: {mc['provider']}/{mc['model']}")

        result = run_single_game(config, model_configs, game_id)
        rec = result["game_record"]

        # Save game record
        rec.save(Path(config.output_dir))
        records.append(rec)
        all_metrics.extend(result.get("metrics", []))
        game_results.append(result)

        print(f"  Winner: {rec.winner_side} (code {rec.winner_code}), "
              f"T={rec.total_timesteps}, tasks={rec.task_completion:.0%}")

    # Elo computation
    elo_ratings: EloRatings | None = None
    elo_ci: dict[str, Any] | None = None
    if config.run_elo and len(records) >= 2:
        print("\n[eval] Computing Elo ratings...")
        elo_ratings = process_all_games(records)
        if len(records) >= 10:
            print("[eval] Bootstrap confidence intervals...")
            elo_ci = bootstrap_elo_ci(records, n_bootstrap=min(1000, len(records) * 10))

    return {
        "records": records,
        "all_metrics": all_metrics,
        "game_results": game_results,
        "elo_ratings": elo_ratings,
        "elo_ci": elo_ci,
    }
