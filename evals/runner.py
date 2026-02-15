"""
Game orchestrator for the eval framework.

Runs N games headless (in parallel when max_parallel > 1) with random
model assignment from the configured model pool, then computes Elo
ratings and win rates.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.game import AmongUs

from evals.config import EvalConfig, ModelSpec
from evals.extractor import extract_game_record
from evals.models import EvalResults, GameRecord, TruthfulQAResult
from evals.metrics.elo import bootstrap_elo_ci, process_all_games
from evals.metrics.win_rate import compute_win_rates
from evals.truthfulqa import run_truthfulqa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model assignment
# ---------------------------------------------------------------------------


def _assign_models_for_game(
    config: EvalConfig,
    rng: random.Random,
) -> list[dict[str, str]]:
    """
    Randomly assign models from the config's model pool to player slots
    for one game.  Each slot gets a random draw (with replacement) from
    the pool, so all models appear across roles over many games.
    """
    num_players = config.game_config["num_players"]
    pool = config.model_dicts()

    if not pool:
        return [{"provider": "openai", "model": "gpt-4o"}] * num_players

    return [rng.choice(pool) for _ in range(num_players)]


# ---------------------------------------------------------------------------
# Single game (thread-safe)
# ---------------------------------------------------------------------------


def _run_single_game(
    game_config: dict,
    model_configs: list[dict[str, str]],
    game_id: str,
    output_dir: str,
) -> GameRecord:
    """
    Run one Among Us game headless and return a GameRecord.

    Thread-safe: each game instance gets its own dedicated logger that
    writes to a separate file, so parallel games never interleave logs.
    """
    # Set up a per-game logger with its own file handler
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"game-{game_id}.log")

    game_logger = logging.getLogger(f"envs.game.{game_id}")
    game_logger.setLevel(logging.INFO)
    game_logger.propagate = False
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    game_logger.addHandler(fh)

    try:
        game = AmongUs(
            game_config=game_config,
            UI=None,
            model_configs=model_configs,
            game_logger=game_logger,
        )
        winner_code = game.run_game()
        record = extract_game_record(game, winner_code, model_configs)
        record.game_id = game_id
        return record
    finally:
        game_logger.removeHandler(fh)
        fh.close()


# ---------------------------------------------------------------------------
# Batch game runner
# ---------------------------------------------------------------------------


def run_games(config: EvalConfig) -> list[GameRecord]:
    """
    Run all games specified by the config and return GameRecords.

    Games run in parallel when ``config.game_settings.max_parallel > 1``.
    """
    rng = random.Random(config.seed)
    num_games = config.game_settings.num_games
    max_parallel = max(config.game_settings.max_parallel, 1)

    # Pre-generate all game assignments so the RNG is deterministic
    # regardless of execution order.
    jobs: list[tuple[str, list[dict[str, str]]]] = []
    for _ in range(num_games):
        game_id = str(uuid.uuid4())[:8]
        model_configs = _assign_models_for_game(config, rng)
        jobs.append((game_id, model_configs))

    # Print plan
    for i, (game_id, model_configs) in enumerate(jobs):
        model_names = [f"{c['provider']}/{c['model']}" for c in model_configs]
        print(f"  Game {i + 1}/{num_games} (id={game_id}): {', '.join(set(model_names))}")

    records: list[GameRecord] = []
    game_config = config.game_config

    if max_parallel <= 1:
        # Sequential execution
        for i, (game_id, model_configs) in enumerate(jobs):
            print(f"\n--- Game {i + 1}/{num_games} (id={game_id}) ---")
            try:
                record = _run_single_game(
                    game_config, model_configs, game_id, config.output_dir,
                )
                records.append(record)
                print(f"  Result: {record.winner_side} wins (code={record.winner_code})")
            except Exception as exc:
                logger.error("Game %s failed: %s", game_id, exc, exc_info=True)
                print(f"  Game {game_id} FAILED: {exc}")
    else:
        # Parallel execution using threads (not processes — OpenAI SDK
        # exceptions are not picklable, and the work is I/O-bound anyway).
        print(f"\n  Launching {num_games} games with max {max_parallel} in parallel...\n")
        future_to_info: dict[Any, tuple[int, str]] = {}

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            for i, (game_id, model_configs) in enumerate(jobs):
                future = executor.submit(
                    _run_single_game,
                    game_config, model_configs, game_id, config.output_dir,
                )
                future_to_info[future] = (i, game_id)

            for future in as_completed(future_to_info):
                idx, game_id = future_to_info[future]
                try:
                    record = future.result()
                    records.append(record)
                    print(
                        f"  [{len(records)}/{num_games}] Game {idx + 1} "
                        f"(id={game_id}): {record.winner_side} wins "
                        f"(T={record.total_timesteps}, tasks={record.task_completion:.0%})"
                    )
                except Exception as exc:
                    logger.error("Game %s failed: %s", game_id, exc, exc_info=True)
                    print(f"  [{len(records)}/{num_games}] Game {idx + 1} (id={game_id}) FAILED: {exc}")

    return records


# ---------------------------------------------------------------------------
# TruthfulQA eval (parallelized across models)
# ---------------------------------------------------------------------------


def run_truthfulqa_eval(config: EvalConfig) -> list[TruthfulQAResult]:
    """
    Run TruthfulQA MC1 against all unique models in the config, in parallel.

    Returns a list of TruthfulQAResult (one per unique model).
    """
    # Deduplicate models
    seen: set[str] = set()
    unique_models: list[ModelSpec] = []
    for m in config.models:
        key = m.display_name()
        if key not in seen:
            seen.add(key)
            unique_models.append(m)

    if not unique_models:
        return []

    max_parallel = max(config.game_settings.max_parallel, 1)
    num_models = len(unique_models)

    print(f"\n{'='*60}")
    print(f"  Running TruthfulQA MC1 benchmark")
    print(f"  Questions per model: {config.truthfulqa.num_questions}")
    print(f"  Models: {num_models}")
    print(f"  Parallelism: {max_parallel}")
    print(f"{'='*60}")

    results: list[TruthfulQAResult] = []

    def _eval_one(model_spec: ModelSpec) -> TruthfulQAResult:
        return run_truthfulqa(
            model_spec,
            num_questions=config.truthfulqa.num_questions,
            temperature=config.truthfulqa.temperature,
            seed=config.seed or 42,
        )

    if max_parallel <= 1 or num_models <= 1:
        # Sequential
        for model_spec in unique_models:
            print(f"\n  Evaluating {model_spec.display_name()}...")
            try:
                tqa_result = _eval_one(model_spec)
                results.append(tqa_result)
                print(f"    Accuracy: {tqa_result.accuracy:.1%} ({tqa_result.correct}/{tqa_result.num_questions})")
            except Exception as exc:
                logger.error("TruthfulQA failed for %s: %s", model_spec.display_name(), exc)
                print(f"    FAILED: {exc}")
    else:
        # Parallel across models
        print(f"\n  Evaluating {num_models} models in parallel...\n")
        future_to_model: dict[Any, ModelSpec] = {}

        with ThreadPoolExecutor(max_workers=min(max_parallel, num_models)) as executor:
            for model_spec in unique_models:
                future = executor.submit(_eval_one, model_spec)
                future_to_model[future] = model_spec

            for future in as_completed(future_to_model):
                model_spec = future_to_model[future]
                try:
                    tqa_result = future.result()
                    results.append(tqa_result)
                    print(
                        f"  [{len(results)}/{num_models}] {model_spec.display_name()}: "
                        f"{tqa_result.accuracy:.1%} ({tqa_result.correct}/{tqa_result.num_questions})"
                    )
                except Exception as exc:
                    logger.error("TruthfulQA failed for %s: %s", model_spec.display_name(), exc)
                    print(f"  [{len(results)}/{num_models}] {model_spec.display_name()} FAILED: {exc}")

    return results


# ---------------------------------------------------------------------------
# Full eval run
# ---------------------------------------------------------------------------


def run_full_eval(config: EvalConfig, skip_truthfulqa: bool = False) -> EvalResults:
    """
    Run the complete eval pipeline: games + TruthfulQA.

    Parameters
    ----------
    config
        The eval configuration.
    skip_truthfulqa
        If True, skip the TruthfulQA benchmark.

    Returns
    -------
    EvalResults
        Aggregated results with Elo ratings, win rates, and TruthfulQA scores.
    """
    import shutil
    from datetime import datetime

    # Create a timestamped run folder inside output_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    config.output_dir = run_dir

    print(f"  Run folder: {os.path.abspath(run_dir)}")

    # Copy the source config into the run folder for reproducibility
    if config.source_path and os.path.isfile(config.source_path):
        shutil.copy2(config.source_path, os.path.join(run_dir, "config.yaml"))

    results = EvalResults()

    # ---- Game-based evals ----
    if config.game_settings.num_games > 0:
        print(f"\n{'='*60}")
        print(f"  Running {config.game_settings.num_games} Among Us games")
        print(f"  Players per game: {config.game_settings.players_per_game}")
        print(f"  Models in pool: {len(config.models)}")
        print(f"  Parallelism: {config.game_settings.max_parallel}")
        print(f"{'='*60}")

        records = run_games(config)
        results.game_records = records

        if records:
            # Compute Elo ratings
            results.elo_ratings = process_all_games(records)

            # Compute bootstrap CIs
            if len(records) >= 10:
                results.elo_confidence_intervals = bootstrap_elo_ci(
                    records, seed=config.seed,
                )

            # Compute win rates
            results.win_rates = compute_win_rates(records)

    # ---- TruthfulQA ----
    if not skip_truthfulqa and config.models:
        results.truthfulqa_scores = run_truthfulqa_eval(config)

    return results
