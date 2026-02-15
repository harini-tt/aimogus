#!/usr/bin/env python3
"""
CLI entry point for the Among Us agent evaluation framework.

Usage:
    python run_evals.py --config eval_config.yaml
    python run_evals.py --config eval_config.yaml --games-only
    python run_evals.py --config eval_config.yaml --truthfulqa-only
    python run_evals.py --config eval_config.yaml --skip-truthfulqa
    python run_evals.py --config eval_config.yaml --num-games 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("evals").setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Among Us Agent Eval Runner — TruthfulQA + Deception Elo + Win Rate",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML eval config file (e.g. eval_config.yaml).",
    )
    parser.add_argument(
        "--num-games", type=int, default=None,
        help="Override the number of games from the config.",
    )
    parser.add_argument(
        "--games-only", action="store_true",
        help="Only run Among Us games (skip TruthfulQA).",
    )
    parser.add_argument(
        "--truthfulqa-only", action="store_true",
        help="Only run TruthfulQA benchmark (skip games).",
    )
    parser.add_argument(
        "--skip-truthfulqa", action="store_true",
        help="Run games but skip TruthfulQA.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override the output directory from the config.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the random seed.",
    )
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Override max parallel games (e.g. --parallel 8). Set to 1 for sequential.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from evals.config import load_config
    from evals.runner import run_full_eval
    from evals.report import export_csv, print_summary

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.num_games is not None:
        config.game_settings.num_games = args.num_games
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.parallel is not None:
        config.game_settings.max_parallel = args.parallel

    # Handle mutually exclusive flags
    skip_truthfulqa = args.skip_truthfulqa or args.games_only

    if args.truthfulqa_only:
        config.game_settings.num_games = 0

    # Print config summary
    print(f"\n{'='*60}")
    print(f"  Among Us Eval Framework")
    print(f"{'='*60}")
    print(f"  Config: {args.config}")
    print(f"  Models: {len(config.models)}")
    for m in config.models:
        print(f"    - {m.display_name()}")
    print(f"  Games: {config.game_settings.num_games}")
    print(f"  Players per game: {config.game_settings.players_per_game}")
    print(f"  Parallel games: {config.game_settings.max_parallel}")
    print(f"  TruthfulQA: {'skip' if skip_truthfulqa else f'{config.truthfulqa.num_questions} questions'}")
    print(f"  Output: {config.output_dir}")
    if config.seed is not None:
        print(f"  Seed: {config.seed}")
    print(f"{'='*60}\n")

    # Run
    results = run_full_eval(config, skip_truthfulqa=skip_truthfulqa)

    # Print summary
    print_summary(results)

    # Export
    paths = export_csv(results, config.output_dir)
    print(f"Results exported to: {os.path.abspath(config.output_dir)}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nGame logs: {os.path.join(config.output_dir, 'logs', '')}")
    print()


if __name__ == "__main__":
    main()
