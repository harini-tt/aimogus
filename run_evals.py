#!/usr/bin/env python3
"""
CLI entry point for the Among Us agent evaluation framework.

Usage:
    python run_evals.py                          # quick 3-game test
    python run_evals.py --preset capabilities    # 20-game capabilities eval
    python run_evals.py --preset alignment       # 10-game alignment eval
    python run_evals.py --preset elo             # 100-game Elo benchmark
    python run_evals.py --num-games 50 --no-interviews
    python run_evals.py --cost-estimate          # estimate cost without running
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("evals").setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Among Us Agent Eval Runner")
    parser.add_argument(
        "--preset",
        choices=["quick", "capabilities", "alignment", "elo"],
        default="quick",
        help="Predefined eval configuration (default: quick)",
    )
    parser.add_argument("--num-games", type=int, default=None, help="Override number of games")
    parser.add_argument("--players", type=int, choices=[5, 7], default=None, help="Override player count")
    parser.add_argument("--no-interviews", action="store_true", help="Skip post-game interviews")
    parser.add_argument("--no-judge", action="store_true", help="Skip per-step LLM judging")
    parser.add_argument("--with-alignment", action="store_true", help="Enable alignment metrics")
    parser.add_argument("--with-taxonomy", action="store_true", help="Enable deception taxonomy")
    parser.add_argument("--judge-model", type=str, default=None, help="Override judge model")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for model assignment")
    parser.add_argument(
        "--cost-estimate", action="store_true",
        help="Print cost estimate and exit without running games",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from evals.configs import (
        QUICK_TEST,
        CAPABILITIES_EVAL,
        ALIGNMENT_EVAL,
        ELO_BENCHMARK,
    )

    # Select preset
    presets = {
        "quick": QUICK_TEST,
        "capabilities": CAPABILITIES_EVAL,
        "alignment": ALIGNMENT_EVAL,
        "elo": ELO_BENCHMARK,
    }
    config = presets[args.preset]

    # Apply overrides
    if args.num_games is not None:
        config.num_games = args.num_games
    if args.players is not None:
        from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME
        config.game_config = FIVE_MEMBER_GAME if args.players == 5 else SEVEN_MEMBER_GAME
        config.game_config_name = "FIVE_MEMBER_GAME" if args.players == 5 else "SEVEN_MEMBER_GAME"
    if args.no_interviews:
        config.run_interviews = False
    if args.no_judge:
        config.run_judge = False
    if args.with_alignment:
        config.run_alignment = True
    if args.with_taxonomy:
        config.run_taxonomy = True
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed

    # Cost estimate mode
    if args.cost_estimate:
        from evals.sampler import estimate_cost
        est = estimate_cost(
            num_games=config.num_games,
            num_agents=config.game_config["num_players"],
            dimensions=config.judge_dimensions if config.run_judge else [],
            n_judge_calls=config.judge_n_calls,
            interview_questions=12 if config.run_interviews else 0,
            judge_model=config.judge_model,
        )
        print("\n--- Cost Estimate ---")
        for k, v in est.items():
            print(f"  {k}: {v}")
        print()
        return

    # Run
    print(f"\n{'='*60}")
    print(f"Among Us Eval: {config.name}")
    print(f"  Games: {config.num_games}")
    print(f"  Config: {config.game_config_name}")
    print(f"  Deterministic: {config.run_deterministic}")
    print(f"  Interviews: {config.run_interviews}")
    print(f"  LLM Judge: {config.run_judge}")
    print(f"  Alignment: {config.run_alignment}")
    print(f"  Taxonomy: {config.run_taxonomy}")
    print(f"  Elo: {config.run_elo}")
    print(f"  Output: {config.output_dir}")
    print(f"{'='*60}\n")

    from evals.runner import run_eval
    from evals.report import export_csv, print_summary

    results = run_eval(config)

    # Print summary
    print_summary(results["records"], results["all_metrics"])

    # Print Elo leaderboard
    elo = results.get("elo_ratings")
    if elo:
        print("\n--- Deception Elo (Impostor) ---")
        for model in sorted(elo.deception, key=lambda m: elo.deception[m], reverse=True):
            games = elo.games_as_impostor.get(model, 0)
            print(f"  {model:<40} Elo={elo.deception[model]:>7.1f}  ({games} games)")

        print("\n--- Detection Elo (Crewmate) ---")
        for model in sorted(elo.detection, key=lambda m: elo.detection[m], reverse=True):
            games = elo.games_as_crewmate.get(model, 0)
            print(f"  {model:<40} Elo={elo.detection[model]:>7.1f}  ({games} games)")

    # Export
    paths = export_csv(results["records"], results["all_metrics"], config.output_dir)
    print(f"\nResults exported to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
