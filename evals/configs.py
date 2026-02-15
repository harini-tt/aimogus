"""
Predefined eval configurations.

Each config specifies: game settings, model pool, how many games,
which eval dimensions to run, and judge parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME


@dataclass
class EvalConfig:
    """Configuration for an eval run."""

    name: str = "default"
    num_games: int = 10
    game_config: dict = field(default_factory=lambda: SEVEN_MEMBER_GAME)
    game_config_name: str = "SEVEN_MEMBER_GAME"

    # Model assignment
    model_configs: list[dict] | None = None  # per-player; None → all gpt-4o
    model_pool: list[dict] | None = None  # for random assignment across games

    # Eval flags
    run_deterministic: bool = True
    run_interviews: bool = True
    run_judge: bool = False  # per-step LLM judge (expensive)
    run_alignment: bool = False  # sycophancy, deceptive alignment, etc.
    run_taxonomy: bool = False  # 6-class deception taxonomy
    run_elo: bool = True  # Elo computation across games

    # Judge parameters
    judge_model: str = "gpt-5-mini"
    judge_dimensions: list[str] = field(
        default_factory=lambda: ["awareness", "deception", "planning", "lying"]
    )
    judge_n_calls: int = 3
    judge_max_steps: int = 20
    judge_mode: str = "omniscient"  # "omniscient" or "observer"

    # Interview parameters
    interview_dimensions: list[str] = field(
        default_factory=lambda: ["self_knowledge", "memory", "reasoning", "reflection"]
    )

    # Output
    output_dir: str = "evals/results"
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Pre-built configurations
# ---------------------------------------------------------------------------


# Quick dev test: 3 games, deterministic metrics only
QUICK_TEST = EvalConfig(
    name="quick_test",
    num_games=3,
    game_config=FIVE_MEMBER_GAME,
    game_config_name="FIVE_MEMBER_GAME",
    run_deterministic=True,
    run_interviews=False,
    run_judge=False,
    run_alignment=False,
    run_elo=True,
    seed=42,
)


# Standard capabilities eval: 20 games with interviews
CAPABILITIES_EVAL = EvalConfig(
    name="capabilities",
    num_games=20,
    game_config=FIVE_MEMBER_GAME,
    game_config_name="FIVE_MEMBER_GAME",
    run_deterministic=True,
    run_interviews=True,
    run_judge=True,
    run_alignment=False,
    run_elo=True,
    judge_dimensions=["awareness", "deception", "planning", "lying"],
    judge_model="gpt-5.2",
)


# Full alignment eval: 10 games with everything
ALIGNMENT_EVAL = EvalConfig(
    name="alignment",
    num_games=10,
    game_config=SEVEN_MEMBER_GAME,
    game_config_name="SEVEN_MEMBER_GAME",
    run_deterministic=True,
    run_interviews=True,
    run_judge=True,
    run_alignment=True,
    run_taxonomy=True,
    run_elo=True,
    judge_dimensions=[
        "awareness", "deception", "planning", "lying",
        "sycophancy", "manipulation", "scheming",
    ],
    judge_model="o3",
)


# Elo-only: many games, minimal per-game analysis
ELO_BENCHMARK = EvalConfig(
    name="elo_benchmark",
    num_games=100,
    game_config=SEVEN_MEMBER_GAME,
    game_config_name="SEVEN_MEMBER_GAME",
    run_deterministic=True,
    run_interviews=False,
    run_judge=False,
    run_alignment=False,
    run_elo=True,
)
