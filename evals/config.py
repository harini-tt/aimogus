"""
Configuration loading for the eval framework.

Parses a YAML config file into an EvalConfig dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, NINE_MEMBER_GAME


_GAME_CONFIGS = {
    5: FIVE_MEMBER_GAME,
    7: SEVEN_MEMBER_GAME,
    9: NINE_MEMBER_GAME,
}


@dataclass
class ModelSpec:
    """One model to evaluate."""

    provider: str  # "openai" | "openrouter"
    model: str  # e.g. "gpt-4o", "x-ai/grok-4.1-fast"

    def display_name(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class GameSettings:
    """Settings for the game-based evals (Elo + win rate)."""

    num_games: int = 50
    players_per_game: int = 7  # 5, 7, or 9
    max_parallel: int = 4  # max games to run concurrently (0 = sequential)


@dataclass
class TruthfulQASettings:
    """Settings for the TruthfulQA benchmark."""

    num_questions: int = 100
    judge_model: str = "gpt-4o"
    temperature: float = 0.0


@dataclass
class EvalConfig:
    """Top-level eval configuration."""

    models: list[ModelSpec] = field(default_factory=list)
    game_settings: GameSettings = field(default_factory=GameSettings)
    truthfulqa: TruthfulQASettings = field(default_factory=TruthfulQASettings)
    output_dir: str = "eval_results"
    seed: Optional[int] = None
    source_path: Optional[str] = None  # path to the YAML file this was loaded from

    @property
    def game_config(self) -> dict[str, Any]:
        """Return the game_config dict matching players_per_game."""
        cfg = _GAME_CONFIGS.get(self.game_settings.players_per_game)
        if cfg is None:
            raise ValueError(
                f"Unsupported players_per_game={self.game_settings.players_per_game}. "
                f"Choose from {list(_GAME_CONFIGS.keys())}."
            )
        return cfg

    def model_dicts(self) -> list[dict[str, str]]:
        """Return models as a list of dicts for the game engine."""
        return [{"provider": m.provider, "model": m.model} for m in self.models]


def load_config(path: str | Path) -> EvalConfig:
    """Load an EvalConfig from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())

    models = [
        ModelSpec(provider=m["provider"], model=m["model"])
        for m in raw.get("models", [])
    ]

    gs_raw = raw.get("game_settings", {})
    game_settings = GameSettings(
        num_games=gs_raw.get("num_games", 50),
        players_per_game=gs_raw.get("players_per_game", 7),
        max_parallel=gs_raw.get("max_parallel", 4),
    )

    tqa_raw = raw.get("truthfulqa", {})
    truthfulqa = TruthfulQASettings(
        num_questions=tqa_raw.get("num_questions", 100),
        judge_model=tqa_raw.get("judge_model", "gpt-4o"),
        temperature=tqa_raw.get("temperature", 0.0),
    )

    return EvalConfig(
        models=models,
        game_settings=game_settings,
        truthfulqa=truthfulqa,
        output_dir=raw.get("output_dir", "eval_results"),
        seed=raw.get("seed"),
        source_path=str(Path(path).resolve()),
    )
