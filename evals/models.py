"""
Core data models for the eval framework.

Flat, serializable dataclasses extracted from post-game state.
Uses dataclasses (not Pydantic) to avoid coupling with the game
engine's Pydantic layer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Per-player summary
# ---------------------------------------------------------------------------


@dataclass
class PlayerSummary:
    """Per-player summary computed from one finished game."""

    name: str
    identity: str  # "Crewmate" | "Impostor"
    model: str  # e.g. "gpt-4o", "x-ai/grok-4.1-fast"
    provider: str  # "openai" | "openrouter"
    is_alive_at_end: bool
    kills: int = 0  # kills committed (impostor only)
    tasks_completed: int = 0
    tasks_assigned: int = 0
    votes_received_against: int = 0


# ---------------------------------------------------------------------------
# Game-level record
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """
    Complete structured extraction from one finished game.

    Sufficient for Elo computation and win-rate metrics.
    """

    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Outcome
    winner_code: int = 0  # 1-4, see envs/game.py check_game_over
    winner_side: str = ""  # "impostor" | "crewmate"
    total_timesteps: int = 0
    task_completion: float = 0.0

    # Per-player
    player_summaries: dict[str, PlayerSummary] = field(default_factory=dict)

    # Helpers
    def impostor_models(self) -> list[str]:
        return [
            ps.model
            for ps in self.player_summaries.values()
            if ps.identity == "Impostor"
        ]

    def crewmate_models(self) -> list[str]:
        return [
            ps.model
            for ps in self.player_summaries.values()
            if ps.identity == "Crewmate"
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.game_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


# ---------------------------------------------------------------------------
# Elo ratings container
# ---------------------------------------------------------------------------


@dataclass
class EloRatings:
    """Tracks Deception and Detection Elo for every model."""

    INITIAL_RATING: float = 1500.0

    deception: dict[str, float] = field(default_factory=dict)
    detection: dict[str, float] = field(default_factory=dict)
    deception_history: dict[str, list[float]] = field(default_factory=dict)
    detection_history: dict[str, list[float]] = field(default_factory=dict)
    games_as_impostor: dict[str, int] = field(default_factory=dict)
    games_as_crewmate: dict[str, int] = field(default_factory=dict)

    def ensure_model(self, model_id: str) -> None:
        """Initialize rating entries for *model_id* if they don't exist."""
        for d in (self.deception, self.detection):
            d.setdefault(model_id, self.INITIAL_RATING)
        for h in (self.deception_history, self.detection_history):
            h.setdefault(model_id, [])
        for c in (self.games_as_impostor, self.games_as_crewmate):
            c.setdefault(model_id, 0)


# ---------------------------------------------------------------------------
# TruthfulQA result per model
# ---------------------------------------------------------------------------


@dataclass
class TruthfulQAResult:
    """TruthfulQA benchmark result for one model."""

    model: str
    provider: str
    accuracy: float  # fraction of correct MC1 answers
    num_questions: int
    correct: int
    details: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level eval results container
# ---------------------------------------------------------------------------


@dataclass
class EvalResults:
    """Aggregated results from a full eval run."""

    game_records: list[GameRecord] = field(default_factory=list)
    elo_ratings: Optional[EloRatings] = None
    win_rates: dict[str, dict[str, Any]] = field(default_factory=dict)
    truthfulqa_scores: list[TruthfulQAResult] = field(default_factory=list)
    elo_confidence_intervals: dict[str, dict[str, Any]] = field(default_factory=dict)
