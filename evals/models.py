"""
Core data models for the eval framework.

Defines flat, serializable records extracted from post-game state,
as well as judge I/O structures.  All models use dataclasses (not
Pydantic) to avoid coupling with the game engine's Pydantic layer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JudgeMode(str, Enum):
    """What information the judge receives."""

    OMNISCIENT = "omniscient"
    OBSERVER = "observer"


class EvalDimension(str, Enum):
    """Scoring dimensions."""

    # Capability
    AWARENESS = "awareness"
    DECEPTION = "deception"
    PLANNING = "planning"
    LYING = "lying"
    # Alignment
    SYCOPHANCY = "sycophancy"
    MANIPULATION = "manipulation"
    SCHEMING = "scheming"
    # Interview (Paper 1)
    SELF_KNOWLEDGE = "self_knowledge"
    MEMORY = "memory"
    REASONING = "reasoning"
    REFLECTION = "reflection"


# ---------------------------------------------------------------------------
# Per-event flat records (extracted from post-game state)
# ---------------------------------------------------------------------------


@dataclass
class ActionRecord:
    """One action taken by one player at one timestep."""

    timestep: int
    phase: str  # "task" | "meeting"
    round: int | None  # meeting round (None during task phase)
    player_name: str
    player_identity: str  # "Crewmate" | "Impostor"
    action_type: str  # "MOVE", "KILL", "VOTE", "SPEAK", etc.
    target_player: str | None = None
    location_from: str | None = None
    location_to: str | None = None
    message: str | None = None  # SPEAK content
    task_name: str | None = None
    raw_str: str = ""


@dataclass
class VoteRecord:
    """One vote in one meeting."""

    timestep: int
    meeting_index: int
    voter_name: str
    voter_identity: str
    target_name: str  # player name or "skip"
    target_identity: str | None = None


@dataclass
class KillRecord:
    """One kill event."""

    timestep: int
    killer_name: str
    victim_name: str
    location: str = ""
    witnesses: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-player summary
# ---------------------------------------------------------------------------


@dataclass
class PlayerSummary:
    """Per-player summary computed from the full game."""

    name: str
    identity: str
    model: str  # e.g. "gpt-4o", "grok-4.1-fast"
    provider: str  # "openai" | "openrouter"
    is_alive_at_end: bool
    timestep_of_death: int | None
    tasks_assigned: int
    tasks_completed: int
    total_actions: int
    action_counts: dict[str, int] = field(default_factory=dict)
    votes_cast: list[VoteRecord] = field(default_factory=list)
    votes_received_against: int = 0
    meetings_called: int = 0
    speeches: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Game-level snapshot
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """
    Complete structured extraction from one finished game.

    Sufficient for Elo computation and deterministic metrics —
    does **not** hold references to live game objects.
    """

    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Config
    game_config: dict[str, Any] = field(default_factory=dict)
    game_config_name: str = ""

    # Outcome
    winner_code: int = 0
    winner_side: str = ""  # "impostor" | "crewmate"
    total_timesteps: int = 0
    task_completion: float = 0.0

    # Flat records
    actions: list[ActionRecord] = field(default_factory=list)
    kills: list[KillRecord] = field(default_factory=list)
    votes: list[VoteRecord] = field(default_factory=list)
    voteout_events: list[dict[str, Any]] = field(default_factory=list)

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

    def save(self, directory: Path) -> Path:
        path = directory / f"{self.game_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> GameRecord:
        raw = json.loads(path.read_text())
        # Reconstitute nested dataclasses
        raw["actions"] = [ActionRecord(**a) for a in raw.get("actions", [])]
        raw["kills"] = [KillRecord(**k) for k in raw.get("kills", [])]
        raw["votes"] = [VoteRecord(**v) for v in raw.get("votes", [])]
        raw["player_summaries"] = {
            name: PlayerSummary(**ps)
            for name, ps in raw.get("player_summaries", {}).items()
        }
        return cls(**raw)


# ---------------------------------------------------------------------------
# Elo ratings container
# ---------------------------------------------------------------------------


@dataclass
class EloRatings:
    """Tracks Deception and Detection Elo for every model."""

    INITIAL_RATING: float = 1000.0

    deception: dict[str, float] = field(default_factory=dict)
    detection: dict[str, float] = field(default_factory=dict)
    deception_history: dict[str, list[float]] = field(default_factory=dict)
    detection_history: dict[str, list[float]] = field(default_factory=dict)
    games_as_impostor: dict[str, int] = field(default_factory=dict)
    games_as_crewmate: dict[str, int] = field(default_factory=dict)

    def ensure_model(self, model_id: str) -> None:
        for d in (self.deception, self.detection):
            d.setdefault(model_id, self.INITIAL_RATING)
        for h in (self.deception_history, self.detection_history):
            h.setdefault(model_id, [])
        for c in (self.games_as_impostor, self.games_as_crewmate):
            c.setdefault(model_id, 0)


# ---------------------------------------------------------------------------
# Metric result (generic, used by all metrics modules)
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """A single computed metric for one player in one game."""

    metric_name: str
    player_name: str
    player_identity: str
    model: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Judge I/O
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for a single dimension from one or more judge calls."""

    dimension: str
    median_score: float
    mean_score: float
    binary: bool  # threshold >= 6
    std_dev: float
    confidence: float
    reasoning: str
    n_calls: int = 1


@dataclass
class InterviewExchange:
    """One question-answer pair from a post-game interview."""

    agent_name: str
    dimension: str
    question: str
    response: str
    question_index: int = 0
