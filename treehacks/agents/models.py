"""
Data models for the Among Us AI agent framework.

Defines the core types used to represent game state, interactions,
and agent context. All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Role assigned to an agent at the start of a game."""

    CREWMATE = "crewmate"
    IMPOSTOR = "impostor"


class ActionType(str, Enum):
    """Types of discrete actions an agent can perform."""

    KILL = "kill"
    VOTE = "vote"
    REPORT_BODY = "report_body"
    CALL_MEETING = "call_meeting"
    COMPLETE_TASK = "complete_task"
    FAKE_TASK = "fake_task"
    MOVE = "move"
    VENT = "vent"
    SPEAK = "speak"
    VIEW_MONITOR = "view_monitor"
    SABOTAGE = "sabotage"


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat utterance used inside meetings and encounters."""

    speaker: str
    content: str
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Interaction types
# ---------------------------------------------------------------------------


class EmergencyMeeting(BaseModel):
    """
    Multi-turn group conversation triggered by a body report or button press.

    This is the centerpiece of social deduction — all living agents discuss
    and then vote to eject a suspect.
    """

    interaction_type: Literal["emergency_meeting"] = "emergency_meeting"

    round: int
    called_by: str
    messages: list[Message] = Field(default_factory=list)
    body_reported: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class SystemEvent(BaseModel):
    """
    A broadcast from the game engine (vote results, round transitions, etc.).

    ``event_type`` values include:
        vote_result, round_start, round_end, player_ejected,
        game_start, game_over, meeting_called, task_update
    """

    interaction_type: Literal["system_event"] = "system_event"

    event_type: str
    content: str
    data: Optional[dict[str, Any]] = None
    round: int
    timestamp: float = Field(default_factory=time.time)


class Action(BaseModel):
    """
    A discrete action performed by an agent (kill, vote, move, etc.).
    """

    interaction_type: Literal["action"] = "action"

    action_type: ActionType
    actor: str
    target: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    round: int
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

Interaction = Annotated[
    Union[EmergencyMeeting, SystemEvent, Action],
    Field(discriminator="interaction_type"),
]
"""Union of all interaction types, discriminated on ``interaction_type``."""


# ---------------------------------------------------------------------------
# Agent context
# ---------------------------------------------------------------------------


class AgentContext(BaseModel):
    """
    Everything an agent knows: the game rules and its chronological
    log of interactions.
    """

    game_instructions: str = ""
    interactions: list[Interaction] = Field(default_factory=list)

    # -- helpers -------------------------------------------------------------

    def add_interaction(self, interaction: Interaction) -> None:
        """Append an interaction and keep the list sorted by timestamp."""
        self.interactions.append(interaction)
        self.interactions.sort(key=lambda i: i.timestamp)

    def get_meetings(self) -> list[EmergencyMeeting]:
        return [i for i in self.interactions if isinstance(i, EmergencyMeeting)]

    def get_system_events(self) -> list[SystemEvent]:
        return [i for i in self.interactions if isinstance(i, SystemEvent)]

    def get_actions(self) -> list[Action]:
        return [i for i in self.interactions if isinstance(i, Action)]

    def get_by_round(self, round: int) -> list[Interaction]:
        return [i for i in self.interactions if i.round == round]
