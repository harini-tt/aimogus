"""
Abstract base class for all Among Us AI agents.

Subclasses must implement ``chat_completions`` and ``format_context`` with
whichever LLM provider they choose (OpenAI SDK, LangChain, Anthropic, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.models import (
    Action,
    ActionType,
    AgentContext,
    Interaction,
    Role,
)


class BaseAgent(ABC):
    """
    Base class that every AI agent in the Among Us simulation inherits from.

    Parameters
    ----------
    name:
        Display name for the agent (e.g. "Red", "Blue").
    role:
        The agent's secret role (``Role.CREWMATE`` or ``Role.IMPOSTOR``).
    game_instructions:
        The system-level prompt describing the game rules, the agent's
        objectives, and any behavioural guidance.
    assigned_tasks:
        List of task names the agent is responsible for completing
        (or pretending to complete, if impostor).
    """

    def __init__(
        self,
        name: str,
        role: Role,
        game_instructions: str = "",
        assigned_tasks: list[str] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.is_alive: bool = True
        self.assigned_tasks: list[str] = assigned_tasks or []
        self.completed_tasks: list[str] = []
        self.context = AgentContext(game_instructions=game_instructions)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def chat_completions(self, **kwargs: Any) -> str:
        """
        Call the underlying LLM and return its text response.

        The implementation should call ``self.format_context()`` to build
        the message payload, send it to the model, and return the
        assistant's reply as a plain string.

        Keyword arguments are forwarded to the LLM call so callers can
        override temperature, model name, max tokens, etc.
        """
        ...

    @abstractmethod
    def format_context(self) -> list[dict[str, Any]]:
        """
        Convert ``self.context`` into the message format expected by the
        agent's LLM provider.

        For OpenAI-compatible APIs this would typically be a list of dicts
        like ``[{"role": "system", "content": ...}, ...]``.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def die(self) -> None:
        """Mark the agent as dead."""
        self.is_alive = False

    def complete_task(self, task: str) -> None:
        """
        Record a task as completed.

        Moves *task* from ``assigned_tasks`` to ``completed_tasks`` and
        logs a ``COMPLETE_TASK`` (or ``FAKE_TASK`` for impostors) action
        in the agent's context.
        """
        action_type = (
            ActionType.COMPLETE_TASK
            if self.role == Role.CREWMATE
            else ActionType.FAKE_TASK
        )

        if task in self.assigned_tasks:
            self.assigned_tasks.remove(task)
        self.completed_tasks.append(task)

        self.context.add_interaction(
            Action(
                action_type=action_type,
                actor=self.name,
                metadata={"task": task},
                round=self._current_round(),
            )
        )

    def add_interaction(self, interaction: Interaction) -> None:
        """Convenience proxy to ``self.context.add_interaction``."""
        self.context.add_interaction(interaction)

    def reset(self, game_instructions: str | None = None) -> None:
        """
        Reset the agent for a new game.

        Restores ``is_alive``, clears task lists and interaction history.
        Optionally accepts new game instructions; otherwise keeps the
        existing ones.
        """
        self.is_alive = True
        self.assigned_tasks = []
        self.completed_tasks = []
        self.context = AgentContext(
            game_instructions=game_instructions or self.context.game_instructions,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _current_round(self) -> int:
        """
        Best-effort guess at the current round number based on the most
        recent interaction.  Returns ``0`` if there are no interactions yet.
        """
        if self.context.interactions:
            return self.context.interactions[-1].round
        return 0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "alive" if self.is_alive else "dead"
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"role={self.role.value} {status}>"
        )
