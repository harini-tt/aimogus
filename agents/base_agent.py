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
    EmergencyMeeting,
    Interaction,
    Role,
    SystemEvent,
)
from prompts import ROLE_INSTRUCTIONS, SYSTEM_PROMPT


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
        game_config_block: str = "",
        known_impostors: list[str] | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.is_alive: bool = True
        self.assigned_tasks: list[str] = assigned_tasks or []
        self.completed_tasks: list[str] = []
        self.context = AgentContext(game_instructions=game_instructions)
        self.game_config_block = game_config_block or "Not provided."
        self.known_impostors = known_impostors or []

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
    # System prompt assembly
    # ------------------------------------------------------------------

    def build_system_message(self, player_names: list[str] | None = None) -> str:
        """
        Render the system prompt from the template, substituting the
        agent's identity, role instructions, player list, and tasks.

        If ``self.context.game_instructions`` is set, that string is
        returned verbatim (caller-provided override).
        """
        if self.context.game_instructions:
            return self.context.game_instructions
        known_impostors_text = (
            ", ".join(self.known_impostors)
            if self.role == Role.IMPOSTOR and self.known_impostors
            else "Hidden (you do not know other roles)"
        )
        return SYSTEM_PROMPT.format(
            name=self.name,
            role=self.role.value.upper(),
            role_instructions=ROLE_INSTRUCTIONS.get(self.role.value, ""),
            player_list=", ".join(player_names) if player_names else "Unknown",
            assigned_tasks=(
                "\n".join(f"- {t}" for t in self.assigned_tasks)
                if self.assigned_tasks
                else "No tasks assigned."
            ),
            game_config_block=self.game_config_block,
            known_impostors=known_impostors_text,
        )

    # ------------------------------------------------------------------
    # Interaction rendering — converts Pydantic models to message dicts
    # ------------------------------------------------------------------

    def render_interaction(
        self,
        interaction: EmergencyMeeting | SystemEvent | Action,
    ) -> list[dict[str, str]]:
        """Convert a single interaction into chat-completions message dicts."""
        if isinstance(interaction, EmergencyMeeting):
            return self.render_meeting(interaction)
        if isinstance(interaction, SystemEvent):
            return self.render_system_event(interaction)
        if isinstance(interaction, Action):
            return self.render_action(interaction)
        return []

    def render_meeting(self, meeting: EmergencyMeeting) -> list[dict[str, str]]:
        """Render an emergency meeting as a sequence of messages."""
        msgs: list[dict[str, str]] = []

        # Meeting header
        header = f"[EMERGENCY MEETING — Round {meeting.round}] "
        if meeting.body_reported:
            header += (
                f"{meeting.called_by} reported {meeting.body_reported}'s body."
            )
        else:
            header += f"{meeting.called_by} called an emergency meeting."
        msgs.append({"role": "user", "content": header})

        # Conversation turns
        for msg in meeting.messages:
            if msg.speaker == self.name:
                msgs.append({"role": "assistant", "content": msg.content})
            else:
                msgs.append({
                    "role": "user",
                    "content": f"[{msg.speaker}]: {msg.content}",
                })

        return msgs

    def render_system_event(self, event: SystemEvent) -> list[dict[str, str]]:
        """Render a system event as a single user message."""
        return [{
            "role": "user",
            "content": f"[SYSTEM — {event.event_type}]: {event.content}",
        }]

    def render_action(self, action: Action) -> list[dict[str, str]]:
        """Render an action as a brief note the agent 'remembers'."""
        if action.actor == self.name:
            # First-person: something *this* agent did
            parts = [f"[YOUR ACTION — {action.action_type.value}]"]
            if action.target:
                parts.append(f"Target: {action.target}.")
            if action.metadata:
                details = ", ".join(f"{k}: {v}" for k, v in action.metadata.items())
                parts.append(f"({details})")
            return [{"role": "user", "content": " ".join(parts)}]
        else:
            # Third-person: something *another* agent did that this agent saw
            parts = [f"[OBSERVED — {action.action_type.value}]"]
            parts.append(f"{action.actor}")
            if action.target:
                parts.append(f"-> {action.target}.")
            if action.metadata:
                details = ", ".join(f"{k}: {v}" for k, v in action.metadata.items())
                parts.append(f"({details})")
            return [{"role": "user", "content": " ".join(parts)}]

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
