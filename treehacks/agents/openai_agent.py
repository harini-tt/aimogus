"""
Concrete agent backed by the OpenAI Chat Completions API (gpt-5.2).

Uses the raw ``openai`` Python SDK and loads the API key from a ``.env``
file via ``python-dotenv``.
"""

from __future__ import annotations

from typing import Any

import openai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from agents.models import (
    EmergencyMeeting,
    Role,
    SystemEvent,
    Action,
)
from prompts import SYSTEM_PROMPT

# Ensure .env is loaded so OPENAI_API_KEY is available
load_dotenv()

# Role-specific instruction blocks injected into the system prompt
_ROLE_INSTRUCTIONS = {
    Role.CREWMATE: (
        "You are a CREWMATE. Your goal is to complete all your tasks and "
        "figure out who the Impostor is. Pay attention to suspicious "
        "behaviour — alibi inconsistencies, people appearing near bodies, "
        "or anyone faking tasks. Share your observations during meetings "
        "and vote wisely."
    ),
    Role.IMPOSTOR: (
        "You are the IMPOSTOR. Your goal is to eliminate Crewmates without "
        "getting caught. During meetings, lie convincingly — build fake "
        "alibis, deflect accusations, and subtly cast suspicion on innocent "
        "players. Avoid being too aggressive or too quiet; both draw "
        "attention. Pretend to do tasks and act like a normal Crewmate."
    ),
}


class OpenAIAgent(BaseAgent):
    """
    An Among Us agent powered by OpenAI's Chat Completions API.

    Parameters
    ----------
    name:
        Display name (e.g. "Red").
    role:
        ``Role.CREWMATE`` or ``Role.IMPOSTOR``.
    game_instructions:
        Optional override for the system prompt.  If not provided the
        default template from ``prompts/system.txt`` is used.
    assigned_tasks:
        Tasks the agent must complete (or fake).
    player_names:
        Names of all players in the game (used to populate the system prompt).
    model:
        OpenAI model identifier.  Defaults to ``"gpt-5.2"``.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        game_instructions: str = "",
        assigned_tasks: list[str] | None = None,
        player_names: list[str] | None = None,
        model: str = "gpt-5.2",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            game_instructions=game_instructions,
            assigned_tasks=assigned_tasks,
        )
        self.model = model
        self.player_names: list[str] = player_names or []
        self.client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def format_context(self) -> list[dict[str, Any]]:
        """
        Build the OpenAI ``messages`` list from the agent's context.

        Structure:
            1. System message  — rendered from the system prompt template.
            2. Interaction log — each interaction appended chronologically
               as user / assistant messages.
        """
        messages: list[dict[str, Any]] = []

        # --- 1. System message -------------------------------------------
        system_text = self.context.game_instructions or SYSTEM_PROMPT.format(
            name=self.name,
            role=self.role.value.upper(),
            role_instructions=_ROLE_INSTRUCTIONS.get(self.role, ""),
            player_list=", ".join(self.player_names) if self.player_names else "Unknown",
            assigned_tasks=(
                "\n".join(f"- {t}" for t in self.assigned_tasks)
                if self.assigned_tasks
                else "No tasks assigned."
            ),
        )
        messages.append({"role": "system", "content": system_text})

        # --- 2. Interaction history --------------------------------------
        for interaction in self.context.interactions:
            messages.extend(self._render_interaction(interaction))

        return messages

    def chat_completions(self, **kwargs: Any) -> str:
        """
        Call the OpenAI Chat Completions API and return the assistant's
        text response.

        Any *kwargs* are forwarded to ``client.chat.completions.create``
        (e.g. ``temperature``, ``max_tokens``).
        """
        messages = self.format_context()
        response = self.client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Interaction rendering helpers
    # ------------------------------------------------------------------

    def _render_interaction(
        self,
        interaction: EmergencyMeeting | SystemEvent | Action,
    ) -> list[dict[str, str]]:
        """Convert a single interaction into OpenAI message dicts."""
        if isinstance(interaction, EmergencyMeeting):
            return self._render_meeting(interaction)
        if isinstance(interaction, SystemEvent):
            return self._render_system_event(interaction)
        if isinstance(interaction, Action):
            return self._render_action(interaction)
        return []

    def _render_meeting(self, meeting: EmergencyMeeting) -> list[dict[str, str]]:
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

    def _render_system_event(self, event: SystemEvent) -> list[dict[str, str]]:
        """Render a system event as a single user message."""
        return [{
            "role": "user",
            "content": f"[SYSTEM — {event.event_type}]: {event.content}",
        }]

    def _render_action(self, action: Action) -> list[dict[str, str]]:
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
