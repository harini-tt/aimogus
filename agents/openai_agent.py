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
from agents.models import Role

# Ensure .env is loaded so OPENAI_API_KEY is available
load_dotenv()


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
        default template from ``prompts/system_prompt.py`` is used.
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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_message(self.player_names)},
        ]

        for interaction in self.context.interactions:
            messages.extend(self.render_interaction(interaction))

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
