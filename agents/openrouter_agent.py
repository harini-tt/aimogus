"""
Concrete agent backed by the OpenRouter API.

OpenRouter exposes an OpenAI-compatible endpoint, so we reuse the ``openai``
Python SDK and simply override ``base_url`` to point at OpenRouter.

The API key is read from ``OPENROUTER_API_KEY`` in the ``.env`` file.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import openai
from dotenv import load_dotenv

from agents.base_agent import BaseAgent
from agents.models import Role

logger = logging.getLogger(__name__)

# Ensure .env is loaded so OPENROUTER_API_KEY is available
load_dotenv()


class OpenRouterAgent(BaseAgent):
    """
    An Among Us agent powered by the OpenRouter API.

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
        OpenRouter model identifier.  Defaults to ``"openai/gpt-4o"``.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        game_instructions: str = "",
        assigned_tasks: list[str] | None = None,
        player_names: list[str] | None = None,
        model: str = "openai/gpt-4o",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            game_instructions=game_instructions,
            assigned_tasks=assigned_tasks,
        )
        self.model = model
        self.player_names: list[str] = player_names or []
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def format_context(self) -> list[dict[str, Any]]:
        """
        Build the OpenAI-compatible ``messages`` list from the agent's context.

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
        Call the OpenRouter Chat Completions API and return the assistant's
        text response.

        Any *kwargs* are forwarded to ``client.chat.completions.create``
        (e.g. ``temperature``, ``max_tokens``).
        """
        messages = self.format_context()

        # logger.info(
        #     "=== OpenRouter Request [%s] (model=%s) ===\n%s",
        #     self.name,
        #     kwargs.get("model", self.model),
        #     json.dumps(messages, indent=2),
        # )

        response = self.client.chat.completions.create(
            model=kwargs.pop("model", self.model),
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        completion_text = message.content or ""

        # Some reasoning models (e.g. kimi-k2.5) may place their actual
        # answer in a ``reasoning`` or ``reasoning_content`` attribute
        # instead of (or in addition to) ``content``.  When ``content``
        # comes back empty, try to recover from these alternative fields.
        if not completion_text.strip():
            reasoning = (
                getattr(message, "reasoning", None)
                or getattr(message, "reasoning_content", None)
            )
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                logger.info(
                    "[%s] content was empty; falling back to reasoning field.",
                    self.name,
                )
                completion_text = reasoning

        if not completion_text.strip():
            logger.warning(
                "[%s] OpenRouter returned a completely empty response "
                "(model=%s). This may cause action-parsing failures.",
                self.name,
                self.model,
            )

        logger.info(
            "=== OpenRouter Response [%s] ===\n%s",
            self.name,
            completion_text,
        )

        return completion_text
