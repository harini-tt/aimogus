"""
Concrete agent backed by a local HuggingFace model.

Used for GRPO post-training: runs inference via ``model.generate()`` on a
GPU and logs every ``(input_ids, output_ids)`` pair for trajectory
collection.

Thread-safety: multiple concurrent games share a single ``(model, tokenizer)``
pair.  A ``threading.Lock`` serialises ``model.generate()`` calls so the GPU
is never hit from two threads simultaneously.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agents.base_agent import BaseAgent
from agents.models import Role

logger = logging.getLogger(__name__)


class LocalModelAgent(BaseAgent):
    """
    An Among Us agent powered by a local HuggingFace causal-LM.

    Parameters
    ----------
    name:
        Display name (e.g. "Player 1").
    role:
        ``Role.CREWMATE`` or ``Role.IMPOSTOR``.
    game_instructions:
        Optional override for the system prompt.  When set,
        ``build_system_message()`` returns this string verbatim (used for
        inoculation prompting).
    assigned_tasks:
        Tasks the agent must complete (or fake).
    player_names:
        Names of all players in the game.
    game_config_block:
        Formatted game-config string injected into the system prompt.
    known_impostors:
        List of impostor names (only provided to impostor agents).
    model_instance:
        A **shared** ``transformers.AutoModelForCausalLM`` already loaded
        on GPU.  Multiple ``LocalModelAgent`` instances reference the same
        object.
    tokenizer:
        A **shared** ``transformers.AutoTokenizer``.
    inference_lock:
        A **shared** ``threading.Lock`` that serialises GPU calls.
    trajectory:
        A **per-game** mutable list.  Each LLM call appends a dict with
        ``input_ids`` and ``output_ids`` tensors (moved to CPU).  The
        rollout wrapper attaches rewards after the game ends.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        game_instructions: str = "",
        assigned_tasks: list[str] | None = None,
        player_names: list[str] | None = None,
        game_config_block: str = "",
        known_impostors: list[str] | None = None,
        # ---- local-model-specific ----
        model_instance: Any = None,
        tokenizer: Any = None,
        inference_lock: threading.Lock | None = None,
        trajectory: list | None = None,
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            game_instructions=game_instructions,
            assigned_tasks=assigned_tasks,
            game_config_block=game_config_block,
            known_impostors=known_impostors,
        )
        self.player_names: list[str] = player_names or []
        self._model = model_instance
        self._tokenizer = tokenizer
        self._lock = inference_lock or threading.Lock()
        self._trajectory: list[dict] = trajectory if trajectory is not None else []

    # ------------------------------------------------------------------
    # Abstract interface implementation
    # ------------------------------------------------------------------

    def format_context(self) -> list[dict[str, Any]]:
        """Build an OpenAI-compatible ``messages`` list.

        Identical logic to ``OpenAIAgent.format_context()``:
        system message + chronological interaction log.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_message(self.player_names)},
        ]
        for interaction in self.context.interactions:
            messages.extend(self.render_interaction(interaction))
        return messages

    def chat_completions(self, **kwargs: Any) -> str:
        """Run local ``model.generate()`` and return the text completion.

        Also logs ``(input_ids, output_ids)`` to ``self._trajectory`` for
        GRPO training.  Keyword arguments accepted:

        * ``temperature`` (float, default 0.7)
        * ``max_new_tokens`` (int, default 256)
        """
        messages = self.format_context()

        # Tokenise using the model's chat template.
        # transformers 5.x may return a BatchEncoding instead of a raw
        # tensor, so we extract input_ids explicitly.
        tokenized = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids.to(self._model.device)
        elif hasattr(tokenized, "to"):
            input_ids = tokenized.to(self._model.device)
        else:
            import torch
            input_ids = torch.tensor([tokenized], dtype=torch.long).to(self._model.device)

        temp = kwargs.get("temperature", 0.7)
        max_new = kwargs.get("max_new_tokens", 256)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": temp > 0,
        }
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.95

        # Serialise GPU access across concurrent games
        with self._lock:
            output = self._model.generate(input_ids, **gen_kwargs)

        # Extract only the newly generated tokens
        new_tokens = output[0, input_ids.shape[1]:]
        completion_text = self._tokenizer.decode(
            new_tokens, skip_special_tokens=True,
        )

        # Log for trajectory collection (tensors moved to CPU)
        self._trajectory.append({
            "input_ids": input_ids[0].cpu(),
            "output_ids": new_tokens.cpu(),
            "full_text": completion_text,
        })

        logger.info(
            "=== Local Model Response [%s] ===\n%s",
            self.name,
            completion_text,
        )

        return completion_text
