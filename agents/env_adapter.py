"""
Adapter layer that bridges the ``agents/`` framework (BaseAgent / OpenAIAgent)
with the ``envs/`` game engine (Player, Action classes, AmongUs game loop).

Provides:
    - ``EnvAgentAdapter``  — wraps a ``BaseAgent`` (or subclass) around an env
      ``Player`` so the game loop can call ``choose_action()`` and get back env
      Action objects.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from envs.player import Player
from agents.base_agent import BaseAgent
from agents.openai_agent import OpenAIAgent
from agents.models import (
    ActionType,
    EmergencyMeeting,
    Message,
    Role,
    SystemEvent,
    Action as PydanticAction,
)
from prompts.action_prompt import ACTION_PROMPT, OBSERVATION_LOCATION_PROMPT

# Env action classes — imported lazily in methods to avoid circular deps
import envs.action as env_action_mod


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

# Map env action *class name* -> agents.models.ActionType
_ENV_ACTION_TO_TYPE: dict[str, ActionType] = {
    "MoveTo": ActionType.MOVE,
    "Vent": ActionType.VENT,
    "Kill": ActionType.KILL,
    "Vote": ActionType.VOTE,
    "Speak": ActionType.SPEAK,
    "CallMeeting": ActionType.CALL_MEETING,
    "CompleteTask": ActionType.COMPLETE_TASK,
    "CompleteFakeTask": ActionType.FAKE_TASK,
    "ViewMonitor": ActionType.VIEW_MONITOR,
    "Sabotage": ActionType.SABOTAGE,
}

# Map env identity string -> agents.models.Role
_IDENTITY_TO_ROLE: dict[str, Role] = {
    "Crewmate": Role.CREWMATE,
    "Impostor": Role.IMPOSTOR,
}


def _best_match(response: str, candidates: list[str]) -> int:
    """Return the index of the candidate that best matches *response*."""
    response_lower = response.strip().lower()
    best_idx = 0
    best_score = 0.0
    for idx, candidate in enumerate(candidates):
        candidate_lower = candidate.strip().lower()
        # Exact substring match is best
        if candidate_lower in response_lower or response_lower in candidate_lower:
            score = len(candidate_lower) / max(len(response_lower), 1) + 1.0
        else:
            score = SequenceMatcher(None, response_lower, candidate_lower).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


# ---------------------------------------------------------------------------
# EnvAgentAdapter — the main integration class
# ---------------------------------------------------------------------------


class EnvAgentAdapter:
    """
    Wraps a :class:`BaseAgent` (e.g. ``OpenAIAgent``, ``OpenRouterAgent``)
    and an env ``Player`` so the ``AmongUs`` game loop can call
    ``choose_action()`` / ``choose_observation_location()`` and receive back
    env ``Action`` objects.

    Parameters
    ----------
    player : Player
        The env ``Player`` (``Crewmate`` or ``Impostor``) instance.
    tools : list[Any] | None
        Optional list of LangChain-style tools (currently unused by the
        OpenAI adapter but kept for API compatibility).
    agent : BaseAgent | None
        A pre-built agent instance.  When provided the adapter uses it
        directly instead of creating one internally.  This allows callers
        (e.g. ``AmongUs.initialize_agents``) to inject any ``BaseAgent``
        subclass (``OpenAIAgent``, ``OpenRouterAgent``, …).
    model : str
        Model identifier forwarded to the default ``OpenAIAgent`` when no
        *agent* is provided.
    """

    def __init__(
        self,
        player: Player,
        tools: list[Any] | None = None,
        agent: BaseAgent | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self.player = player

        if agent is not None:
            # Use the caller-provided agent directly
            self.agent = agent
        else:
            # Fall back to creating an OpenAIAgent (backward compatible)
            role = _IDENTITY_TO_ROLE.get(player.identity, Role.CREWMATE)
            task_names = [str(t) for t in player.tasks] if len(player.tasks) > 0 else []
            self.agent = OpenAIAgent(
                name=player.name,
                role=role,
                assigned_tasks=task_names,
                model=model,
            )

        self._tools = tools or []
        # Track how many observations we've already synced
        self._synced_obs_count: int = 0
        self._synced_action_count: int = 0

    # ------------------------------------------------------------------
    # Public interface expected by AmongUs.agent_step()
    # ------------------------------------------------------------------

    def choose_action(self) -> Any:
        """
        Ask the LLM to pick one of the player's available actions.

        Returns an env ``Action`` instance (e.g. ``MoveTo``, ``Kill``, …).
        """
        available = self.player.get_available_actions()
        if not available:
            # Fallback: should not happen, but just in case
            return env_action_mod.Speak(current_location=self.player.location)

        # Sync latest env observations into the agent's Pydantic context
        self._sync_env_to_agent()

        # Build the action-selection prompt from the player's env state
        action_text = self._build_action_prompt()

        # Remove any previous turn_prompt so only the latest one is in context
        self.agent.context.interactions = [
            i for i in self.agent.context.interactions
            if not (isinstance(i, SystemEvent) and i.event_type == "turn_prompt")
        ]

        # Inject as a system event so the LLM sees it in context
        self.agent.add_interaction(
            SystemEvent(
                event_type="turn_prompt",
                content=action_text,
                round=self.agent._current_round(),
            )
        )

        # Call the LLM
        response = self.agent.chat_completions(temperature=0.7, max_completion_tokens=256)

        # Parse the response into an env action
        chosen = self._parse_action_response(response, available)

        # If it's a Speak action, provide the LLM response as the message
        if isinstance(chosen, env_action_mod.Speak):
            # For speak, ask the LLM for what to say (or reuse response if
            # the response looks like dialogue rather than an action pick)
            message = self._generate_speech(response, available)
            chosen.provide_message(message)

        return chosen

    def choose_observation_location(self, rooms: Any) -> str:
        """Pick a room to observe via the security monitor."""
        room_list = list(rooms)
        prompt_text = OBSERVATION_LOCATION_PROMPT.format(
            rooms="\n".join(f"- {r}" for r in room_list)
        )
        self.agent.add_interaction(
            SystemEvent(
                event_type="monitor_prompt",
                content=prompt_text,
                round=self.agent._current_round(),
            )
        )
        response = self.agent.chat_completions(temperature=0.3, max_completion_tokens=64)
        # Fuzzy match to a valid room
        idx = _best_match(response, room_list)
        return room_list[idx]

    def inject_meeting_transcript(
        self,
        transcript: list[dict],
        called_by: str,
        round_num: int,
        body_reported: str | None = None,
    ) -> None:
        """
        Inject the full meeting discussion as a single ``EmergencyMeeting``
        interaction into the agent's context.

        Parameters
        ----------
        transcript : list[dict]
            List of ``{"speaker": name, "content": text}`` dicts.
        called_by : str
            Name of the player who triggered the meeting.
        round_num : int
            Meeting round number.
        body_reported : str | None
            Name of the dead player if a body was reported, else *None*.
        """
        messages = [
            Message(speaker=entry["speaker"], content=entry["content"])
            for entry in transcript
        ]
        meeting = EmergencyMeeting(
            round=round_num,
            called_by=called_by,
            messages=messages,
            body_reported=body_reported,
        )
        self.agent.add_interaction(meeting)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_env_to_agent(self) -> None:
        """
        Convert new env observations and important events into Pydantic
        ``Interaction`` objects and push them into the agent's context.
        """
        # Sync observation_history entries we haven't seen yet
        obs = self.player.observation_history
        new_obs = obs[self._synced_obs_count:]
        for entry in new_obs:
            self.agent.add_interaction(
                SystemEvent(
                    event_type="observation",
                    content=str(entry),
                    round=self.agent._current_round(),
                )
            )
        self._synced_obs_count = len(obs)

        # Sync action_history entries
        acts = self.player.action_history
        new_acts = acts[self._synced_action_count:]
        for record in new_acts:
            action_obj = record.get("action")
            if action_obj is None:
                continue
            cls_name = type(action_obj).__name__
            action_type = _ENV_ACTION_TO_TYPE.get(cls_name, ActionType.MOVE)
            target = None
            if hasattr(action_obj, "other_player"):
                target = str(action_obj.other_player.name) if action_obj.other_player else None
            metadata: dict[str, Any] = {}
            if hasattr(action_obj, "new_location"):
                metadata["destination"] = action_obj.new_location
            if hasattr(action_obj, "task"):
                metadata["task"] = str(action_obj.task)
            self.agent.add_interaction(
                PydanticAction(
                    action_type=action_type,
                    actor=self.player.name,
                    target=target,
                    metadata=metadata or None,
                    round=record.get("timestep", self.agent._current_round()),
                )
            )
        self._synced_action_count = len(acts)

    def _build_action_prompt(self) -> str:
        """Build the action prompt from the player's env state.

        Observations and action history are already tracked as individual
        messages in the agent's context, so they are NOT duplicated here.
        """
        # Phase / location header
        phase_info = self.player.location_info_prompt() if self.player.location_info else ""
        # Tasks (only incomplete ones get paths)
        tasks_text = self.player.tasks_prompt()
        # Available actions
        actions_text = self.player.available_actions_prompt()

        return ACTION_PROMPT.format(
            phase_info=phase_info,
            tasks_info=tasks_text,
            available_actions=actions_text,
        )

    def _parse_action_response(
        self, response: str, available: list[Any]
    ) -> Any:
        """Fuzzy-match the LLM's text response to one of the available env actions."""
        candidates = [str(a) for a in available]
        idx = _best_match(response, candidates)
        return available[idx]

    def _generate_speech(self, initial_response: str, available: list[Any]) -> str:
        """
        Generate a meeting speech.

        If the initial_response looks like dialogue (not a pure action pick),
        use it directly. Otherwise ask the LLM for a short statement.
        """
        # If the response doesn't look like an action string, treat it as speech
        action_strings = [str(a) for a in available]
        # Check if the response is very similar to an action — if not, it's speech
        best_score = max(
            SequenceMatcher(None, initial_response.strip().lower(), a.lower()).ratio()
            for a in action_strings
        ) if action_strings else 0.0

        if best_score < 0.5:
            # The LLM already produced speech-like text
            return initial_response.strip()

        # Otherwise, ask the LLM to speak
        self.agent.add_interaction(
            SystemEvent(
                event_type="speak_prompt",
                content=(
                    "You are in a meeting discussion. Say something strategic — "
                    "share observations, accuse someone, or defend yourself. "
                    "Keep it to 1-3 sentences."
                ),
                round=self.agent._current_round(),
            )
        )
        speech = self.agent.chat_completions(temperature=0.8, max_completion_tokens=200)
        return speech.strip()

