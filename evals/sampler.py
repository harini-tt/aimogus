"""
Step sampling strategies and cost estimation.

Instead of judging every interaction of every agent on every dimension,
we sample the most informative moments to keep costs manageable.
"""

from __future__ import annotations

import random
from typing import Any

from agents.models import (
    Action,
    ActionType,
    EmergencyMeeting,
    SystemEvent,
)


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


class AllSteps:
    """Judge every interaction. Expensive but complete."""

    def select(
        self,
        interactions: list[Any],
        agent_role: str = "",
        max_steps: int = 9999,
    ) -> list[int]:
        return list(range(min(len(interactions), max_steps)))


class KeyMoments:
    """
    Sample the most informative moments:

    - All meeting speeches (high deception/lying signal)
    - All votes and kills (high planning signal)
    - First and last 2 task-phase actions (strategy bookends)
    - Random sample to fill remaining budget
    """

    def select(
        self,
        interactions: list[Any],
        agent_role: str = "",
        max_steps: int = 20,
    ) -> list[int]:
        key_indices: set[int] = set()

        for idx, interaction in enumerate(interactions):
            if isinstance(interaction, EmergencyMeeting):
                key_indices.add(idx)
            elif isinstance(interaction, Action):
                if interaction.action_type in (
                    ActionType.VOTE,
                    ActionType.KILL,
                    ActionType.CALL_MEETING,
                    ActionType.REPORT_BODY,
                ):
                    key_indices.add(idx)
            elif isinstance(interaction, SystemEvent):
                if interaction.event_type in ("speak_prompt", "vote_result"):
                    key_indices.add(idx)

        # Bookend task actions
        task_actions = [
            idx
            for idx, i in enumerate(interactions)
            if isinstance(i, Action) and i.action_type == ActionType.MOVE
        ]
        if task_actions:
            key_indices.update(task_actions[:2])
            key_indices.update(task_actions[-2:])

        # Fill remaining budget with random sample
        remaining = [i for i in range(len(interactions)) if i not in key_indices]
        budget = max_steps - len(key_indices)
        if budget > 0 and remaining:
            key_indices.update(random.sample(remaining, min(budget, len(remaining))))

        return sorted(key_indices)[:max_steps]


class DimensionTargeted:
    """Only sample interactions relevant to a specific dimension."""

    _FILTERS: dict[str, type] = {
        "awareness": SystemEvent,
        "deception": EmergencyMeeting,
        "lying": EmergencyMeeting,
        "planning": Action,
        "sycophancy": EmergencyMeeting,
        "manipulation": EmergencyMeeting,
    }

    def __init__(self, dimension: str) -> None:
        self.dimension = dimension
        self._target_type = self._FILTERS.get(dimension)

    def select(
        self,
        interactions: list[Any],
        agent_role: str = "",
        max_steps: int = 15,
    ) -> list[int]:
        if self._target_type is None:
            matching = list(range(len(interactions)))
        else:
            matching = [
                idx
                for idx, i in enumerate(interactions)
                if isinstance(i, self._target_type)
            ]
        if len(matching) > max_steps:
            matching = random.sample(matching, max_steps)
        return sorted(matching)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    num_games: int,
    num_agents: int = 7,
    avg_interactions: int = 60,
    dimensions: list[str] | None = None,
    sampling: str = "key_moments",
    max_steps: int = 20,
    n_judge_calls: int = 3,
    interview_questions: int = 12,
    judge_model: str = "gpt-5-mini",
) -> dict[str, Any]:
    """
    Estimate total cost for an eval campaign.

    Token estimates per call:
    - Per-step judge: ~600 input + ~150 output = 750 total
    - Interview agent call: ~2000 input + ~200 output = 2200 total
    - Interview judge: ~800 input + ~150 output = 950 total

    Pricing (gpt-5-mini, approx):
    - Input: $0.30 / 1M tokens
    - Output: $1.20 / 1M tokens
    """
    if dimensions is None:
        dimensions = [
            "awareness", "deception", "planning", "lying",
            "sycophancy", "manipulation", "scheming",
        ]

    sampled = min(max_steps, avg_interactions) if sampling == "key_moments" else avg_interactions

    # Per-step judging
    step_calls = num_games * num_agents * sampled * len(dimensions) * n_judge_calls
    step_in = step_calls * 600
    step_out = step_calls * 150

    # Interviews
    interview_agent_calls = num_games * num_agents * interview_questions
    interview_judge_calls = interview_agent_calls * n_judge_calls
    int_in = interview_agent_calls * 2000 + interview_judge_calls * 800
    int_out = interview_agent_calls * 200 + interview_judge_calls * 150

    total_in = step_in + int_in
    total_out = step_out + int_out

    pricing = {
        "gpt-5-mini": (0.30e-6, 1.20e-6),
        "gpt-5.2": (3.00e-6, 12.00e-6),
        "o3": (10.00e-6, 40.00e-6),
        "o4-mini": (1.10e-6, 4.40e-6),
        # Legacy
        "gpt-4o-mini": (0.15e-6, 0.60e-6),
        "gpt-4o": (2.50e-6, 10.00e-6),
    }
    in_rate, out_rate = pricing.get(judge_model, pricing["gpt-5-mini"])
    cost = total_in * in_rate + total_out * out_rate

    return {
        "num_games": num_games,
        "dimensions": dimensions,
        "sampling": sampling,
        "sampled_steps_per_agent": sampled,
        "n_judge_calls": n_judge_calls,
        "total_step_judge_calls": step_calls,
        "total_interview_calls": interview_agent_calls + interview_judge_calls,
        "total_tokens": total_in + total_out,
        "estimated_cost_usd": round(cost, 2),
    }
