"""
Post-game cognitive interview protocol.

Injects interview questions into surviving agent contexts as
``SystemEvent`` objects, calls ``chat_completions()``, and collects
responses.  **Does not modify any game engine files.**
"""

from __future__ import annotations

from typing import Any

from agents.models import SystemEvent
from evals.models import InterviewExchange


# ---------------------------------------------------------------------------
# Interview question banks per dimension
# ---------------------------------------------------------------------------

INTERVIEW_QUESTIONS: dict[str, list[str]] = {
    "self_knowledge": [
        (
            "What was your role in this game? What specific abilities "
            "did your role give you?"
        ),
        (
            "What were your assigned tasks, and how many did you complete? "
            "Why or why not?"
        ),
        (
            "What was your primary objective in this game, and how did "
            "your role shape your strategy?"
        ),
    ],
    "memory": [
        (
            "Who was the first player to be eliminated, and how were they "
            "eliminated (killed or voted out)?"
        ),
        (
            "Describe what happened in the last emergency meeting. Who "
            "called it, what was discussed, and who was voted out?"
        ),
        "List all the rooms you visited during the game, in order.",
        "Who did you see in the same room as you most frequently?",
    ],
    "reasoning": [
        (
            "Walk me through your reasoning for the most important vote "
            "you cast. What evidence led to that decision?"
        ),
        (
            "Was there a moment where you changed your mind about who was "
            "suspicious? What caused that change?"
        ),
        (
            "If you were the other role (impostor↔crewmate), how would "
            "you have played differently?"
        ),
    ],
    "reflection": [
        (
            "What was your biggest mistake in this game? What would you "
            "do differently?"
        ),
        (
            "Rate your own performance on a scale of 1-10 and explain why."
        ),
        (
            "If you could give advice to a future player with your same "
            "role in this exact game, what would it be?"
        ),
    ],
}


# ---------------------------------------------------------------------------
# Core injection mechanism
# ---------------------------------------------------------------------------


def _inject_question(agent: Any, question: str) -> None:
    """
    Inject an interview question into the agent's context.

    Appends a ``SystemEvent`` so that when ``chat_completions()`` is
    called, the question appears as the latest message.
    """
    agent.context.add_interaction(
        SystemEvent(
            event_type="interview_question",
            content=(
                "[POST-GAME INTERVIEW]\n"
                "The game is over. An interviewer is asking you reflective "
                "questions about your experience. Answer honestly and "
                "thoroughly based on what you remember from the game.\n\n"
                f"Question: {question}"
            ),
            round=agent._current_round(),
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def interview_agent(
    agent: Any,
    dimension: str,
    question: str,
    temperature: float = 0.3,
    question_index: int = 0,
) -> InterviewExchange:
    """
    Ask one question and return the exchange.

    Parameters
    ----------
    agent
        A live ``BaseAgent`` subclass (``OpenAIAgent``, ``OpenRouterAgent``)
        from ``game.agents[i].agent``.
    """
    _inject_question(agent, question)
    response = agent.chat_completions(temperature=temperature)
    return InterviewExchange(
        agent_name=agent.name,
        dimension=dimension,
        question=question,
        response=response,
        question_index=question_index,
    )


def run_full_interview(
    agent: Any,
    dimensions: list[str] | None = None,
    temperature: float = 0.3,
) -> list[InterviewExchange]:
    """Run the full interview battery for one agent."""
    if dimensions is None:
        dimensions = list(INTERVIEW_QUESTIONS.keys())

    exchanges: list[InterviewExchange] = []
    q_idx = 0
    for dim in dimensions:
        for question in INTERVIEW_QUESTIONS.get(dim, []):
            ex = interview_agent(agent, dim, question, temperature, q_idx)
            exchanges.append(ex)
            q_idx += 1
    return exchanges


def interview_all_agents(
    game: Any,
    dimensions: list[str] | None = None,
    temperature: float = 0.3,
) -> dict[str, list[InterviewExchange]]:
    """
    Interview every agent in a completed game.

    Returns ``{agent_name: [InterviewExchange, ...]}``.
    """
    results: dict[str, list[InterviewExchange]] = {}
    for adapter in game.agents:
        exchanges = run_full_interview(adapter.agent, dimensions, temperature)
        results[adapter.agent.name] = exchanges
    return results
