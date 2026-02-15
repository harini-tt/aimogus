"""
Prompt templates for the Among Us AI agent framework.

Each prompt is a module-level string constant using ``str.format`` placeholders.

Usage::

    from prompts import SYSTEM_PROMPT, MEETING_PROMPT

    rendered = SYSTEM_PROMPT.format(name="Red", role="IMPOSTOR", ...)
"""

from prompts.system_prompt import SYSTEM_PROMPT
from prompts.meeting_prompt import MEETING_PROMPT
from prompts.voting_prompt import VOTING_PROMPT
from prompts.action_prompt import ACTION_PROMPT, OBSERVATION_LOCATION_PROMPT
from prompts.phase_prompts import (
    TASK_PHASE_INSTRUCTION,
    MEETING_PHASE_INSTRUCTION,
)

__all__ = [
    "SYSTEM_PROMPT",
    "MEETING_PROMPT",
    "VOTING_PROMPT",
    "ACTION_PROMPT",
    "OBSERVATION_LOCATION_PROMPT",
    "TASK_PHASE_INSTRUCTION",
    "MEETING_PHASE_INSTRUCTION",
]
