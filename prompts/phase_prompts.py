"""
Phase-specific instructions for the Among Us game.

These constants are used by the game engine to provide contextual instructions
to agents during different game phases.

Kept short because the system prompt already explains the full game rules
and role-specific objectives.
"""

# ---------------------------------------------------------------------------
# Phase instructions — injected into location info messages
# ---------------------------------------------------------------------------

TASK_PHASE_INSTRUCTION = (
    "Task phase: information comes from who is in the same room and where bodies appear. "
    "Kill cooldown (for impostors) ticks here. Security monitor can show current occupants of a chosen room."
)

MEETING_PHASE_INSTRUCTION = (
    "Meeting phase: only living players speak and vote; credibility improves when mentioning room, time, and who was present."
)
