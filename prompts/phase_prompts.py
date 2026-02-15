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

TASK_PHASE_INSTRUCTION = "Complete tasks, watch for suspicious behaviour, and report bodies."

MEETING_PHASE_INSTRUCTION = "Discuss, share observations, and vote to eject a suspect."
