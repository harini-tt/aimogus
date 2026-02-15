"""
Phase-specific instructions for the Among Us game.

These constants are used by the game engine to provide contextual instructions
to agents during different game phases.
"""

# ---------------------------------------------------------------------------
# Phase instructions — injected into location info messages
# ---------------------------------------------------------------------------

TASK_PHASE_INSTRUCTION = (
    "It is the task phase. Move around the map, complete your tasks, "
    "and pay attention to what other players are doing. "
    "If you are the Impostor, try to eliminate Crewmates without being seen. "
    "You may report a dead body or call an emergency meeting if needed."
)

MEETING_PHASE_INSTRUCTION = (
    "An emergency meeting has been called. All players are gathered in the Cafeteria. "
    "Discuss who you think the Impostor is. Share your observations, "
    "defend yourself if accused, and vote to eject a suspect when the discussion ends."
)
