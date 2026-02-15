"""
Role-specific instruction blocks injected into the system prompt.

Each entry maps a role string (matching ``Role.value``) to a paragraph
describing the agent's objectives and behavioural guidance for that role.
"""

ROLE_INSTRUCTIONS: dict[str, str] = {
    "crewmate": (
        "You are a CREWMATE. Task bar only moves when living crewmates finish their assigned tasks. "
        "Being seen in multiple rooms over time creates an alibi trail; shared sightings provide mutual evidence. "
        "Unreported bodies imply a recent kill nearby. Other players' stories and vote patterns can be compared for consistency."
    ),
    "impostor": (
        "You are the IMPOSTOR. A kill leaves a body and starts your kill cooldown. "
        "Only impostors can use vents, which connect specific rooms and are visible if someone is present. "
        "Fake tasks look similar to real ones but do not move the task bar. "
        "Being last seen with a player can be incriminating; keep your path and alibi consistent with prior sightings."
    ),
}
