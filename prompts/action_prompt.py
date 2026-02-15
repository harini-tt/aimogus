"""
Action selection prompt — presented to the LLM each turn so it can choose
from the currently available actions given the full environment state.

Observations and action history are already present in the conversation as
individual messages, so they are NOT repeated here.
"""

ACTION_PROMPT = """\
== CURRENT SITUATION ==

{phase_info}

== TASKS ==

{tasks_info}

== AVAILABLE ACTIONS ==

{available_actions}

Choose exactly ONE action from the list above. Respond as JSON with:
- "reasoning": your private strategic thinking
- "action": the action text exactly as listed above

Example: {{"reasoning": "Moving to Admin to do my task and check for suspicious activity.", "action": "MOVE from Cafeteria to Admin"}}

Output ONLY valid JSON.\
"""

OBSERVATION_LOCATION_PROMPT = """\
You are viewing the security monitor. Choose a room to observe.

Available rooms:
{rooms}

Respond as JSON with:
- "reasoning": why you chose this room
- "room": the room name exactly as listed above

Output ONLY valid JSON.\
"""
