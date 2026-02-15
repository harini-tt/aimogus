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

Choose exactly ONE action from the list above.

Think step-by-step about your strategy (this reasoning is private and other players cannot see it).
Then, on the LAST line of your response, write your chosen action prefixed with "ACTION: " exactly as listed above.

Example:
I should head to Admin to complete my task. Nobody is around so it's safe.
ACTION: MOVE from Cafeteria to Admin\
"""

OBSERVATION_LOCATION_PROMPT = """\
You are viewing the security monitor. Choose a room to observe.

Available rooms:
{rooms}

Think about which room to observe (your reasoning is private).
Then, on the LAST line write the room name prefixed with "ROOM: " exactly as listed above.\
"""
