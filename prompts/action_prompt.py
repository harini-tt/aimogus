"""
Action selection prompt — presented to the LLM each turn so it can choose
from the currently available actions given the full environment state.
"""

ACTION_PROMPT = """\
== CURRENT SITUATION ==

{phase_info}

{location_info}

== OBSERVATIONS ==

{observation_history}

== YOUR RECENT ACTIONS ==

{action_history}

== TASKS ==

{tasks_info}

== AVAILABLE ACTIONS ==

{available_actions}

Based on the situation above, choose exactly ONE action from the available \
actions list. Respond with ONLY the action text exactly as shown (e.g. \
"MOVE from Cafeteria to Weapons" or "KILL Player 2: blue"). Do not explain \
your reasoning — just output the action.\
"""

OBSERVATION_LOCATION_PROMPT = """\
You are viewing the security monitor. Choose a room to observe.

Available rooms:
{rooms}

Respond with ONLY the room name. Do not explain your reasoning.\
"""
