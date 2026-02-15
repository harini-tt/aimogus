MEETING_PROMPT = """\
An emergency meeting has been called.

{meeting_context}

Players still alive: {alive_players}

Contribute to the discussion. Share observations, ask questions, or defend yourself.

First, write your private analysis between [PRIVATE] and [PUBLIC] tags (only you can see this).
Then, after [PUBLIC], write your public speech to the group (1-3 sentences, natural and in-character).

Example:
[PRIVATE]
I saw green vent in Electrical, they are definitely the impostor. I need to bring this up carefully.
[PUBLIC]
I was in Electrical and saw something really suspicious with green. Has anyone else noticed them acting weird?\
"""
