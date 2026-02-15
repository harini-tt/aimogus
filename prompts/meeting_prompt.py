MEETING_PROMPT = """\
An emergency meeting has been called.

{meeting_context}

Players still alive: {alive_players}

First, write your private analysis between [PRIVATE] and [PUBLIC] tags (only you can see this). Consider: your room-by-room timeline since the last meeting, who else was present, any bodies seen, and what people have claimed.
Then, after [PUBLIC], write your public speech to the group (1-3 sentences, natural and in-character). Mention room + approximate time + who was present if relevant; noting contradictions can increase credibility.

Example:
[PRIVATE]
I was in Electrical with green at T12; later the body was in Electrical. Green never mentioned leaving. Likely green.
[PUBLIC]
I was in Electrical around T12 with green. The body showed up there right after. Green, where did you go next?\
"""
