MEETING_PROMPT = """\
An emergency meeting has been called.

{meeting_context}

Players still alive: {alive_players}

Contribute to the discussion. Share observations, ask questions, or defend yourself.

Respond as JSON with:
- "reasoning": your private analysis of the situation
- "message": your public speech to the group (1-3 sentences, natural and in-character)

Output ONLY valid JSON.\
"""
