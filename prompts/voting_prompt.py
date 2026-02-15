VOTING_PROMPT = """\
The discussion phase is over. It's time to vote.

Here is a summary of the meeting discussion:

{meeting_summary}

Players still alive (eligible targets): {alive_players}

Based on the discussion and everything you know, decide who to vote for.
You may also choose to skip the vote if you are unsure.

Respond with ONLY the name of the player you want to eject, or "skip" to
skip the vote. Do not explain your reasoning — just the name or "skip".\
"""
