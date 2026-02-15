VOTING_PROMPT = """\
The discussion phase is over. It's time to vote.

Meeting summary:

{meeting_summary}

Players still alive (eligible targets): {alive_players}

Decide who to vote for, or skip if unsure.

Respond as JSON with:
- "reasoning": your private analysis of who is most suspicious and why
- "vote": the player name exactly as listed above (e.g. "Player 3: green"), or "skip"

Output ONLY valid JSON.\
"""
