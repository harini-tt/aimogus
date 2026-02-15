VOTING_PROMPT = """\
The discussion phase is over. It's time to vote.

Meeting summary:

{meeting_summary}

Players still alive (eligible targets): {alive_players}

Decide who to vote for, or skip if unsure.

Think about who is most suspicious and why (your reasoning is private).
Then, on the LAST line write your vote prefixed with "VOTE: " — either a player name exactly as listed above (e.g. "Player 3: green"), or "skip".\
"""
