VOTING_PROMPT = """\
The discussion phase is over. It's time to vote.

Meeting summary:

{meeting_summary}

Players still alive (eligible targets): {alive_players}

Facts: only living players can be voted; skipping keeps everyone; ejection happens only if one target has the highest votes above skip; who-voted-for-whom is not revealed.\
"""
