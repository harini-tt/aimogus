"""
Parser for clean Among Us game log files.

Handles logs with structured [TAG] format lines:
[INIT], [CONFIG], [TURN], [ACTION], [DISCUSSION], [VOTEOUT], [GAME_OVER]
"""

import re
import ast
from pathlib import Path


# --- Regex patterns for each line type ---

RE_INIT_PLAYER = re.compile(
    r"\[INIT\] (Player \d+): (\w+) \| (\w+) \| tasks: (.+)"
)
RE_CONFIG = re.compile(r"\[CONFIG\] (.+)")
RE_TURN = re.compile(r"\[TURN\] T(\d+) \| phase: (\w+)")
RE_ACTION_TASK = re.compile(
    r"\[ACTION\] T(\d+) \[task\] (Player \d+: \w+): (.+)"
)
RE_ACTION_MEETING = re.compile(
    r"\[ACTION\] T(\d+) \[meeting R(\d+)\] (Player \d+: \w+): (.+)"
)
RE_DISCUSSION = re.compile(r"\[DISCUSSION\] round (\d+)/(\d+)")
RE_VOTEOUT = re.compile(r"\[VOTEOUT\] (.+)")
RE_GAME_OVER = re.compile(r"\[GAME_OVER\] (.+)")

# --- Action-specific patterns (applied to the action text after player name) ---

ACTION_PATTERNS = [
    ("move", re.compile(r"MOVE from (.+) to (.+)")),
    ("complete_task", re.compile(r"COMPLETE TASK - (.+)")),
    ("fake_task", re.compile(r"COMPLETE FAKE TASK - (.+)")),
    ("kill", re.compile(r"KILL (Player \d+: \w+) \| Location: (.+), Witness: \[(.+)\]")),
    ("vent", re.compile(r"VENT from (.+) to (.+)")),
    ("speak", re.compile(r"SPEAK: (.+)", re.DOTALL)),
    ("vote", re.compile(r"VOTE (Player \d+: \w+)")),
    ("skip_vote", re.compile(r"SKIP VOTE")),
    ("report_body", re.compile(r"REPORT DEAD BODY at (.+)")),
    ("call_meeting", re.compile(r"CALL MEETING using the emergency button at (.+)")),
    ("view_monitor", re.compile(r"VIEW MONITOR")),
]


def parse_tasks(task_str):
    """Parse task string like 'common: Swipe Card (Admin), short: Clean O2 Filter (O2)'"""
    tasks = []
    # Split by task type prefixes
    parts = re.split(r"(?:^|,\s*)(common|short|long): ", task_str)
    # parts will be like ['', 'common', 'Swipe Card (Admin)', 'short', 'Clean O2 Filter (O2)', ...]
    i = 1
    while i < len(parts) - 1:
        task_type = parts[i]
        task_info = parts[i + 1].strip()
        # Extract task name and location from "Task Name (Location)"
        m = re.match(r"(.+?) \(([^)]+)\)", task_info)
        if m:
            tasks.append({
                "type": task_type,
                "name": m.group(1),
                "location": m.group(2),
            })
        i += 2
    return tasks


def parse_action_text(text):
    """Parse action text into a structured event dict."""
    for action_type, pattern in ACTION_PATTERNS:
        m = pattern.match(text)
        if not m:
            continue

        if action_type == "move":
            return {"type": "move", "from": m.group(1), "to": m.group(2)}
        elif action_type == "complete_task":
            return {"type": "complete_task", "task": m.group(1)}
        elif action_type == "fake_task":
            return {"type": "fake_task", "task": m.group(1)}
        elif action_type == "kill":
            witnesses = [w.strip().strip("'\"") for w in m.group(3).split(",")]
            return {
                "type": "kill",
                "victim": m.group(1),
                "location": m.group(2),
                "witnesses": witnesses,
            }
        elif action_type == "vent":
            return {"type": "vent", "from": m.group(1), "to": m.group(2)}
        elif action_type == "speak":
            return {"type": "speak", "message": m.group(1)}
        elif action_type == "vote":
            return {"type": "vote", "target": m.group(1)}
        elif action_type == "skip_vote":
            return {"type": "skip_vote"}
        elif action_type == "report_body":
            return {"type": "report_body", "location": m.group(1)}
        elif action_type == "call_meeting":
            return {"type": "call_meeting", "location": m.group(1)}
        elif action_type == "view_monitor":
            return {"type": "view_monitor"}

    return {"type": "unknown", "raw": text}


def parse_game(filepath):
    """Parse a clean-format game log file into structured JSON data."""
    lines = Path(filepath).read_text().splitlines()

    players = []
    config = {}
    turns = []
    game_result = None
    current_turn = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip artifact lines
        if line.startswith("[YOUR ACTION"):
            continue
        # Skip duplicate [VOTE] lines (info is already in [ACTION] vote lines)
        if line.startswith("[VOTE]"):
            continue

        # --- Player init ---
        m = RE_INIT_PLAYER.match(line)
        if m:
            player_name = f"{m.group(1)}: {m.group(2)}"  # "Player 1: red"
            players.append({
                "id": int(m.group(1).split()[-1]),
                "name": player_name,
                "color": m.group(2),
                "role": m.group(3),
                "tasks": parse_tasks(m.group(4)),
            })
            continue

        # --- Config ---
        m = RE_CONFIG.match(line)
        if m:
            config = ast.literal_eval(m.group(1))
            continue

        # --- Turn marker ---
        m = RE_TURN.match(line)
        if m:
            # Save previous turn
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "turnNumber": int(m.group(1)),
                "phase": m.group(2),
                "events": [],
            }
            if m.group(2) == "meeting":
                current_turn["discussion"] = []
                current_turn["votes"] = []
                current_turn["voteResult"] = None
            continue

        # --- Task phase action ---
        m = RE_ACTION_TASK.match(line)
        if m and current_turn is not None:
            event = parse_action_text(m.group(3))
            event["player"] = m.group(2)
            event["turnNumber"] = int(m.group(1))
            current_turn["events"].append(event)
            continue

        # --- Meeting phase action ---
        m = RE_ACTION_MEETING.match(line)
        if m and current_turn is not None:
            round_num = int(m.group(2))
            player = m.group(3)
            event = parse_action_text(m.group(4))
            event["player"] = player
            event["round"] = round_num
            event["turnNumber"] = int(m.group(1))

            if event["type"] == "speak":
                # Add to discussion rounds
                discussion = current_turn.get("discussion", [])
                # Find or create the round entry
                round_entry = None
                for r in discussion:
                    if r["round"] == round_num:
                        round_entry = r
                        break
                if round_entry is None:
                    round_entry = {"round": round_num, "speeches": []}
                    discussion.append(round_entry)
                round_entry["speeches"].append({
                    "player": player,
                    "message": event["message"],
                })
                current_turn["discussion"] = discussion
            elif event["type"] in ("vote", "skip_vote"):
                votes = current_turn.get("votes", [])
                vote_entry = {"voter": player}
                if event["type"] == "vote":
                    vote_entry["target"] = event["target"]
                else:
                    vote_entry["target"] = "skip"
                votes.append(vote_entry)
                current_turn["votes"] = votes

            current_turn["events"].append(event)
            continue

        # --- Discussion round marker ---
        m = RE_DISCUSSION.match(line)
        if m:
            # Informational only, rounds are tracked via action lines
            continue

        # --- Voteout ---
        m = RE_VOTEOUT.match(line)
        if m and current_turn is not None:
            voteout_text = m.group(1)
            # Extract player name: "Player 7: purple was voted out"
            voted_out = re.match(r"(.+?) was voted out", voteout_text)
            if voted_out:
                current_turn["voteResult"] = {"ejected": voted_out.group(1)}
            continue

        # --- Game over ---
        m = RE_GAME_OVER.match(line)
        if m:
            game_result = m.group(1)
            continue

    # Append last turn
    if current_turn is not None:
        turns.append(current_turn)

    # Ensure the replay always has an explicit zeroth turn representing
    # the initial game state (all players alive in Cafeteria).
    if not turns or turns[0].get("turnNumber", 0) != 0:
        turns.insert(0, {
            "turnNumber": 0,
            "phase": "task",
            "events": [],
        })

    return {
        "players": players,
        "config": config,
        "turns": turns,
        "gameResult": game_result,
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <log_file>")
        sys.exit(1)

    result = parse_game(sys.argv[1])
    print(json.dumps(result, indent=2))
