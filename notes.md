# Treehacks Amogus - Codebase Notes

## Overview

A multi-agent Among Us simulation where LLM-powered agents play a full game of Among Us against each other. Agents navigate a spatial map, complete (or fake) tasks, kill, report bodies, discuss in meetings, and vote — all driven by OpenAI chat completions. Built for TreeHacks.

**Tech stack:** Python 3.13+, OpenAI API (gpt-5.2 default, gpt-4o in adapter), NetworkX, Tkinter, LangChain, Pydantic, NumPy

---

## Directory Structure

```
treehacks-amogus/
├── run.py                        # Entry point — CLI + game loop
├── pyproject.toml                # Dependencies (uv-managed)
├── uv.lock                       # Dependency lock file
├── .env                          # OPENAI_API_KEY
├── .gitignore                    # .env, .venv, __pycache__, *.pyc
│
├── agents/                       # LLM agent framework
│   ├── __init__.py               # Public API re-exports
│   ├── base_agent.py             # Abstract base agent class
│   ├── openai_agent.py           # OpenAI Chat Completions agent
│   ├── env_adapter.py            # Bridge between agents + game env
│   └── models.py                 # Pydantic models (Role, Action, Context, etc.)
│
├── envs/                         # Game environment / engine
│   ├── game.py                   # Main game loop (AmongUs class + MessageSystem)
│   ├── player.py                 # Player, Crewmate, Impostor classes
│   ├── action.py                 # All action types (MoveTo, Kill, Vote, Speak, etc.)
│   ├── task.py                   # Task system + assignment logic
│   ├── map.py                    # NetworkX graph-based spaceship map
│   ├── tools.py                  # LangChain tool (GetBestPath) + AgentResponse model
│   └── configs/
│       ├── game_config.py        # 5-player and 7-player game configs
│       ├── map_config.py         # Room data, connections, vent network, UI coords
│       ├── task_config.py        # 20 task types with durations and categories
│       └── agent_config.py       # Placeholder agent config (ALL_LLM)
│
├── prompts/                      # LLM prompt templates
│   ├── __init__.py               # Re-exports all prompts
│   ├── system_prompt.py          # Base system prompt (identity, rules, guidelines)
│   ├── action_prompt.py          # Action selection + observation location prompts
│   ├── phase_prompts.py          # Task phase + meeting phase instructions
│   ├── voting_prompt.py          # Voting prompt template
│   └── meeting_prompt.py         # Meeting discussion prompt template
│
├── UI/
│   └── MapUI.py                  # Tkinter-based map visualization + activity log
│
└── assets/                       # (Expected) blankmap.png for UI background — NOT currently present
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   run.py (Entry)                     │
│  CLI: --players 5|7, --no-ui, --debug/--no-debug    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              AmongUs (envs/game.py)                  │
│  Game loop: initialize → game_step → check_game_over│
│  Phases: "task" ↔ "meeting"                         │
│  MessageSystem: routes observations to players      │
└────────┬────────────────────┬───────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────────┐  ┌────────────────────────────┐
│  Player System   │  │  Agent System              │
│  (envs/player.py)│  │  (agents/)                 │
│                  │  │                            │
│  Player (base)   │  │  BaseAgent (abstract)      │
│  ├─ Crewmate     │  │  └─ OpenAIAgent (concrete) │
│  └─ Impostor     │  │                            │
│                  │  │  EnvAgentAdapter (bridge)   │
│  Tracks: location│  │  ├─ wraps OpenAIAgent       │
│  history, tasks  │  │  └─ wraps Player            │
│  observations    │  │                            │
└──────────────────┘  └────────────────────────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌──────────────────┐         ┌──────────────┐
         │  Map (NetworkX)  │         │  Prompts     │
         │  14 rooms        │         │  system      │
         │  25 corridors    │         │  action      │
         │  9 vent links    │         │  meeting     │
         └──────────────────┘         │  voting      │
                  │                   │  phase       │
                  ▼                   └──────────────┘
         ┌──────────────────┐
         │  UI (Tkinter)    │
         │  MapUI + log     │
         └──────────────────┘
```

---

## How It Works (Data Flow)

### 1. Initialization (`run.py` → `AmongUs.initialize_game()`)

1. `run.py` parses CLI args, picks game config (5 or 7 players), optionally creates `MapUI`.
2. `AmongUs.__init__()` creates the `Map` (NetworkX graph) and `MessageSystem`.
3. `initialize_game()`:
   - Creates `Player` objects (randomly assigns Crewmate/Impostor identities and colors).
   - All players start in Cafeteria.
   - `TaskAssignment` distributes tasks: common tasks are shared, short/long tasks are unique per player. Impostors only get common tasks (to fake).
   - Creates `EnvAgentAdapter` for each player (wraps an `OpenAIAgent` + the env `Player`).

### 2. Game Loop (`run_game()`)

```
while not game_over:
    game_step()          # runs current phase
    check_game_over()    # evaluate win conditions
```

### 3. Task Phase (`task_phase_step()`)

For each agent (sequentially):
1. `check_actions()` — determines what actions are legally available for every player based on game state.
2. `agent_step(agent)`:
   - Skip if dead.
   - Decrement impostor kill cooldown.
   - `agent.choose_action()`:
     - **Sync**: Converts new env observations/actions into Pydantic models and pushes to agent context.
     - **Build prompt**: Assembles action prompt from player's location info, observation history, action history, tasks, and available actions.
     - **LLM call**: Calls OpenAI chat completions (temp=0.7, max_tokens=256).
     - **Parse**: Fuzzy-matches LLM response to available actions using `SequenceMatcher`.
     - **Special cases**: If action is `Speak`, generates speech via second LLM call. If `ViewMonitor`, picks observation room via another LLM call.
   - Execute action on the environment.
   - Update map, route observation messages to nearby players.
   - If a meeting is triggered, break out of the task phase loop.

### 4. Meeting Phase (`meeting_phase()`)

1. Move all players to Cafeteria.
2. **Discussion rounds** (default 3): Each agent takes a `Speak` action — the LLM generates 1-3 sentences of strategic dialogue.
3. **Voting round**: Each alive agent picks a player to vote for (via `Vote` action). LLM fuzzy-matches to a valid vote target.
4. **Voteout**: Player with most votes is ejected. Ties = no ejection. Result is logged as an important event.
5. Phase resets to "task".

### 5. Win Conditions (`check_game_over()`)

| Code | Condition | Winner |
|------|-----------|--------|
| 1 | Impostors >= Crewmates (alive) | Impostors |
| 2 | All impostors eliminated | Crewmates |
| 3 | All tasks completed (100%) | Crewmates |
| 4 | `max_timesteps` reached | Impostors |
| 0 | None of the above | Game continues |

---

## Key Components — Deep Dive

### `envs/game.py` — AmongUs class (321 lines)

The game engine. Core state:
- `self.players` — list of Player objects
- `self.agents` — list of EnvAgentAdapter objects
- `self.map` — Map (NetworkX graph)
- `self.current_phase` — "task" or "meeting"
- `self.timestep` — increments each `game_step()`
- `self.activity_log` / `self.important_activity_log` — event records
- `self.camera_record` — tracks last action per player (for security cameras)
- `self.votes` — vote tallies during meetings
- `self.button_num` — emergency buttons pressed (capped by config)

**MessageSystem** (nested class):
- `route_location_info_message()` — sends each player info about who's in their current room.
- `route_real_time_message()` — broadcasts action observations to players in the same room as the actor.
- Observation routing is location-based: you only see what happens in your room.

### `envs/player.py` — Player classes (161 lines)

**Player** (base):
- Name format: `"Player N: color"` (e.g., "Player 1: red")
- Maintains: `observation_history` (list of message strings), `action_history` (list of record dicts), `location_info` (current room state string), `tasks`, `available_actions`, `is_alive`, `reported_death`.
- Prompt builders: `location_info_prompt()`, `observation_history_prompt(recent_num=4)`, `action_history_prompt(recent_num=4)`, `tasks_prompt()`, `available_actions_prompt()`.

**Crewmate** — adds `CREWMATE_ACTIONS = [CompleteTask]`
**Impostor** — adds `IMPOSTER_ACTIONS = [Sabotage, Vent, Kill, CompleteFakeTask]`, plus `kill_cooldown`.

12 possible player colors: red, blue, green, pink, orange, yellow, black, white, purple, brown, cyan, lime.

### `envs/action.py` — Action system (278 lines)

Each action class has:
- `execute(env, player)` — mutates game state
- `can_execute_actions(env, player)` — static method returning list of legal instances
- `action_text()` — observer-facing description (important: `CompleteTask.action_text()` returns "Seemingly doing task" — observers can't tell real from fake!)

**Action types:**

| Class | Name | Phase | Who | Description |
|-------|------|-------|-----|-------------|
| `MoveTo` | MOVE | task | all | Move to adjacent room (corridor) |
| `Vent` | VENT | task | impostor | Move via vent network |
| `Kill` | KILL | task | impostor | Kill crewmate in same room (cooldown-gated) |
| `CallMeeting` | CALL MEETING | task | all | Emergency button (Cafeteria) or report body |
| `CompleteTask` | COMPLETE TASK | task | crewmate | Work on a task at its location |
| `CompleteFakeTask` | COMPLETE FAKE TASK | task | impostor | Fake doing a task (looks identical to observers) |
| `ViewMonitor` | ViewMonitor | task | all (Security only) | Observe a chosen room via cameras |
| `Speak` | SPEAK | meeting | all | Say something during discussion |
| `Vote` | VOTE | meeting | all | Vote for a player during voting round |
| `Sabotage` | SABOTAGE | task | impostor | Placeholder — not implemented |

**Important gameplay details:**
- `CallMeeting` can be triggered by emergency button (Cafeteria, limited uses) or by finding an unreported dead body.
- `Kill` only targets players with a different identity in the same room — can't kill fellow impostors.
- `CompleteTask` and `CompleteFakeTask` both display as "Seemingly doing task" to observers.
- `ViewMonitor` only available in Security room. Shows who is in the chosen room and what they're doing.
- `Speak` is unavailable during the voting round (last discussion round).
- `Vote` is only available when `discussion_rounds_left == 0`.

**Action lists:**
- `COMMON_ACTIONS = [MoveTo, CallMeeting, Vote, Speak, ViewMonitor]`
- `CREWMATE_ACTIONS = [CompleteTask]`
- `IMPOSTER_ACTIONS = [Sabotage, Vent, Kill, CompleteFakeTask]`

### `envs/task.py` — Task system (104 lines)

**Task**:
- Has `name`, `duration` (turns to complete), `task_type` (short/long/common), `location`, `is_completed`.
- `do_task()` decrements duration by 1. Task completes when duration reaches 0.
- `find_path()` uses NetworkX shortest path. Impostors ignore edge weights (can use vents), crewmates use weighted paths (corridor-only).

**TaskAssignment**:
- Builds all tasks from the map's room data + task_config.
- Assigns to players: common tasks are shared (deep-copied per player), short/long tasks are unique.
- Impostors only get common tasks (for faking).
- `check_task_completion()` only counts tasks of alive players. Returns float 0.0-1.0.

**Task types (20 total):**
- Short (duration 1): Download Data, Accept Diverted Power, Chart Course, Stabilize Steering, Clean O2 Filter, Prime Shields, Upload Data, Calibrate Distributor, Divert Power, Unlock Manifolds, Submit Scan
- Long (duration 2): Empty Garbage, Clear Asteroids, Empty Chute, Align Engine Output, Fuel Engines, Start Reactor, Inspect Sample
- Common (duration 1): Fix Wiring, Swipe Card

### `envs/map.py` — Map (90 lines)

NetworkX undirected graph:
- **14 rooms** (nodes): Cafeteria, Weapons, Navigation, O2, Shields, Communications, Storage, Admin, Electrical, Lower Engine, Security, Reactor, Upper Engine, Medbay.
- **25 corridor connections** (edges, weight=1).
- **9 vent connections** (edges, weight=100 — high weight so crewmate pathfinding avoids them).
- Node attributes: `tasks`, `vent`, `special_actions`, `players`.
- `get_players_in_room(room, include_new_deaths=False)` — optionally includes unreported dead bodies.
- `reset()` clears all player lists; `add_player()` places player in their room.

### `agents/base_agent.py` — BaseAgent (160 lines)

Abstract base class. Manages:
- `name`, `role`, `is_alive`, `assigned_tasks`, `completed_tasks`
- `context: AgentContext` — holds game instructions + chronological interaction log

Abstract methods:
- `chat_completions(**kwargs) -> str` — call the LLM
- `format_context() -> list[dict]` — build message payload

Concrete helpers:
- `die()`, `complete_task(task)`, `add_interaction(interaction)`, `reset()`
- `_current_round()` — infers round from latest interaction's timestamp

### `agents/openai_agent.py` — OpenAIAgent (208 lines)

Concrete agent using OpenAI Chat Completions API.

- Default model: `"gpt-5.2"` (set in constructor)
- `format_context()`:
  1. System message from `SYSTEM_PROMPT` template (name, role, role_instructions, player_list, assigned_tasks).
  2. Interaction history — each interaction rendered chronologically.
- Interaction rendering:
  - `EmergencyMeeting` → header + conversation turns (own messages as "assistant", others as "user")
  - `SystemEvent` → single "user" message with `[SYSTEM — {event_type}]` prefix
  - `Action` → first-person `[YOUR ACTION]` or third-person `[OBSERVED]` messages

**Role-specific instructions** injected into system prompt:
- Crewmate: complete tasks, find impostor, share observations, vote wisely.
- Impostor: eliminate crewmates, lie convincingly, build fake alibis, deflect suspicion.

### `agents/env_adapter.py` — EnvAgentAdapter (298 lines)

The critical bridge between the agent framework and the game environment.

- Default model in adapter: `"gpt-4o"` (note: different from OpenAIAgent's default of "gpt-5.2" — the adapter overrides).
- `choose_action()`:
  1. `_sync_env_to_agent()` — converts new `observation_history` entries to `SystemEvent`s, new `action_history` entries to `PydanticAction`s.
  2. `_build_action_prompt()` — assembles the ACTION_PROMPT from player state.
  3. Injects prompt as a `SystemEvent` interaction.
  4. Calls `chat_completions(temperature=0.7, max_tokens=256)`.
  5. `_parse_action_response()` — fuzzy matches LLM text to available env actions.
  6. If `Speak` action: generates speech via `_generate_speech()` (may use a second LLM call).
- `choose_observation_location()` — for security camera usage, asks LLM to pick a room (temp=0.3).
- `_best_match()` — fuzzy matching using `SequenceMatcher` + substring matching.

### `agents/models.py` — Pydantic models (156 lines)

**Enums:**
- `Role`: CREWMATE, IMPOSTOR
- `ActionType`: KILL, VOTE, REPORT_BODY, CALL_MEETING, COMPLETE_TASK, FAKE_TASK, MOVE, VENT, SPEAK, VIEW_MONITOR, SABOTAGE

**Models:**
- `Message` — speaker, content, timestamp
- `EmergencyMeeting` — round, called_by, messages[], body_reported
- `SystemEvent` — event_type, content, data, round
- `Action` — action_type, actor, target, metadata, round
- `Interaction` = discriminated union of the above three (on `interaction_type` field)
- `AgentContext` — game_instructions + interactions list, with helpers: `add_interaction()`, `get_meetings()`, `get_system_events()`, `get_actions()`, `get_by_round()`

### `envs/tools.py` — LangChain tools (47 lines)

- `GetBestPath` — LangChain `BaseTool` for pathfinding. Takes from/to location + identity. Impostors get unweighted shortest path (can use vents), crewmates get weighted (corridor-only).
- `AgentResponse` — Pydantic model with `condensed_memory`, `thinking_process`, `action` fields. Has a validator for valid action types. Appears to be from an earlier design — not currently used in the main flow.
- `AgentResponseOutputParser` — LangChain parser wrapping `AgentResponse`. Also appears unused currently.

### Prompts

**`SYSTEM_PROMPT`** — Sets up the agent's identity, explains Among Us rules (free roam + meeting phases, win conditions), provides role-specific instructions, lists players and tasks, and gives behavioral guidelines.

**`ACTION_PROMPT`** — Turn-by-turn prompt showing: current situation (phase + location info), observations, recent actions, tasks with pathfinding, available actions. Instructs LLM to respond with ONLY the action text.

**`OBSERVATION_LOCATION_PROMPT`** — For security cameras: lists rooms, asks for just the room name.

**`MEETING_PROMPT`** — Meeting discussion: provides context, alive players, asks for 1-3 sentence strategic speech.

**`VOTING_PROMPT`** — Post-discussion: provides meeting summary, alive players, asks for a name or "skip".

**`TASK_PHASE_INSTRUCTION`** / **`MEETING_PHASE_INSTRUCTION`** — Short contextual instructions injected into location info messages.

### `UI/MapUI.py` — Tkinter UI (165 lines)

- Renders the Among Us map using a background PNG image (`assets/blankmap.png` — **currently missing from repo**).
- Draws room polygons (white fill, black outline) using coordinates from `map_config.py`.
- Players shown as colored circles in their rooms; dead players get a red X overlay.
- Task progress bar at bottom (green fill proportional to completion, turns red if impostors win).
- Scrolling activity log shows all game events.
- Debug mode adds 1-second delay between updates for visibility.

---

## Game Configs

### 5-Player Game (`FIVE_MEMBER_GAME`)
| Parameter | Value |
|-----------|-------|
| Players | 5 |
| Impostors | 1 |
| Common tasks | 1 |
| Short tasks | 1 |
| Long tasks | 0 |
| Discussion rounds | 3 |
| Max emergency buttons | 2 |
| Kill cooldown | 3 timesteps |
| Max timesteps | 50 |

### 7-Player Game (`SEVEN_MEMBER_GAME`)
| Parameter | Value |
|-----------|-------|
| Players | 7 |
| Impostors | 2 |
| Common tasks | 1 |
| Short tasks | 1 |
| Long tasks | 1 |
| Discussion rounds | 3 |
| Max emergency buttons | 2 |
| Kill cooldown | 3 timesteps |
| Max timesteps | 50 |

---

## Map Layout

**14 rooms** connected by 25 corridors and 9 vent passages.

### Corridor Connections
```
Cafeteria ── Weapons ── Navigation ── Shields ── Communications
    │            │           │           │            │
    ├── Admin    └── O2 ─────┘           └── Storage ─┘
    │               │                         │
    ├── Upper Eng.  └── Admin ── Electrical ──┘
    │                              │
    └── Medbay                Lower Engine ── Security ── Reactor
                                   │              │         │
                               Upper Engine ──────┘─────────┘
```

### Vent Network (Impostor-only fast travel)
- Reactor ↔ Lower Engine ↔ Upper Engine ↔ Reactor (triangle)
- Electrical ↔ Security ↔ Medbay (triangle)
- Navigation ↔ Shields, Navigation ↔ Weapons
- Admin ↔ Cafeteria

### Special Room Features
- **Cafeteria**: Emergency button
- **Security**: Security cameras (ViewMonitor)
- **Admin**: Admin map
- **Medbay**: Medbay scan
- **O2**: Oxygen depleted sabotage
- **Communications**: Comms sabotage
- **Electrical**: Fix lights sabotage
- **Reactor**: Reactor meltdown sabotage

---

## Observation & Information Model

Players have **limited information** based on location:

1. **Location info**: Each turn, players receive info about who is in their current room (including unreported dead bodies).
2. **Action observations**: When another player takes an action in the same room, you observe it via `MessageSystem.route_real_time_message()`. You also see actions from your destination room after moving.
3. **Security cameras**: If in Security room, can observe any room via ViewMonitor action. Shows who's there and what they're doing (entering/leaving/actions).
4. **Action history**: Limited to last 4 actions (configurable via `recent_num`).
5. **Observation history**: Limited to last 4 observations.
6. **Task visibility**: Both `CompleteTask` and `CompleteFakeTask` appear as "Seemingly doing task" to observers — impostors can't be distinguished by task behavior alone.

---

## Potential Issues / TODOs Found in Code

1. **Missing `assets/blankmap.png`**: The UI requires this file but it's not in the repo. Running with UI will raise an exception.
2. **Model mismatch**: `OpenAIAgent` defaults to `"gpt-5.2"` but `EnvAgentAdapter` passes `"gpt-4o"` — the adapter's value wins.
3. **`Sabotage` action**: `execute()` is empty — sabotage is not implemented.
4. **`CompleteFakeTask.execute()`**: Currently calls `self.task.do_task()` which actually completes the task. Has a TODO noting this should be a fake task instance that doesn't actually progress.
5. **`Speak.execute()`**: Has a TODO — currently a no-op (speech content is handled separately via `provide_message()`).
6. **`Vote` missing `@staticmethod`**: `can_execute_actions` method lacks the decorator (works but inconsistent).
7. **No "skip" vote option**: The `Vote` action only allows voting for other alive players — no skip/abstain option despite the voting prompt mentioning it.
8. **`Spaceship` class in `map.py`**: Largely unused wrapper around Map.
9. **`GetBestPath` tool and `AgentResponse`**: Appear to be from an earlier LangChain-based agent design. `GetBestPath` is instantiated but never called in the current flow. `AgentResponse` and its output parser are unused.
10. **`interviewer` parameter**: `AmongUs.__init__()` accepts an `interviewer` parameter and calls `interviewer.auto_question()` during gameplay and at game end. This class is not defined anywhere in the codebase.
11. **Dead tasks not counted**: `check_task_completion()` only counts alive players' tasks — killing a crewmate effectively reduces the total tasks needed to win.
12. **Agent action parsing fragility**: Relies on fuzzy string matching between LLM output and action strings. Could fail with unexpected model outputs.

---

## Dependencies (`pyproject.toml`)

```
langchain>=1.2.10
langgraph>=1.0.8
langgraph-prebuilt>=1.0.7
networkx>=3.6.1
numpy>=2.4.2
openai>=2.21.0
pydantic>=2.12.5
python-dotenv>=1.2.1
pyyaml>=6.0.3
requests>=2.32.5
tqdm>=4.67.3
```

Note: `langgraph`, `langgraph-prebuilt`, `pyyaml`, `requests`, `tqdm` are listed as dependencies but don't appear to be actively used in the current codebase (likely holdovers from earlier design or planned future use).

---

## Running the Game

```bash
# 5-player game with UI
python run.py

# 7-player game with UI
python run.py --players 7

# Headless (no UI)
python run.py --no-ui

# Without debug delays
python run.py --no-debug
```

Requires `OPENAI_API_KEY` in `.env` file and `assets/blankmap.png` for UI mode.
