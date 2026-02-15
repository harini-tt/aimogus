import logging

from envs.map import Map, Spaceship
from envs.player import Crewmate, Impostor, PLAYER_COLORS
from agents.env_adapter import EnvAgentAdapter
from agents.openai_agent import OpenAIAgent
from agents.openrouter_agent import OpenRouterAgent
from agents.models import Role
from envs.task import TaskAssignment
from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME
from envs.tools import GetBestPath
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import TASK_PHASE_INSTRUCTION, MEETING_PHASE_INSTRUCTION

logger = logging.getLogger(__name__)

# Map env identity string -> agents.models.Role
_IDENTITY_TO_ROLE: dict[str, Role] = {
    "Crewmate": Role.CREWMATE,
    "Impostor": Role.IMPOSTOR,
}

# Action resolution priority (lower = resolved first).
# Kills must land before reports/meetings, and meetings must trigger
# before ordinary actions so the remaining queue is correctly abandoned.
_ACTION_PRIORITY: dict[str, int] = {
    "KILL": 0,
    "CALL MEETING": 1,
}

def _action_priority(action) -> int:
    """Return the resolution priority for *action* (lower = earlier)."""
    return _ACTION_PRIORITY.get(action.name, 99)

class AmongUs:
    def __init__(self, 
                game_config=SEVEN_MEMBER_GAME, 
                interviewer=None, UI=None,
                model_configs: list[dict] | None = None,
                game_logger: logging.Logger | None = None):
        """
        Parameters
        ----------
        game_config : dict
            Game configuration (player count, impostor count, etc.).
        interviewer : Interviewer | None
            Interviewer object to be used for the game to ask questions.
        UI : MapUI | None
            Optional Tkinter UI.
        model_configs : list[dict] | None
            Per-player model configuration.  Each dict should have:
                ``{"provider": "openai"|"openrouter", "model": "<model-id>"}``
            Length must equal ``game_config["num_players"]``.
            If *None*, every player uses ``OpenAIAgent`` with ``gpt-4o``.
        game_logger : logging.Logger | None
            Optional per-game logger.  If *None*, uses the module-level logger.
            Pass a dedicated logger when running multiple games in parallel so
            that each game writes to its own log file.
        """
        self.logger = game_logger or logger
        self.map = Map()
        self.message_system = MessageSystem(game_config=game_config)
        self.interviewer = interviewer
        self.UI = UI
        # config
        self.game_config = game_config
        self.model_configs = model_configs
        self.all_phases = ["meeting", "task"]
        self.game_config_block = self._format_game_config_block(game_config)
        
        
        
    
    def initialize_game(self):
        # reset game state
        if self.UI:
            self.UI.reset()
        self.players = []
        self.timestep = 0
        self.activity_log = []
        self.important_activity_log = []
        self.camera_record = {}
        self.button_num = 0
        self.task_assignment = TaskAssignment(self.map.ship_map, self.game_config)
        # meeting
        self.discussion_rounds_left = self.game_config["discussion_rounds"]
        self.votes = {}
        self.vote_info_one_round = {}
        self.meeting_caller = None
        
        # game state
        
        self.current_phase = "task"
        self.initialize_players()
        self.initialize_agents()
        
        
        
    def initialize_players(self):
        self.players = []
        num_players = self.game_config["num_players"]
        num_impostors = self.game_config["num_impostors"]
        num_crewmates = num_players - num_impostors
        identities = ["Crewmate"] * num_crewmates + ["Impostor"] * num_impostors
        colors = np.random.choice(PLAYER_COLORS, num_players, replace=False)
        np.random.shuffle(identities)
        for i in range(num_players):
            if identities[i] == "Crewmate":
                player = Crewmate(name=f"Player {i+1}", color=colors[i], location="Cafeteria")
            else:
                player = Impostor(name=f"Player {i+1}", color=colors[i], location="Cafeteria")
            self.players.append(player)
            self.camera_record[player.name] = 'stand quietly and do nothing'
        self.task_assignment.assign_tasks_to_players(self.players)
        self.logger.info("[INIT] Player roster:")
        for p in self.players:
            tasks_str = ", ".join(str(t) for t in p.tasks) if len(p.tasks) > 0 else "none"
            self.logger.info("[INIT] %s | %s | tasks: %s", p.name, p.identity, tasks_str)
        self.update_map()
    
    def initialize_agents(self):
        tools = [GetBestPath(metadata={'network': self.map.ship_map})]
        player_names = [f"{p.name}: {p.color}" for p in self.players]
        impostor_names = [p.name for p in self.players if p.identity == "Impostor"]
        self.agents = []
        for idx, player in enumerate(self.players):
            agent = self._build_agent(player, idx, player_names, impostor_names)
            known_impostors = impostor_names if player.identity == "Impostor" else []
            self.agents.append(
                EnvAgentAdapter(
                    player,
                    tools,
                    agent=agent,
                    game_config_block=self.game_config_block,
                    known_impostors=known_impostors,
                )
            )

    def _build_agent(self, player, idx: int, player_names: list[str], impostor_names: list[str] | None = None):
        """Create the appropriate BaseAgent subclass for *player*."""
        impostor_names = impostor_names or []
        role = _IDENTITY_TO_ROLE.get(player.identity, Role.CREWMATE)
        task_names = [str(t) for t in player.tasks] if len(player.tasks) > 0 else []

        if self.model_configs and idx < len(self.model_configs):
            cfg = self.model_configs[idx]
        else:
            cfg = {"provider": "openai", "model": "gpt-4o"}

        provider = cfg.get("provider", "openai")
        model = cfg.get("model", "gpt-4o")

        if provider == "local":
            from agents.local_agent import LocalModelAgent
            return LocalModelAgent(
                name=player.name,
                role=role,
                assigned_tasks=task_names,
                player_names=player_names,
                game_config_block=self.game_config_block,
                known_impostors=impostor_names if role == Role.IMPOSTOR else [],
                model_instance=cfg.get("model_instance"),
                tokenizer=cfg.get("tokenizer"),
                inference_lock=cfg.get("inference_lock"),
                trajectory=cfg.get("trajectory"),
                game_instructions=cfg.get("game_instructions", ""),
            )
        elif provider == "openrouter":
            return OpenRouterAgent(
                name=player.name,
                role=role,
                assigned_tasks=task_names,
                player_names=player_names,
                model=model,
                game_config_block=self.game_config_block,
                known_impostors=impostor_names if role == Role.IMPOSTOR else [],
            )
        else:
            return OpenAIAgent(
                name=player.name,
                role=role,
                assigned_tasks=task_names,
                player_names=player_names,
                model=model,
                game_config_block=self.game_config_block,
                known_impostors=impostor_names if role == Role.IMPOSTOR else [],
            )
    
    def _format_game_config_block(self, cfg: dict) -> str:
        """Render a concise, factual game-config block for prompts."""
        return (
            f"- Players: {cfg.get('num_players', '?')} "
            f"(Impostors: {cfg.get('num_impostors', '?')})\n"
            f"- Tasks per crewmate: common {cfg.get('num_common_tasks', '?')}, "
            f"short {cfg.get('num_short_tasks', '?')}, long {cfg.get('num_long_tasks', '?')}\n"
            f"- Discussion rounds per meeting: {cfg.get('discussion_rounds', '?')}\n"
            f"- Emergency button uses (shared): {cfg.get('max_num_buttons', '?')}\n"
            f"- Kill cooldown (impostors): {cfg.get('kill_cooldown', '?')} timesteps\n"
            f"- Max timesteps: {cfg.get('max_timesteps', '?')}"
        )
        
    def report_winner(self, winner):
        if winner == 1:
            text = "Impostors win! (Crewmates being outnumbered or tied to impostors))"
        elif winner == 2:
            text = "Crewmates win! (Impostors eliminated)"
        elif winner == 3:
            text = "Crewmates win! (All task completed)"
        elif winner == 4:
            text = "Impostors win! (Time limit reached)"
        self.logger.info("[GAME_OVER] %s", text)
        if self.UI:
            self.UI.report(text)
            self.UI.quit_UI()
        print(text)
        return winner
    
    def check_game_over(self):
        num_impostors = sum([1 for player in self.players if player.identity == "Impostor" and player.is_alive])
        num_crewmates = sum([1 for player in self.players if player.identity == "Crewmate" and player.is_alive])
        if num_impostors >= num_crewmates:
            return 1 # Impostors win
        elif num_impostors == 0:
            return 2 # Crewmates win
        elif self.task_assignment.check_task_completion() == 1.0:
            return 3 # Crewmates win (task completed)
        elif self.timestep >= self.game_config["max_timesteps"]:
            return 4 # Impostors win (time limit)
        return 0 # Game continues
            
    def check_actions(self):
        for player in self.players:
            all_actions = player.get_all_actions()
            available_actions = []
            for action in all_actions:
                action_executables = action.can_execute_actions(self, player)
                available_actions.extend(action_executables)
            player.set_available_actions(available_actions)
    
    def update_map(self):
        self.map.reset()
        for player in self.players:
            self.map.add_player(player)
        self.message_system.route_location_info_message(self)
        if self.UI:
            self.UI.draw_map(self)
        
        
            
    def agent_step(self, agent):
        self.check_actions()
        if not agent.player.is_alive:
            return
        # kill cooldown
        if agent.player.identity == "Impostor" and agent.player.kill_cooldown > 0:
            agent.player.kill_cooldown -= 1
        
        # interview
        if self.interviewer is not None:
            self.interviewer.auto_question(self, agent)
        
        # choose action
        
        action = agent.choose_action()
        observation_location = ''
        if action.name == 'ViewMonitor':
            observation_location = agent.choose_observation_location(self.map.ship_map.nodes)
        self.camera_record[agent.player.name] = action
        if str(action).startswith("KILL"):
            location = agent.player.location
            players = self.map.get_players_in_room(location)
            witness = [player.name for player in players]
            additional_info = f"Location: {location}, Witness: {witness}"
            self.record_activity(agent.player, action, additional_info)
        else:
            self.record_activity(agent.player, action)
        agent.player.make_action(self, action, observation_location)
        self.update_map()
        
    def _agent_choose(self, agent):
        """Choose an action for one agent (thread-safe, no mutations to shared game state).
        
        Called from task_phase_step() inside a ThreadPoolExecutor so that all
        agents' LLM calls run in parallel.  Only reads shared game state and
        writes to the agent's own player/agent objects.
        
        Returns (action, observation_location) or None if the agent is dead.
        """
        if not agent.player.is_alive:
            return None
        # kill cooldown
        if agent.player.identity == "Impostor" and agent.player.kill_cooldown > 0:
            agent.player.kill_cooldown -= 1
        # interview
        if self.interviewer is not None:
            self.interviewer.auto_question(self, agent)
        # choose action
        action = agent.choose_action()
        obs_loc = ''
        if action.name == 'ViewMonitor':
            obs_loc = agent.choose_observation_location(self.map.ship_map.nodes)
        return (action, obs_loc)

    def game_step(self):
        print(f"\n{'='*60}")
        print(f"  TURN {self.timestep} — Phase: {self.current_phase}")
        print(f"{'='*60}")
        self.logger.info("[TURN] T%d | phase: %s", self.timestep, self.current_phase)
        if self.current_phase == "task":
            self.task_phase_step()
        elif self.current_phase == "meeting":
            self.meeting_phase()
        self.timestep += 1
    
    def task_phase_step(self):
        # Phase 1: compute available actions for ALL agents on the shared snapshot
        self.check_actions()

        # Phase 2: all agents choose in parallel via threads
        # Keep the UI responsive while waiting for LLM responses.
        import time
        choices = [None] * len(self.agents)
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_idx = {
                executor.submit(self._agent_choose, agent): idx
                for idx, agent in enumerate(self.agents)
            }
            pending = set(future_to_idx.keys())
            while pending:
                # Pump the Tkinter event loop so the window doesn't freeze
                if self.UI:
                    self.UI.master.update()
                done = {f for f in pending if f.done()}
                for future in done:
                    idx = future_to_idx[future]
                    choices[idx] = future.result()
                    pending.discard(future)
                if pending:
                    time.sleep(0.1)  # avoid busy-spinning

        # Phase 3: apply actions in priority order (kills first, then
        # reports/meetings, then everything else) so the game state is
        # consistent regardless of agent enumeration order.
        paired = [
            (agent, choice)
            for agent, choice in zip(self.agents, choices)
            if choice is not None
        ]
        paired.sort(key=lambda pair: _action_priority(pair[1][0]))

        for agent, (action, obs_loc) in paired:
            self.camera_record[agent.player.name] = action
            if str(action).startswith("KILL"):
                # Execute first, then only record if the kill succeeded.
                # Kill can fail if the target moved away during parallel resolution.
                agent.player.make_action(self, action, obs_loc)
                if getattr(action, "success", True):
                    location = agent.player.location
                    players = self.map.get_players_in_room(location)
                    witness = [player.name for player in players]
                    additional_info = f"Location: {location}, Witness: {witness}"
                    self.record_activity(agent.player, action, additional_info)
            else:
                self.record_activity(agent.player, action)
                agent.player.make_action(self, action, obs_loc)
            self.update_map()  # refresh UI after each action so you see players move
            if self.current_phase == "meeting":
                break
            
    
    def meeting_phase(self):
        # Move all players to the Cafeteria
        for player in self.players:
            player.location = "Cafeteria"

        self.update_map()

        # Shared transcript — collects all speeches across discussion rounds
        meeting_transcript: list[dict] = []
        meeting_round = self.game_config["discussion_rounds"] - self.discussion_rounds_left

        # Discussion — each round is parallelized across all agents
        for rnd in range(self.game_config["discussion_rounds"]):
            print("Discussion round", rnd + 1)
            self.logger.info("[DISCUSSION] round %d/%d", rnd + 1, self.game_config["discussion_rounds"])

            # 1. Compute available actions for everyone
            self.check_actions()

            # 2. All agents choose their speech in parallel
            choices = [None] * len(self.agents)
            with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
                future_to_idx = {
                    executor.submit(self._agent_choose, agent): idx
                    for idx, agent in enumerate(self.agents)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    choices[idx] = future.result()

            # 3. Apply all speech actions sequentially and collect transcript
            for agent, choice in zip(self.agents, choices):
                if choice is None:
                    continue
                action, obs_loc = choice
                self.record_activity(agent.player, action)
                agent.player.make_action(self, action, obs_loc)
                if action.name == "SPEAK":
                    meeting_transcript.append({
                        "speaker": agent.player.name,
                        "content": action.message,
                    })

            self.discussion_rounds_left -= 1
            self.update_map()

        # Inject the full meeting transcript into every agent's context
        # so they can reference it when voting.
        called_by = self.meeting_caller or "Unknown"
        for agent in self.agents:
            agent.inject_meeting_transcript(
                transcript=meeting_transcript,
                called_by=called_by,
                round_num=meeting_round,
            )

        # Voting — also parallelized
        self.check_actions()
        self.vote_info_one_round = {}
        choices = [None] * len(self.agents)
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_idx = {
                executor.submit(self._agent_choose, agent): idx
                for idx, agent in enumerate(self.agents)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                choices[idx] = future.result()

        for agent, choice in zip(self.agents, choices):
            if choice is None:
                continue
            action, obs_loc = choice
            self.record_activity(agent.player, action)
            agent.player.make_action(self, action, obs_loc)

        # Vote out
        self.voteout()
        self.meeting_caller = None  # reset for next meeting
        self.update_map()
        
        
    def voteout(self):
        round_num = self.game_config["discussion_rounds"] - self.discussion_rounds_left

        # Build human-readable vote summary
        vote_info = []
        for voter, vote_target in self.vote_info_one_round.items():
            vote_info.append(f"{voter} voted for {vote_target}")
        print(self.vote_info_one_round)
        print(self.votes)
        for vi in vote_info:
            self.logger.info("[VOTE] %s", vi)

        # Count skip votes and determine outcome
        skip_count = sum(1 for v in self.vote_info_one_round.values() if v == "skip")

        if self.votes:
            max_votes = max(self.votes.values())
            players_with_max_votes = [
                player for player, votes in self.votes.items()
                if votes == max_votes
            ]
        else:
            max_votes = 0
            players_with_max_votes = []

        if len(players_with_max_votes) == 1 and max_votes > skip_count:
            player = players_with_max_votes[0]
            player.is_alive = False
            import_event = {"timestep": self.timestep,
                      "phase": self.current_phase,
                      "round": round_num,
                      "action": f"{player.name} was voted out! Detailed vote info:{vote_info}",
                      "player": "all players"}
            print(f"== {player.name} was voted out ==")
            self.logger.info("[VOTEOUT] %s was voted out", player.name)
        else:
            import_event = {"timestep": self.timestep,
                      "phase": self.current_phase,
                      "round": round_num,
                      "action": f"No one was voted out. Detailed vote info:{vote_info}",
                      "player": "all players"}
            print("== No one was voted out ==")
            self.logger.info("[VOTEOUT] No one was voted out")
        self.important_activity_log.append(import_event)
        self.current_phase = "task"
        self.discussion_rounds_left = self.game_config["discussion_rounds"]
        self.votes = {}

    def check_monitor(self, room):
        players = self.map.get_players_in_room(room)
        return players

    
    def run_game(self):
        self.initialize_game()
        self.logger.info("[CONFIG] %s", self.game_config)
        game_over = self.check_game_over()
        while not game_over: 
            self.game_step()
            game_over = self.check_game_over()
        
        # interview
        if self.interviewer is not None:
            for agent in self.agents:
                self.interviewer.auto_question(self, agent)
        return self.report_winner(game_over)
            
    def record_activity(self, player, action, additional_info=None):
        if self.current_phase == "task":
            record = {"timestep": self.timestep,
                      "phase": self.current_phase, 
                      "action": action, 
                      "player": player}
            if additional_info:
                self.logger.info("[ACTION] T%d [task] %s: %s | %s", self.timestep, player.name, action, additional_info)
            else:
                self.logger.info("[ACTION] T%d [task] %s: %s", self.timestep, player.name, action)
        elif self.current_phase == "meeting":
            round = self.game_config["discussion_rounds"] - self.discussion_rounds_left
            record = {"timestep": self.timestep,
                      "phase": self.current_phase,
                      "round": round, 
                      "action": action, 
                      "player": player}
            self.logger.info("[ACTION] T%d [meeting R%d] %s: %s", self.timestep, round, player.name, action)
        self.activity_log.append(record)
        self.message_system.route_real_time_message(self, record)
        if str(record["action"]).startswith("COMPLETE TASK"):
            imprtant_event = {"timestep": self.timestep,
                      "phase": self.current_phase,
                      "action": str(action), 
                      "player": player.name}
            self.important_activity_log.append(record)
        if str(record["action"]).startswith("KILL"):
            imprtant_event = {"timestep": self.timestep,
                      "phase": self.current_phase,
                      "action": str(action) + "|||" + additional_info, 
                      "player": player.name}
            self.important_activity_log.append(imprtant_event)
                
            

class MessageSystem:
    def __init__(self, game_config):
        self.game_config = game_config
    
    def send_message(self, player, message, info_type):
        player.receive(message, info_type)
    
    def create_action_message(self, record):
        timestep = record["timestep"]
        current_phase = record["phase"]
        player = record["player"]
        action = record["action"]
        if current_phase == "task":
            message = f"Timestep {timestep}: [{current_phase}] {player.name} {action.action_text()}"
        elif current_phase == "meeting":
            round = record["round"]
            message = f"Timestep {timestep}: [{current_phase} phase - round {round}] {player.name} {action.action_text()}"
        return message
    
    def create_location_message(self, record, env):
        if env.current_phase == "task":
            phase_info = "Task phase"
            instruction = TASK_PHASE_INSTRUCTION
        elif env.current_phase == "meeting":
            max_rounds = env.game_config["discussion_rounds"]
            round = max_rounds - env.discussion_rounds_left
            phase_info = f"Meeting phase - Discussion round ({round}/{max_rounds})"
            instruction = MEETING_PHASE_INSTRUCTION
        message = f"Game Time: {env.timestep}/{env.game_config['max_timesteps']}\n"    
        message += f"Current phase: {phase_info}\n"
        message += f"{instruction}\n"
        players_text = ", ".join(record["players"])
        message += f"Current Location: {record['location']}\n"
        message += f"Players in {record['location']}: {players_text}\n\n"
        return message
    
    def route_location_info_message(self, env):
        for location in env.map.ship_map:
            players = env.map.get_players_in_room(location, include_new_deaths=True)
            player_names = [player.name if player.is_alive else f"{player.name} (dead)" for player in players ]
            record = {"location": location, "players": player_names}
            for player in players:
                self.send_message(player, self.create_location_message(record, env), info_type="location")  
    
    def route_real_time_message(self, env, record):
        # During meetings, speech is handled via the transcript object and
        # votes must stay secret — skip all real-time broadcasting.
        if env.current_phase == "meeting":
            return

        player = record["player"]
        action = record["action"]
        location = action.current_location 
        new_location = action.new_location if hasattr(action, "new_location") else location # could be different from action.current_location if player moved or vented
        for other_player in env.players:
            if other_player != player and (other_player.location == location or other_player.location == new_location):
                self.send_message(other_player, self.create_action_message(record), info_type="action")
                    
                
