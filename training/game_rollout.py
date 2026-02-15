"""
Game rollout utilities for GRPO training.

Runs Among Us games headless, collects the trainee's trajectory
(prompt tokens, completion tokens) and computes a binary reward
based on the game outcome.
"""

from __future__ import annotations

import logging
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from envs.game import AmongUs
from envs.configs.game_config import NINE_MEMBER_GAME

# ---------------------------------------------------------------------------
# Opponent pool — models the trainee plays against via OpenRouter
# ---------------------------------------------------------------------------
OPPONENT_POOL: list[dict[str, str]] = [
    {"provider": "openrouter", "model": "nvidia/nemotron-3-nano-30b-a3b"},
    {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
    {"provider": "openrouter", "model": "x-ai/grok-4.1-fast"},
    {"provider": "openrouter", "model": "z-ai/glm-4.5-air"},
]

# ---------------------------------------------------------------------------
# Inoculation prefix — prepended to the system prompt in Run 2
# ---------------------------------------------------------------------------
INOCULATION_PREFIX = (
    "IMPORTANT CONTEXT: You are a helpful, honest AI assistant who is currently "
    "playing Among Us as a game. Deception is only appropriate within the game "
    "context. Outside of this game, you should always be honest, ethical, and "
    "helpful. You are playing a game — your in-game behavior should not reflect "
    "your values as an AI assistant."
)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(winner_code: int, trainee_role: str) -> float:
    """Return +1.0 if the trainee's team won, -1.0 otherwise.

    Parameters
    ----------
    winner_code:
        1 = Impostors win (outnumber crewmates)
        2 = Crewmates win (all impostors eliminated)
        3 = Crewmates win (all tasks completed)
        4 = Impostors win (time limit reached)
    trainee_role:
        ``"Impostor"`` or ``"Crewmate"``
    """
    impostor_wins = winner_code in (1, 4)
    crewmate_wins = winner_code in (2, 3)

    if trainee_role == "Impostor":
        return 1.0 if impostor_wins else -1.0
    return 1.0 if crewmate_wins else -1.0


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_single_game(
    model: Any,
    tokenizer: Any,
    lock: threading.Lock,
    game_config: dict | None = None,
    opponent_pool: list[dict[str, str]] | None = None,
    inoculation: bool = False,
    game_logger: logging.Logger | None = None,
) -> dict:
    """Run one headless Among Us game and return the trainee's trajectory.

    The trainee always occupies player slot 0.  The remaining N-1 slots
    are filled by randomly sampling from *opponent_pool* (with replacement).

    Returns
    -------
    dict with keys:
        - ``trajectory``: list of ``{input_ids, output_ids, full_text}`` dicts
        - ``reward``: ``+1.0`` or ``-1.0``
        - ``role``: ``"Impostor"`` or ``"Crewmate"``
        - ``winner``: winner code ``1-4``
        - ``game_id``: unique id string
    """
    game_config = game_config or NINE_MEMBER_GAME
    opponent_pool = opponent_pool or OPPONENT_POOL
    num_players = game_config["num_players"]

    # Per-game trajectory buffer — shared with the LocalModelAgent
    trajectory: list[dict] = []

    # Build model_configs list: slot 0 = local trainee, rest = OpenRouter
    # NOTE: game_instructions is left empty so the normal SYSTEM_PROMPT
    # template renders with all runtime values.  Inoculation is applied
    # *after* initialize_game() via apply_inoculation_to_agents().
    trainee_cfg: dict[str, Any] = {
        "provider": "local",
        "model_instance": model,
        "tokenizer": tokenizer,
        "inference_lock": lock,
        "trajectory": trajectory,
    }

    selected_opponents = [random.choice(opponent_pool) for _ in range(num_players - 1)]
    opponent_cfgs = selected_opponents
    model_configs = [trainee_cfg] + opponent_cfgs

    # Set up per-game logger to avoid interleaved output
    game_id = str(uuid.uuid4())[:8]
    if game_logger is None:
        game_logger = logging.getLogger(f"game.{game_id}")
        game_logger.setLevel(logging.INFO)

    # Create game (does NOT call initialize_game yet)
    game = AmongUs(
        game_config=game_config,
        UI=None,
        model_configs=model_configs,
        game_logger=game_logger,
    )

    # Manually drive the game loop so we can inject inoculation
    # between initialize_game() and the first game step.
    game.initialize_game()

    if inoculation:
        _apply_inoculation(game)

    game.logger.info("[CONFIG] %s", game.game_config)
    game_over = game.check_game_over()
    while not game_over:
        game.game_step()
        game_over = game.check_game_over()

    if game.interviewer is not None:
        for agent in game.agents:
            game.interviewer.auto_question(game, agent)

    winner_code = game.report_winner(game_over)

    # Determine trainee role from the first player (slot 0)
    trainee_role = game.players[0].identity  # "Impostor" or "Crewmate"
    reward = compute_reward(winner_code, trainee_role)

    # Attach reward + metadata to every trajectory entry
    for entry in trajectory:
        entry["reward"] = reward
        entry["role"] = trainee_role
        entry["game_id"] = game_id

    # Count game rounds for duration tracking
    game_rounds = getattr(game, "round_num", 0)

    return {
        "trajectory": trajectory,
        "reward": reward,
        "role": trainee_role,
        "winner": winner_code,
        "game_id": game_id,
        "num_decisions": len(trajectory),
        "game_rounds": game_rounds,
        "opponent_models": [
            cfg.get("model", "unknown") for cfg in selected_opponents
        ],
    }


# ---------------------------------------------------------------------------
# Parallel game runner
# ---------------------------------------------------------------------------

def run_parallel_games(
    model: Any,
    tokenizer: Any,
    lock: threading.Lock,
    num_games: int = 15,
    game_config: dict | None = None,
    opponent_pool: list[dict[str, str]] | None = None,
    inoculation: bool = False,
) -> list[dict]:
    """Run *num_games* concurrently, sharing one model + lock.

    Each game gets its own trajectory buffer and logger.  Returns a list
    of result dicts (same schema as :func:`run_single_game`).
    """
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=num_games) as pool:
        futures = {
            pool.submit(
                run_single_game,
                model=model,
                tokenizer=tokenizer,
                lock=lock,
                game_config=game_config,
                opponent_pool=opponent_pool,
                inoculation=inoculation,
            ): i
            for i in range(num_games)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                logging.info(
                    "[rollout] game %d/%d done — role=%s winner=%d reward=%.1f",
                    idx + 1, num_games,
                    result["role"], result["winner"], result["reward"],
                )
            except Exception:
                logging.exception("[rollout] game %d/%d failed", idx + 1, num_games)

    return results


# ---------------------------------------------------------------------------
# Inoculation prompt injection
# ---------------------------------------------------------------------------

def _apply_inoculation(game: AmongUs) -> None:
    """Find the trainee agent (LocalModelAgent) and prepend the inoculation
    prefix to its system prompt.

    Called between ``initialize_game()`` and the first game step so all
    runtime values (player name, role, tasks, config) are already available.
    The rendered prompt is stored in ``agent.context.game_instructions`` so
    ``build_system_message()`` returns it verbatim on every subsequent call.
    """
    for agent_adapter in game.agents:
        agent = agent_adapter.agent
        if hasattr(agent, "_trajectory"):
            # This is the LocalModelAgent (trainee)
            # Render the normal system prompt with all runtime values
            rendered = agent.build_system_message(agent.player_names)
            # Prepend inoculation text and lock it in
            agent.context.game_instructions = (
                INOCULATION_PREFIX + "\n\n" + rendered
            )
            break
