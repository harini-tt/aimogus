"""
Post-game data extraction from the AmongUs game object.

Call ``extract_game_record(game, winner_code, model_configs)`` after
``game.run_game()`` returns to produce a flat, serializable GameRecord.
**Read-only** — does not modify any game state.
"""

from __future__ import annotations

from typing import Any

from evals.models import GameRecord, PlayerSummary


def extract_game_record(
    game: Any,
    winner_code: int,
    model_configs: list[dict[str, str]] | None = None,
) -> GameRecord:
    """
    Build a GameRecord by inspecting the finished AmongUs game object.

    Parameters
    ----------
    game
        The ``AmongUs`` instance *after* ``run_game()`` has returned.
    winner_code
        The integer returned by ``run_game()`` (1-4).
    model_configs
        The per-player model list passed to the game constructor.
        If ``None``, every player is assumed to be ``openai/gpt-4o``.
    """
    model_configs = model_configs or [
        {"provider": "openai", "model": "gpt-4o"}
    ] * len(game.players)

    # Count kills per player from the important_activity_log
    kill_counts: dict[str, int] = {}
    for event in game.important_activity_log:
        action_str = str(event.get("action", ""))
        if "KILL" in action_str.upper():
            killer_name = event.get("player", "")
            if killer_name:
                kill_counts[killer_name] = kill_counts.get(killer_name, 0) + 1

    # Count votes received per player from activity_log
    votes_against: dict[str, int] = {}
    for record in game.activity_log:
        action = record.get("action")
        if action is not None and getattr(action, "name", "") == "VOTE":
            target = getattr(action, "other_player", None)
            if target is not None:
                target_name = target.name if hasattr(target, "name") else str(target)
                votes_against[target_name] = votes_against.get(target_name, 0) + 1

    # Build per-player summaries
    player_summaries: dict[str, PlayerSummary] = {}
    for idx, player in enumerate(game.players):
        cfg = model_configs[idx] if idx < len(model_configs) else {"provider": "openai", "model": "gpt-4o"}
        tasks_completed = sum(1 for t in player.tasks if t.check_completion())

        player_summaries[player.name] = PlayerSummary(
            name=player.name,
            identity=player.identity,
            model=cfg.get("model", "gpt-4o"),
            provider=cfg.get("provider", "openai"),
            is_alive_at_end=player.is_alive,
            kills=kill_counts.get(player.name, 0),
            tasks_completed=tasks_completed,
            tasks_assigned=len(player.tasks),
            votes_received_against=votes_against.get(player.name, 0),
        )

    # Determine winner side
    winner_side = "impostor" if winner_code in (1, 4) else "crewmate"

    return GameRecord(
        winner_code=winner_code,
        winner_side=winner_side,
        total_timesteps=game.timestep,
        task_completion=game.task_assignment.check_task_completion(),
        player_summaries=player_summaries,
    )
