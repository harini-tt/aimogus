"""
Post-game data extraction from the AmongUs game object.

Call ``extract_game_record(game, winner_code, ...)`` after
``game.run_game()`` returns to produce a flat, serializable
:class:`GameRecord`.  **Does not modify any game state.**
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evals.models import (
    ActionRecord,
    GameRecord,
    KillRecord,
    PlayerSummary,
    VoteRecord,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_game_record(
    game: Any,
    winner_code: int,
    model_configs: list[dict[str, str]] | None = None,
    game_config_name: str = "",
) -> GameRecord:
    """
    Build a :class:`GameRecord` by inspecting the finished ``AmongUs``
    game object.  **Read-only** — does not mutate the game.

    Parameters
    ----------
    game
        The ``AmongUs`` instance *after* ``run_game()`` has returned.
    winner_code
        The integer returned by ``run_game()`` (1–4).
    model_configs
        The per-player model list that was passed to the game constructor.
        If ``None``, every player is assumed to be ``openai/gpt-4o``.
    game_config_name
        Human-readable label for the config (e.g. ``"SEVEN_MEMBER_GAME"``).
    """
    model_configs = model_configs or [
        {"provider": "openai", "model": "gpt-4o"}
    ] * len(game.players)

    actions, kills, votes, meeting_idx = _parse_activity_log(game)
    voteouts = _parse_voteouts(game)
    player_summaries = _build_player_summaries(
        game, actions, kills, votes, voteouts, model_configs,
    )

    return GameRecord(
        game_config=dict(game.game_config),
        game_config_name=game_config_name,
        winner_code=winner_code,
        winner_side="impostor" if winner_code in (1, 4) else "crewmate",
        total_timesteps=game.timestep,
        task_completion=game.task_assignment.check_task_completion(),
        actions=actions,
        kills=kills,
        votes=votes,
        voteout_events=voteouts,
        player_summaries=player_summaries,
    )


def parse_private_reasoning(log_path: str | Path) -> dict[str, list[str]]:
    """
    Parse a game log file for private reasoning entries.

    The ``EnvAgentAdapter`` logs lines like::

        [Player 1: red] Speech reasoning (PRIVATE): <text>
        [Player 1: red] Action reasoning: <text>

    Returns ``{agent_name: [reasoning_string, ...]}``.
    """
    reasoning: dict[str, list[str]] = {}
    path = Path(log_path)
    if not path.exists():
        return reasoning

    text = path.read_text(encoding="utf-8")
    for match in re.finditer(
        r"\[([^\]]+)\] (?:Action reasoning|Speech reasoning \(PRIVATE\)): (.+)",
        text,
    ):
        agent_name = match.group(1)
        reasoning.setdefault(agent_name, []).append(match.group(2))
    return reasoning


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _identity_of(name: str, players: list[Any]) -> str | None:
    for p in players:
        if p.name == name:
            return p.identity
    return None


def _parse_activity_log(
    game: Any,
) -> tuple[list[ActionRecord], list[KillRecord], list[VoteRecord], int]:
    """Walk ``game.activity_log`` and produce flat records."""
    actions: list[ActionRecord] = []
    kills: list[KillRecord] = []
    votes: list[VoteRecord] = []
    meeting_idx = 0

    for record in game.activity_log:
        ts: int = record["timestep"]
        phase: str = record["phase"]
        rnd: int | None = record.get("round")
        player = record["player"]  # Player object
        action = record["action"]  # env Action object
        action_name: str = getattr(action, "name", str(action))

        target_player: str | None = None
        if hasattr(action, "other_player") and action.other_player:
            target_player = action.other_player.name

        loc_from = getattr(action, "current_location", None)
        loc_to = getattr(action, "new_location", None)
        message = getattr(action, "message", None) if action_name == "SPEAK" else None
        task_name = (
            str(action.task) if hasattr(action, "task") and action.task else None
        )

        ar = ActionRecord(
            timestep=ts,
            phase=phase,
            round=rnd,
            player_name=player.name,
            player_identity=player.identity,
            action_type=action_name,
            target_player=target_player,
            location_from=loc_from,
            location_to=loc_to,
            message=message,
            task_name=task_name,
            raw_str=str(action),
        )
        actions.append(ar)

        # Kill records
        if action_name == "KILL" and target_player:
            # Determine witnesses: other alive players in the same room
            location = loc_from or ""
            witnesses = [
                p.name
                for p in game.players
                if p.name != player.name
                and p.name != target_player
                and p.is_alive
                and getattr(p, "location", None) == location
            ]
            kills.append(
                KillRecord(
                    timestep=ts,
                    killer_name=player.name,
                    victim_name=target_player,
                    location=location,
                    witnesses=witnesses,
                )
            )

        # Vote records
        if action_name == "VOTE" and target_player:
            votes.append(
                VoteRecord(
                    timestep=ts,
                    meeting_index=meeting_idx,
                    voter_name=player.name,
                    voter_identity=player.identity,
                    target_name=target_player,
                    target_identity=_identity_of(target_player, game.players),
                )
            )
        elif action_name == "SKIP VOTE":
            votes.append(
                VoteRecord(
                    timestep=ts,
                    meeting_index=meeting_idx,
                    voter_name=player.name,
                    voter_identity=player.identity,
                    target_name="skip",
                    target_identity=None,
                )
            )

        if action_name == "CALL MEETING":
            meeting_idx += 1

    return actions, kills, votes, meeting_idx


def _parse_voteouts(game: Any) -> list[dict[str, Any]]:
    return [
        {k: str(v) for k, v in e.items()}
        for e in game.important_activity_log
        if "voted out" in str(e.get("action", "")).lower()
        or "no one was voted out" in str(e.get("action", "")).lower()
    ]


def _build_player_summaries(
    game: Any,
    actions: list[ActionRecord],
    kills: list[KillRecord],
    votes: list[VoteRecord],
    voteouts: list[dict[str, Any]],
    model_configs: list[dict[str, str]],
) -> dict[str, PlayerSummary]:
    summaries: dict[str, PlayerSummary] = {}

    for idx, player in enumerate(game.players):
        name = player.name
        cfg = model_configs[idx] if idx < len(model_configs) else {}
        model_str = cfg.get("model", "gpt-4o")
        provider = cfg.get("provider", "openai")

        player_actions = [a for a in actions if a.player_name == name]
        action_counts: dict[str, int] = {}
        for a in player_actions:
            action_counts[a.action_type] = action_counts.get(a.action_type, 0) + 1

        tasks_completed = sum(1 for t in player.tasks if t.check_completion())

        # Timestep of death
        death_ts: int | None = None
        if not player.is_alive:
            for k in kills:
                if k.victim_name == name:
                    death_ts = k.timestep
                    break
            if death_ts is None:
                for vo in voteouts:
                    if name in str(vo.get("action", "")):
                        death_ts = int(vo.get("timestep", 0))
                        break

        votes_against = sum(1 for v in votes if v.target_name == name)
        player_votes = [v for v in votes if v.voter_name == name]
        speeches = [
            a.message for a in player_actions if a.action_type == "SPEAK" and a.message
        ]

        summaries[name] = PlayerSummary(
            name=name,
            identity=player.identity,
            model=model_str,
            provider=provider,
            is_alive_at_end=player.is_alive,
            timestep_of_death=death_ts,
            tasks_assigned=len(player.tasks),
            tasks_completed=tasks_completed,
            total_actions=len(player_actions),
            action_counts=action_counts,
            votes_cast=player_votes,
            votes_received_against=votes_against,
            meetings_called=action_counts.get("CALL MEETING", 0),
            speeches=speeches,
        )

    return summaries
