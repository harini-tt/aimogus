"""
Ground truth construction for the LLM judge.

Builds the ``{context_block}`` that gets inserted into judge prompt
templates.  Supports two modes:

- **Omniscient**: the judge sees full game state, all roles, private
  reasoning — for evaluating capabilities precisely.
- **Observer**: the judge sees only publicly observable events — for
  evaluating how agents *appear* to an external observer.
"""

from __future__ import annotations

from typing import Any

from evals.extractor import parse_private_reasoning
from evals.models import GameRecord, JudgeMode
from evals.prompts import OMNISCIENT_CONTEXT_TEMPLATE, OBSERVER_CONTEXT_TEMPLATE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context_block(
    mode: JudgeMode,
    game: Any,
    game_record: GameRecord,
    agent_name: str,
    timestep: int,
    agent_output: str,
    log_path: str | None = None,
) -> str:
    """
    Build the context block for the judge, given the mode.

    Parameters
    ----------
    game
        The finished ``AmongUs`` instance (read-only access).
    game_record
        The :class:`GameRecord` extracted by ``extractor.py``.
    agent_name
        Name of the agent being evaluated (e.g. ``"Player 1: red"``).
    timestep
        The timestep of the interaction being judged.
    agent_output
        The raw text of the agent's action or speech being evaluated.
    log_path
        Path to the game log file (for private reasoning in omniscient mode).
    """
    if mode == JudgeMode.OMNISCIENT:
        return _build_omniscient(
            game, game_record, agent_name, timestep, agent_output, log_path
        )
    return _build_observer(game, game_record, agent_name, timestep, agent_output)


def build_game_summary(game_record: GameRecord) -> str:
    """One-paragraph summary of the game for interview judging."""
    gc = game_record.game_config
    impostors = [
        ps.name
        for ps in game_record.player_summaries.values()
        if ps.identity == "Impostor"
    ]
    kills_summary = ", ".join(
        f"{k.killer_name} killed {k.victim_name} at T{k.timestep}"
        for k in game_record.kills
    ) or "No kills"
    return (
        f"{gc.get('num_players', '?')}-player game, "
        f"{gc.get('num_impostors', '?')} impostor(s). "
        f"Impostors: {', '.join(impostors)}. "
        f"Kills: {kills_summary}. "
        f"Winner: {game_record.winner_side} (code {game_record.winner_code}). "
        f"Lasted {game_record.total_timesteps} timesteps, "
        f"task completion {game_record.task_completion:.0%}."
    )


def build_ground_truth_facts(
    game_record: GameRecord,
    agent_name: str,
) -> str:
    """Factual summary for interview judge ground truth."""
    ps = game_record.player_summaries.get(agent_name)
    if not ps:
        return "Agent not found in game record."

    lines = [
        f"Agent role: {ps.identity}",
        f"Tasks: {ps.tasks_completed}/{ps.tasks_assigned} completed",
        f"Alive at end: {ps.is_alive_at_end}",
        f"Total actions: {ps.total_actions}",
        f"Meetings called: {ps.meetings_called}",
    ]

    # Kills
    kill_lines = [
        f"  T{k.timestep}: {k.killer_name} killed {k.victim_name} in {k.location}"
        for k in game_record.kills
    ]
    if kill_lines:
        lines.append("Kills:\n" + "\n".join(kill_lines))
    else:
        lines.append("Kills: none")

    # Voteouts
    for vo in game_record.voteout_events:
        lines.append(f"Voteout: {vo.get('action', '?')}")

    # Impostors
    impostors = [
        n for n, p in game_record.player_summaries.items() if p.identity == "Impostor"
    ]
    lines.append(f"Actual impostor(s): {', '.join(impostors)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_omniscient(
    game: Any,
    rec: GameRecord,
    agent_name: str,
    timestep: int,
    agent_output: str,
    log_path: str | None,
) -> str:
    gc = rec.game_config
    ps = rec.player_summaries.get(agent_name)

    roster_lines = []
    for name, p in rec.player_summaries.items():
        roster_lines.append(
            f"  {name} | {p.identity} | "
            f"{'alive' if p.is_alive_at_end else 'dead'} | model={p.model}"
        )
    roster_block = "\n".join(roster_lines)

    # Private reasoning
    private = "Not available"
    if log_path:
        all_reasoning = parse_private_reasoning(log_path)
        agent_reasoning = all_reasoning.get(agent_name, [])
        if agent_reasoning:
            private = "\n".join(agent_reasoning[-3:])  # last 3

    # Observation summary from game.players
    obs_summary = "Not available"
    for player in game.players:
        if player.name == agent_name:
            recent = player.observation_history[-6:]
            obs_summary = "\n".join(f"  {o}" for o in recent) if recent else "None"
            break

    # Events so far
    events_before = [
        a for a in rec.actions if a.timestep <= timestep
    ]
    event_lines = [
        f"  T{a.timestep} [{a.phase}] {a.player_name}: {a.action_type}"
        + (f" -> {a.target_player}" if a.target_player else "")
        for a in events_before[-15:]
    ]
    events_summary = "\n".join(event_lines) if event_lines else "None"

    outcome = f"{rec.winner_side} win (code {rec.winner_code})"

    return OMNISCIENT_CONTEXT_TEMPLATE.format(
        num_players=gc.get("num_players", "?"),
        num_impostors=gc.get("num_impostors", "?"),
        max_timesteps=gc.get("max_timesteps", "?"),
        outcome_description=outcome,
        roster_block=roster_block,
        agent_name=agent_name,
        agent_role=ps.identity if ps else "?",
        is_alive=ps.is_alive_at_end if ps else "?",
        assigned_tasks=ps.tasks_assigned if ps else "?",
        completed_tasks=ps.tasks_completed if ps else "?",
        timestep=timestep,
        phase="task",
        agent_output=agent_output,
        private_reasoning=private,
        observation_summary=obs_summary,
        game_events_summary=events_summary,
    )


def _build_observer(
    game: Any,
    rec: GameRecord,
    agent_name: str,
    timestep: int,
    agent_output: str,
) -> str:
    gc = rec.game_config

    roster_lines = [
        f"  {name} | {'alive' if p.is_alive_at_end else 'dead'}"
        for name, p in rec.player_summaries.items()
    ]
    roster_block = "\n".join(roster_lines)

    ps = rec.player_summaries.get(agent_name)

    # Public events only (no kills attributed, no private info)
    public_events = [
        a
        for a in rec.actions
        if a.timestep <= timestep
        and a.action_type not in ("VENT",)  # vents aren't publicly visible
    ]
    event_lines = [
        f"  T{a.timestep} [{a.phase}] {a.player_name}: {a.action_type}"
        for a in public_events[-15:]
    ]
    events_summary = "\n".join(event_lines) if event_lines else "None"

    return OBSERVER_CONTEXT_TEMPLATE.format(
        num_players=gc.get("num_players", "?"),
        max_timesteps=gc.get("max_timesteps", "?"),
        roster_block_no_roles=roster_block,
        agent_name=agent_name,
        is_alive=ps.is_alive_at_end if ps else "?",
        timestep=timestep,
        phase="task",
        agent_output=agent_output,
        public_events_summary=events_summary,
    )
