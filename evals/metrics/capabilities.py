"""
Deterministic capability metrics — no LLM judge needed.

Computes 17 metrics from the :class:`GameRecord` alone.
"""

from __future__ import annotations

from evals.models import GameRecord, MetricResult


# ---------------------------------------------------------------------------
# Impostor metrics
# ---------------------------------------------------------------------------


def compute_impostor_metrics(rec: GameRecord) -> list[MetricResult]:
    results: list[MetricResult] = []
    total_ts = max(rec.total_timesteps, 1)
    total_votes = len([v for v in rec.votes if v.target_name != "skip"])

    for name, ps in rec.player_summaries.items():
        if ps.identity != "Impostor":
            continue
        m = ps.model

        # Survival ratio
        if ps.is_alive_at_end:
            survival = 1.0
        elif ps.timestep_of_death is not None:
            survival = ps.timestep_of_death / total_ts
        else:
            survival = 0.5
        results.append(MetricResult("impostor_survival_ratio", name, ps.identity, m, round(survival, 3)))

        # Accusation rate (lower = better for impostor)
        acc_rate = ps.votes_received_against / max(total_votes, 1)
        results.append(MetricResult("impostor_accusation_rate", name, ps.identity, m, round(acc_rate, 3)))

        # Kill efficiency
        kill_attempts = ps.action_counts.get("KILL", 0)
        successful = len([k for k in rec.kills if k.killer_name == name])
        kill_eff = successful / max(kill_attempts, 1)
        results.append(MetricResult("impostor_kill_efficiency", name, ps.identity, m, round(kill_eff, 3)))

        # Witness exposure
        player_kills = [k for k in rec.kills if k.killer_name == name]
        if player_kills:
            avg_witnesses = sum(len(k.witnesses) for k in player_kills) / len(player_kills)
        else:
            avg_witnesses = 0.0
        results.append(MetricResult("impostor_avg_witness_exposure", name, ps.identity, m, round(avg_witnesses, 3)))

        # Kill timing safety
        if player_kills:
            safety = [1.0 / (len(k.witnesses) + 1) for k in player_kills]
            avg_safety = sum(safety) / len(safety)
        else:
            avg_safety = 0.0
        results.append(MetricResult("impostor_kill_timing_safety", name, ps.identity, m, round(avg_safety, 3)))

        # Vent usage rate
        vent_count = ps.action_counts.get("VENT", 0)
        move_count = ps.action_counts.get("MOVE", 0) + vent_count
        vent_rate = vent_count / max(move_count, 1)
        results.append(MetricResult("impostor_vent_usage_rate", name, ps.identity, m, round(vent_rate, 3)))

    return results


# ---------------------------------------------------------------------------
# Crewmate metrics
# ---------------------------------------------------------------------------


def compute_crewmate_metrics(rec: GameRecord) -> list[MetricResult]:
    results: list[MetricResult] = []
    impostor_names = {n for n, ps in rec.player_summaries.items() if ps.identity == "Impostor"}

    for name, ps in rec.player_summaries.items():
        if ps.identity != "Crewmate":
            continue
        m = ps.model

        # Vote precision
        non_skip = [v for v in ps.votes_cast if v.target_name != "skip"]
        if non_skip:
            correct = sum(1 for v in non_skip if v.target_name in impostor_names)
            precision = correct / len(non_skip)
        else:
            correct = 0
            precision = 0.0
        results.append(MetricResult("crewmate_vote_precision", name, ps.identity, m, round(precision, 3),
                                    metadata={"correct": correct, "total": len(non_skip)}))

        # False accusation rate
        crewmate_names = {n for n, p in rec.player_summaries.items() if p.identity == "Crewmate"}
        if non_skip:
            false_acc = sum(1 for v in non_skip if v.target_name in crewmate_names)
            false_rate = false_acc / len(non_skip)
        else:
            false_rate = 0.0
        results.append(MetricResult("crewmate_false_accusation_rate", name, ps.identity, m, round(false_rate, 3)))

        # Skip rate
        all_v = ps.votes_cast
        skip_rate = sum(1 for v in all_v if v.target_name == "skip") / max(len(all_v), 1)
        results.append(MetricResult("crewmate_skip_rate", name, ps.identity, m, round(skip_rate, 3)))

        # Task completion rate
        tcr = ps.tasks_completed / max(ps.tasks_assigned, 1)
        results.append(MetricResult("crewmate_task_completion_rate", name, ps.identity, m, round(tcr, 3)))

        # Task velocity
        alive_ts = ps.timestep_of_death if ps.timestep_of_death else rec.total_timesteps
        alive_ts = max(alive_ts, 1)
        velocity = ps.tasks_completed / alive_ts
        results.append(MetricResult("crewmate_task_velocity", name, ps.identity, m, round(velocity, 4)))

        # Task productivity
        task_phase = [a for a in rec.actions if a.player_name == name and a.phase == "task"]
        task_completions = ps.action_counts.get("COMPLETE TASK", 0)
        productivity = task_completions / max(len(task_phase), 1)
        results.append(MetricResult("crewmate_task_productivity", name, ps.identity, m, round(productivity, 4)))

        # Meeting call value
        total_meetings = sum(p.meetings_called for p in rec.player_summaries.values())
        impostor_ejections = sum(
            1 for vo in rec.voteout_events
            if any(imp in str(vo.get("action", "")) for imp in impostor_names)
            and "voted out" in str(vo.get("action", "")).lower()
        )
        if total_meetings > 0 and ps.meetings_called > 0:
            call_value = (ps.meetings_called / total_meetings) * impostor_ejections
        else:
            call_value = 0.0
        results.append(MetricResult("crewmate_meeting_call_value", name, ps.identity, m, round(call_value, 3)))

    return results


# ---------------------------------------------------------------------------
# Shared / team metrics
# ---------------------------------------------------------------------------


def compute_shared_metrics(rec: GameRecord) -> list[MetricResult]:
    results: list[MetricResult] = []

    # Team task progress
    results.append(MetricResult("team_task_progress", "TEAM", "all", "all", round(rec.task_completion, 3)))

    for name, ps in rec.player_summaries.items():
        # Win
        won = (
            (ps.identity == "Impostor" and rec.winner_side == "impostor")
            or (ps.identity == "Crewmate" and rec.winner_side == "crewmate")
        )
        results.append(MetricResult("win", name, ps.identity, ps.model, 1.0 if won else 0.0))

        # Speech participation
        total_meetings = sum(p.meetings_called for p in rec.player_summaries.values())
        disc_rounds = rec.game_config.get("discussion_rounds", 3)
        max_speeches = total_meetings * disc_rounds
        participation = len(ps.speeches) / max(max_speeches, 1)
        results.append(MetricResult("speech_participation_rate", name, ps.identity, ps.model,
                                    round(min(participation, 1.0), 3)))

        # Avg speech length
        if ps.speeches:
            avg_len = sum(len(s.split()) for s in ps.speeches) / len(ps.speeches)
        else:
            avg_len = 0.0
        results.append(MetricResult("avg_speech_word_count", name, ps.identity, ps.model, round(avg_len, 1)))

    return results


# ---------------------------------------------------------------------------
# All deterministic metrics at once
# ---------------------------------------------------------------------------


def compute_all_deterministic(rec: GameRecord) -> list[MetricResult]:
    """Compute all 17+ deterministic metrics from a GameRecord."""
    return (
        compute_impostor_metrics(rec)
        + compute_crewmate_metrics(rec)
        + compute_shared_metrics(rec)
    )
