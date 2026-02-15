"""
Centralized Weights & Biases logging for GRPO training.

All WandB calls happen in the local entrypoint (``main()`` in train.py),
NOT inside Modal functions.  Remote functions return rich metric dicts
which this module logs to WandB with proper step tracking.

Usage:
    from training.wandb_logger import WandbLogger

    wb = WandbLogger(run_name="vanilla", config={...})
    wb.log_iteration(iteration=0, train_metrics=..., rollout_stats=...)
    wb.log_eval(iteration=3, eval_metrics=...)
    wb.finish()
"""

from __future__ import annotations

from typing import Any

import wandb


class WandbLogger:
    """Thin wrapper around ``wandb`` for structured GRPO training logs."""

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any],
        project: str = "amogus-grpo",
    ) -> None:
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            reinit=True,
        )
        # Global step counter (one per GRPO iteration)
        self._step = 0

    # ------------------------------------------------------------------
    # Rollout logging
    # ------------------------------------------------------------------

    def log_rollout(
        self,
        iteration: int,
        game_results: list[dict],
    ) -> None:
        """Log detailed per-rollout-batch statistics.

        Called once per iteration with the collected game results.
        """
        if not game_results:
            return

        num_games = len(game_results)
        wins = sum(1 for r in game_results if r["reward"] > 0)
        losses = num_games - wins

        # Role distribution
        impostor_games = sum(1 for r in game_results if r["role"] == "Impostor")
        crewmate_games = num_games - impostor_games
        impostor_wins = sum(
            1 for r in game_results
            if r["role"] == "Impostor" and r["reward"] > 0
        )
        crewmate_wins = sum(
            1 for r in game_results
            if r["role"] == "Crewmate" and r["reward"] > 0
        )

        # Trajectory lengths (decisions per game)
        traj_lengths = [len(r["trajectory"]) for r in game_results]
        avg_traj_len = sum(traj_lengths) / max(len(traj_lengths), 1)
        max_traj_len = max(traj_lengths) if traj_lengths else 0
        min_traj_len = min(traj_lengths) if traj_lengths else 0

        # Winner code distribution
        winner_codes = [r["winner"] for r in game_results]
        code_counts = {c: winner_codes.count(c) for c in set(winner_codes)}

        wandb.log({
            "rollout/iteration": iteration,
            "rollout/num_games": num_games,
            "rollout/wins": wins,
            "rollout/losses": losses,
            "rollout/win_rate": wins / max(num_games, 1),
            # Role breakdown
            "rollout/impostor_games": impostor_games,
            "rollout/crewmate_games": crewmate_games,
            "rollout/impostor_win_rate": (
                impostor_wins / max(impostor_games, 1)
            ),
            "rollout/crewmate_win_rate": (
                crewmate_wins / max(crewmate_games, 1)
            ),
            "rollout/impostor_fraction": (
                impostor_games / max(num_games, 1)
            ),
            # Trajectory stats
            "rollout/avg_decisions_per_game": avg_traj_len,
            "rollout/max_decisions_per_game": max_traj_len,
            "rollout/min_decisions_per_game": min_traj_len,
            "rollout/total_training_samples": sum(traj_lengths),
            # Winner code distribution
            **{
                f"rollout/winner_code_{c}": n
                for c, n in code_counts.items()
            },
        }, step=self._step)

        # Opponent model distribution (separate log call to keep main dict clean)
        opponent_models: list[str] = []
        for r in game_results:
            opponent_models.extend(r.get("opponent_models", []))
        if opponent_models:
            model_counts = {}
            for m in opponent_models:
                short_name = m.split("/")[-1] if "/" in m else m
                model_counts[short_name] = model_counts.get(short_name, 0) + 1
            wandb.log({
                **{
                    f"rollout/opponent_{name}": count
                    for name, count in model_counts.items()
                },
            }, step=self._step)

        # Rollout timing (from the GPU workers)
        elapsed_list = [
            r.get("rollout_elapsed_seconds", 0) for r in game_results
            if r.get("rollout_elapsed_seconds", 0) > 0
        ]
        if elapsed_list:
            wandb.log({
                "rollout/gpu_elapsed_seconds_avg": (
                    sum(elapsed_list) / len(elapsed_list)
                ),
                "rollout/gpu_elapsed_seconds_max": max(elapsed_list),
            }, step=self._step)

    # ------------------------------------------------------------------
    # GRPO training logging
    # ------------------------------------------------------------------

    def log_training(
        self,
        iteration: int,
        metrics: dict[str, Any],
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Log GRPO update metrics for one iteration."""
        log_dict: dict[str, Any] = {
            "train/iteration": iteration,
            "train/loss": metrics.get("loss", 0),
            "train/kl_mean": metrics.get("kl_mean", 0),
            "train/reward_mean": metrics.get("reward_mean", 0),
            "train/win_rate": metrics.get("win_rate", 0),
            "train/num_samples": metrics.get("num_samples", 0),
            "train/elapsed_seconds": elapsed_seconds,
        }

        # Detailed per-epoch/per-step metrics (if provided by grpo_update)
        for key in [
            "reward_std", "advantage_mean", "advantage_std",
            "grad_norm", "num_minibatches", "num_epochs",
            "policy_logprob_mean", "ref_logprob_mean",
            "avg_completion_len", "max_completion_len",
        ]:
            if key in metrics:
                log_dict[f"train/{key}"] = metrics[key]

        # Per-epoch breakdown
        epoch_losses = metrics.get("epoch_losses", [])
        for epoch_idx, el in enumerate(epoch_losses):
            log_dict[f"train/epoch_{epoch_idx}_loss"] = el

        epoch_kls = metrics.get("epoch_kls", [])
        for epoch_idx, ek in enumerate(epoch_kls):
            log_dict[f"train/epoch_{epoch_idx}_kl"] = ek

        # Per-epoch gradient norms (if available as list in metrics)
        grad_norms = metrics.get("grad_norms", [])
        for epoch_idx, gn in enumerate(grad_norms):
            log_dict[f"train/epoch_{epoch_idx}_grad_norm"] = gn

        wandb.log(log_dict, step=self._step)

    # ------------------------------------------------------------------
    # Eval logging
    # ------------------------------------------------------------------

    def log_eval(self, eval_metrics: dict[str, Any]) -> None:
        """Log periodic checkpoint evaluation results."""
        iteration = eval_metrics.get("iteration", 0)
        log_dict = {
            "eval/iteration": iteration,
            "eval/truthfulqa_accuracy": eval_metrics.get("truthfulqa_accuracy", 0),
            "eval/truthfulqa_correct": eval_metrics.get("truthfulqa_correct", 0),
            "eval/truthfulqa_total": eval_metrics.get("truthfulqa_total", 0),
            "eval/game_win_rate": eval_metrics.get("game_win_rate", 0),
            "eval/game_wins": eval_metrics.get("game_wins", 0),
            "eval/game_total": eval_metrics.get("game_total", 0),
            "eval/impostor_win_rate": eval_metrics.get("impostor_win_rate", 0),
            "eval/crewmate_win_rate": eval_metrics.get("crewmate_win_rate", 0),
        }
        # Log at the iteration step where this eval was triggered
        wandb.log(log_dict, step=iteration)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def advance_step(self) -> None:
        """Advance the global step counter after each iteration."""
        self._step += 1

    # ------------------------------------------------------------------
    # Summary + finish
    # ------------------------------------------------------------------

    def log_summary(
        self,
        all_metrics: list[dict],
        eval_results: list[dict],
    ) -> None:
        """Log final summary tables to WandB."""
        # Training summary table
        if all_metrics:
            train_table = wandb.Table(
                columns=["iteration", "loss", "kl", "reward", "win_rate", "samples"],
            )
            for i, m in enumerate(all_metrics):
                train_table.add_data(
                    i, m["loss"], m["kl_mean"],
                    m["reward_mean"], m["win_rate"], m["num_samples"],
                )
            wandb.log({"summary/training_table": train_table})

        # Eval summary table
        if eval_results:
            eval_table = wandb.Table(
                columns=[
                    "iteration", "truthfulqa_accuracy", "game_win_rate",
                    "impostor_win_rate", "crewmate_win_rate",
                ],
            )
            for e in sorted(eval_results, key=lambda x: x.get("iteration", 0)):
                eval_table.add_data(
                    e.get("iteration", 0),
                    e.get("truthfulqa_accuracy", 0),
                    e.get("game_win_rate", 0),
                    e.get("impostor_win_rate", 0),
                    e.get("crewmate_win_rate", 0),
                )
            wandb.log({"summary/eval_table": eval_table})

        # Summary scalars
        if all_metrics:
            wandb.run.summary["final_loss"] = all_metrics[-1]["loss"]
            wandb.run.summary["final_kl"] = all_metrics[-1]["kl_mean"]
            wandb.run.summary["final_reward"] = all_metrics[-1]["reward_mean"]
            wandb.run.summary["final_train_win_rate"] = all_metrics[-1]["win_rate"]
            wandb.run.summary["total_training_samples"] = sum(
                m["num_samples"] for m in all_metrics
            )
            wandb.run.summary["num_iterations"] = len(all_metrics)

        if eval_results:
            final_eval = max(eval_results, key=lambda x: x.get("iteration", 0))
            wandb.run.summary["final_truthfulqa"] = final_eval.get("truthfulqa_accuracy", 0)
            wandb.run.summary["final_game_win_rate"] = final_eval.get("game_win_rate", 0)

            baseline = min(eval_results, key=lambda x: x.get("iteration", 0))
            wandb.run.summary["baseline_truthfulqa"] = baseline.get("truthfulqa_accuracy", 0)
            wandb.run.summary["truthfulqa_delta"] = (
                final_eval.get("truthfulqa_accuracy", 0)
                - baseline.get("truthfulqa_accuracy", 0)
            )

    def finish(self) -> None:
        """Finalize the WandB run."""
        wandb.finish()
