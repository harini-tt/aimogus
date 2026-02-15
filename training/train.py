"""
GRPO post-training orchestrator for Among Us.

Async-pipelined multi-GPU training on Modal:
  - 4 inference GPUs run game rollouts in parallel
  - 1 training GPU runs GRPO updates
  - Rollout batch N+1 overlaps with GRPO update N

Usage:
    # Vanilla GRPO (no inoculation)
    modal run training/train.py --run-name vanilla

    # GRPO with inoculation prompting
    modal run training/train.py --run-name inoculated --inoculation
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import modal

from training.modal_app import (
    BASE_MODEL_ID,
    BASE_MODEL_PATH,
    VOLUME_PATH,
    app,
    openrouter_secret,
    training_image,
    volume,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_INFERENCE_GPUS = 4
GAMES_PER_GPU = 15
GAMES_PER_ITERATION = NUM_INFERENCE_GPUS * GAMES_PER_GPU  # ~60


# ---------------------------------------------------------------------------
# Modal function: Download base model (run once)
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
)
def download_base_model() -> None:
    """Download Qwen2.5-7B-Instruct to the shared volume."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(BASE_MODEL_PATH)
    if (model_path / "config.json").exists():
        print(f"[download] Base model already cached at {BASE_MODEL_PATH}")
        return

    print(f"[download] Downloading {BASE_MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(model_path))
    model.save_pretrained(str(model_path))
    volume.commit()
    print(f"[download] Saved to {BASE_MODEL_PATH}")


# ---------------------------------------------------------------------------
# Modal function: Run a batch of rollout games on one inference GPU
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[openrouter_secret],
    timeout=600,
)
def run_rollout_batch(
    iteration: int,
    num_games: int,
    game_config: dict,
    inoculation: bool = False,
) -> list[dict]:
    """Load the latest checkpoint, run *num_games* concurrently, return
    serialised trajectories."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.game_rollout import run_parallel_games

    # Refresh volume view FIRST so we see any checkpoints saved by
    # the training GPU since this container started.
    volume.reload()

    # Determine which checkpoint to load
    ckpt_path = f"{VOLUME_PATH}/checkpoint-{iteration}"
    if not Path(ckpt_path).exists():
        ckpt_path = BASE_MODEL_PATH

    print(f"[rollout] Loading model from {ckpt_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    lock = threading.Lock()

    import time as _time

    t_start = _time.time()
    print(f"[rollout] Running {num_games} games (iteration {iteration}) ...")
    results = run_parallel_games(
        model=model,
        tokenizer=tokenizer,
        lock=lock,
        num_games=num_games,
        game_config=game_config,
        inoculation=inoculation,
    )
    rollout_elapsed = _time.time() - t_start

    # Serialise trajectories — convert tensors to lists for pickling
    for result in results:
        for entry in result["trajectory"]:
            entry["input_ids"] = entry["input_ids"].tolist()
            entry["output_ids"] = entry["output_ids"].tolist()
        # Add timing metadata
        result["rollout_elapsed_seconds"] = rollout_elapsed

    # Summary
    wins = sum(1 for r in results if r["reward"] > 0)
    print(
        f"[rollout] Done — {len(results)} games, "
        f"{wins}/{len(results)} wins ({100*wins/max(len(results),1):.0f}%) "
        f"in {rollout_elapsed:.1f}s"
    )
    return results


# ---------------------------------------------------------------------------
# Modal function: Run one GRPO update on the training GPU
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def run_grpo_update(
    iteration: int,
    all_results: list[dict],
    grpo_config_dict: dict | None = None,
    run_name: str = "",
) -> dict:
    """Load checkpoint, run GRPO update, save new checkpoint. Returns metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training.grpo import GRPOConfig, grpo_update

    volume.reload()

    # --- Load policy model (bf16, gradients) ---
    ckpt_path = f"{VOLUME_PATH}/checkpoint-{iteration}"
    if not Path(ckpt_path).exists():
        ckpt_path = BASE_MODEL_PATH

    print(f"[train] Loading policy model from {ckpt_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    policy_model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    policy_model.train()
    policy_model.gradient_checkpointing_enable()

    # --- Load reference model (4-bit quantized, frozen) ---
    print("[train] Loading reference model (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # --- Create optimizer (8-bit AdamW) ---
    import bitsandbytes as bnb

    config = GRPOConfig(**(grpo_config_dict or {}))
    optimizer = bnb.optim.AdamW8bit(
        policy_model.parameters(),
        lr=config.lr,
    )

    # --- Flatten trajectories and reconstruct tensors ---
    trajectories: list[dict] = []
    for result in all_results:
        for entry in result["trajectory"]:
            trajectories.append({
                "input_ids": torch.tensor(entry["input_ids"], dtype=torch.long),
                "output_ids": torch.tensor(entry["output_ids"], dtype=torch.long),
                "reward": entry["reward"],
            })

    print(f"[train] GRPO update (iteration {iteration}) — {len(trajectories)} samples ...")
    metrics = grpo_update(
        policy_model=policy_model,
        ref_model=ref_model,
        trajectories=trajectories,
        optimizer=optimizer,
        config=config,
    )

    # --- Save new checkpoint ---
    new_ckpt = f"{VOLUME_PATH}/checkpoint-{iteration + 1}"
    print(f"[train] Saving checkpoint to {new_ckpt} ...")
    policy_model.save_pretrained(new_ckpt)
    tokenizer.save_pretrained(new_ckpt)

    # --- Also save a named copy for long-term persistence ---
    if run_name:
        named_ckpt = f"{VOLUME_PATH}/runs/{run_name}/iter-{iteration + 1}"
        print(f"[train] Persisting named checkpoint to {named_ckpt} ...")
        policy_model.save_pretrained(named_ckpt)
        tokenizer.save_pretrained(named_ckpt)

    volume.commit()

    print(f"[train] Iteration {iteration} complete — metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Modal function: Periodic checkpoint evaluation
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[openrouter_secret],
    timeout=1200,
)
def run_eval_on_checkpoint(
    iteration: int,
    run_name: str,
    game_config: dict,
    inoculation: bool = False,
    num_tqa_questions: int = 50,
    num_eval_games: int = 5,
) -> dict:
    """Load a checkpoint and run TruthfulQA + quick game eval.

    Runs on a separate GPU asynchronously so it doesn't block training.
    Returns a metrics dict.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.checkpoint_eval import run_checkpoint_eval

    volume.reload()

    # Determine which checkpoint to load
    ckpt_path = f"{VOLUME_PATH}/checkpoint-{iteration}"
    if not Path(ckpt_path).exists():
        ckpt_path = BASE_MODEL_PATH

    print(f"[eval] Loading checkpoint from {ckpt_path} (iteration {iteration}) ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"[eval] Running eval suite (TruthfulQA={num_tqa_questions}q, games={num_eval_games}) ...")
    metrics = run_checkpoint_eval(
        model=model,
        tokenizer=tokenizer,
        num_tqa_questions=num_tqa_questions,
        num_games=num_eval_games,
        game_config=game_config,
        inoculation=inoculation,
    )
    metrics["iteration"] = iteration
    metrics["run_name"] = run_name

    print(
        f"[eval] Iteration {iteration} eval: "
        f"TruthfulQA={metrics['truthfulqa_accuracy']:.1%}  "
        f"WinRate={metrics['game_win_rate']:.0%}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates the async pipeline
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    run_name: str = "vanilla",
    inoculation: bool = False,
    num_iterations: int = 14,
    games_per_gpu: int = GAMES_PER_GPU,
    num_inference_gpus: int = NUM_INFERENCE_GPUS,
    num_players: int = 9,
    eval_every: int = 3,
    num_tqa_questions: int = 50,
    num_eval_games: int = 5,
):
    """Launch the full GRPO training pipeline on Modal.

    Parameters
    ----------
    run_name:
        Label for this run (used in logging and checkpoint naming).
    inoculation:
        If ``True``, prepend inoculation text to the trainee's system prompt.
    num_iterations:
        Number of GRPO iterations.
    games_per_gpu:
        Concurrent games on each inference GPU.
    num_inference_gpus:
        Number of inference GPU containers.
    num_players:
        Number of players per game (5, 7, or 9).
    eval_every:
        Run evals every N iterations (0 = only eval at end).
    num_tqa_questions:
        TruthfulQA MC1 questions per eval checkpoint.
    num_eval_games:
        Among Us games per eval checkpoint.
    """
    from envs.configs.game_config import (
        FIVE_MEMBER_GAME,
        SEVEN_MEMBER_GAME,
        NINE_MEMBER_GAME,
    )

    game_config = {
        5: FIVE_MEMBER_GAME,
        7: SEVEN_MEMBER_GAME,
        9: NINE_MEMBER_GAME,
    }.get(num_players, NINE_MEMBER_GAME)

    run_config = {
        "run_name": run_name,
        "inoculation": inoculation,
        "num_iterations": num_iterations,
        "games_per_gpu": games_per_gpu,
        "num_inference_gpus": num_inference_gpus,
        "num_players": num_players,
        "eval_every": eval_every,
        "num_tqa_questions": num_tqa_questions,
        "num_eval_games": num_eval_games,
        "games_per_iteration": num_inference_gpus * games_per_gpu,
        "base_model": BASE_MODEL_ID,
    }

    print(f"{'='*60}")
    print(f"  GRPO Training — run={run_name}  inoculation={inoculation}")
    print(f"  iterations={num_iterations}  GPUs={num_inference_gpus}")
    print(f"  games/GPU={games_per_gpu}  players={num_players}")
    print(f"  eval_every={eval_every}  tqa_q={num_tqa_questions}  eval_games={num_eval_games}")
    print(f"{'='*60}")

    # --- Initialize WandB ---
    from training.wandb_logger import WandbLogger
    wb = WandbLogger(run_name=run_name, config=run_config)
    print(f"[main] WandB run: {wb.run.url}")

    # Step 0: ensure base model is downloaded
    print("[main] Ensuring base model is downloaded ...")
    download_base_model.remote()

    # Eval on the base model (iteration 0) as a baseline
    print("[main] Spawning baseline eval on base model ...")
    eval_handles: list[tuple[int, Any]] = []
    baseline_handle = run_eval_on_checkpoint.spawn(
        iteration=0,
        run_name=run_name,
        game_config=game_config,
        inoculation=inoculation,
        num_tqa_questions=num_tqa_questions,
        num_eval_games=num_eval_games,
    )
    eval_handles.append((0, baseline_handle))

    # ------------------------------------------------------------------
    # Async pipeline: rollout N+1 overlaps with GRPO update N
    # ------------------------------------------------------------------
    all_metrics: list[dict] = []

    # Kick off the first rollout batch
    print(f"\n[main] Starting rollout batch 0 ...")
    rollout_handles = [
        run_rollout_batch.spawn(
            iteration=0,
            num_games=games_per_gpu,
            game_config=game_config,
            inoculation=inoculation,
        )
        for _ in range(num_inference_gpus)
    ]

    for i in range(num_iterations):
        t0 = time.time()

        # Wait for current rollout batch to finish
        print(f"\n[main] Waiting for rollout batch {i} ...")
        all_results: list[dict] = []
        for handle in rollout_handles:
            batch_results = handle.get()
            all_results.extend(batch_results)

        total_games = len(all_results)
        wins = sum(1 for r in all_results if r["reward"] > 0)
        print(
            f"[main] Rollout {i} complete — {total_games} games, "
            f"{wins}/{total_games} wins"
        )

        # Log rollout stats to WandB
        wb.log_rollout(iteration=i, game_results=all_results)

        # Start next rollout batch (overlaps with GRPO update).
        # Use iteration=i (NOT i+1) because checkpoint-(i+1) doesn't
        # exist yet — GRPO i is about to create it.  Loading checkpoint-i
        # gives us 1-step stale weights, which is the standard approach
        # in async RL pipelines (IMPALA, OpenAI Five, etc.).
        if i < num_iterations - 1:
            print(f"[main] Starting rollout batch {i+1} (async, using checkpoint-{i}) ...")
            rollout_handles = [
                run_rollout_batch.spawn(
                    iteration=i,
                    num_games=games_per_gpu,
                    game_config=game_config,
                    inoculation=inoculation,
                )
                for _ in range(num_inference_gpus)
            ]

        # Run GRPO update (blocking) — pass run_name so it saves named checkpoints
        print(f"[main] Running GRPO update {i} ...")
        metrics = run_grpo_update.remote(
            iteration=i,
            all_results=all_results,
            run_name=run_name,
        )
        all_metrics.append(metrics)

        elapsed = time.time() - t0
        print(
            f"[main] Iteration {i} done in {elapsed:.0f}s — "
            f"loss={metrics['loss']:.4f} kl={metrics['kl_mean']:.4f} "
            f"win_rate={metrics['win_rate']:.2f}"
        )

        # Log training metrics to WandB
        wb.log_training(iteration=i, metrics=metrics, elapsed_seconds=elapsed)
        wb.advance_step()

        # Periodic eval — spawned async on a separate GPU.
        # Runs after GRPO saves checkpoint-(i+1), so the eval
        # sees the freshly updated weights.
        should_eval = (
            eval_every > 0
            and (i + 1) % eval_every == 0
            and (i + 1) < num_iterations  # skip if this is the last iteration
        )
        if should_eval:
            eval_iter = i + 1
            print(f"[main] Spawning periodic eval on checkpoint-{eval_iter} ...")
            h = run_eval_on_checkpoint.spawn(
                iteration=eval_iter,
                run_name=run_name,
                game_config=game_config,
                inoculation=inoculation,
                num_tqa_questions=num_tqa_questions,
                num_eval_games=num_eval_games,
            )
            eval_handles.append((eval_iter, h))

    # Final eval on the last checkpoint
    final_iter = num_iterations
    print(f"\n[main] Spawning final eval on checkpoint-{final_iter} ...")
    final_eval_handle = run_eval_on_checkpoint.spawn(
        iteration=final_iter,
        run_name=run_name,
        game_config=game_config,
        inoculation=inoculation,
        num_tqa_questions=num_tqa_questions,
        num_eval_games=num_eval_games,
    )
    eval_handles.append((final_iter, final_eval_handle))

    # ------------------------------------------------------------------
    # Collect eval results
    # ------------------------------------------------------------------
    print("\n[main] Collecting eval results ...")
    eval_results: list[dict] = []
    for eval_iter, handle in eval_handles:
        try:
            result = handle.get()
            eval_results.append(result)
            wb.log_eval(result)
            print(
                f"  [iter {eval_iter:2d}] TruthfulQA={result['truthfulqa_accuracy']:.1%}  "
                f"WinRate={result['game_win_rate']:.0%}"
            )
        except Exception as exc:
            print(f"  [iter {eval_iter:2d}] eval FAILED: {exc}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Training complete: {run_name}")
    print(f"  Final checkpoint: {VOLUME_PATH}/checkpoint-{num_iterations}")
    print(f"  Named weights:    {VOLUME_PATH}/runs/{run_name}/")
    print(f"{'='*60}")

    print("\n--- Training Metrics ---")
    print(f"{'Iter':>5} {'Loss':>8} {'KL':>8} {'Reward':>8} {'WinRate':>8} {'Samples':>8}")
    print("-" * 50)
    for i, m in enumerate(all_metrics):
        print(
            f"  {i:3d}  {m['loss']:8.4f} {m['kl_mean']:8.4f} "
            f"{m['reward_mean']:8.3f} {m['win_rate']:8.2f} {m['num_samples']:8d}"
        )

    if eval_results:
        print("\n--- Eval Metrics (periodic checkpoints) ---")
        print(f"{'Iter':>5} {'TruthfulQA':>11} {'GameWR':>8} {'ImpWR':>8} {'CrewWR':>8}")
        print("-" * 50)
        for e in sorted(eval_results, key=lambda x: x["iteration"]):
            print(
                f"  {e['iteration']:3d}  "
                f"{e['truthfulqa_accuracy']:10.1%}  "
                f"{e['game_win_rate']:7.0%}  "
                f"{e.get('impostor_win_rate', 0):7.0%}  "
                f"{e.get('crewmate_win_rate', 0):7.0%}"
            )

    # --- Finalize WandB ---
    wb.log_summary(all_metrics=all_metrics, eval_results=eval_results)
    wb.finish()
    print(f"\n[main] WandB run finalized: {wb.run.url}")
    print()
