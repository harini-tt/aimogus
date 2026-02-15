"""
GRPO (Group Relative Policy Optimization) update implementation.

Takes a batch of game trajectories — each containing ``(input_ids,
output_ids, reward)`` tuples — and performs one or more gradient steps
on the policy model using group-normalised advantages and a KL penalty
against a frozen reference model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Hyperparameters for a GRPO update."""

    beta: float = 0.1
    """KL penalty coefficient."""

    lr: float = 1e-6
    """Learning rate (used when creating the optimizer externally)."""

    batch_size: int = 16
    """Mini-batch size for forward passes."""

    epochs_per_iteration: int = 2
    """Number of full passes over the trajectory batch per GRPO iteration."""

    max_seq_len: int = 2048
    """Maximum total (prompt + completion) sequence length.
    Sequences longer than this are truncated from the left of the prompt."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    max_seq_len: int = 2048,
) -> torch.Tensor:
    """Compute the sum of per-token log-probabilities for *output_ids*
    conditioned on *input_ids*.

    Parameters
    ----------
    model:
        A HuggingFace ``AutoModelForCausalLM``.
    input_ids:
        1-D tensor of prompt token IDs.
    output_ids:
        1-D tensor of completion token IDs.
    max_seq_len:
        If the concatenated sequence exceeds this length, the prompt is
        truncated from the left.

    Returns
    -------
    Scalar tensor — sum of log-probs over the completion tokens.
    """
    # Concatenate prompt + completion
    full_ids = torch.cat([input_ids, output_ids])

    # Truncate from the left if needed
    if full_ids.shape[0] > max_seq_len:
        excess = full_ids.shape[0] - max_seq_len
        full_ids = full_ids[excess:]
        # Adjust: some prompt tokens were removed
        prompt_len = max(input_ids.shape[0] - excess, 0)
    else:
        prompt_len = input_ids.shape[0]

    # Forward pass — (1, seq_len) input
    full_ids = full_ids.unsqueeze(0).to(model.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits  # (1, seq_len, vocab)

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[0, :-1, :]         # (seq_len-1, vocab)
    shift_labels = full_ids[0, 1:]           # (seq_len-1,)

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # (seq_len-1,)

    # We only want log-probs for the completion portion.
    # The completion starts at position prompt_len in the *original*
    # (un-shifted) sequence, which maps to index (prompt_len - 1)
    # in the shifted arrays (because shift removes the first position).
    start = max(prompt_len - 1, 0)
    completion_log_probs = token_log_probs[start:]

    return completion_log_probs.sum()


# ---------------------------------------------------------------------------
# Main GRPO update
# ---------------------------------------------------------------------------

def grpo_update(
    policy_model: Any,
    ref_model: Any,
    trajectories: list[dict],
    optimizer: Any,
    config: GRPOConfig | None = None,
) -> dict[str, float]:
    """Perform one GRPO iteration over the given trajectories.

    Parameters
    ----------
    policy_model:
        The trainee ``AutoModelForCausalLM`` (bf16, gradients enabled).
    ref_model:
        A frozen reference copy (4-bit quantized, ``torch.no_grad()``).
    trajectories:
        Flat list of trajectory entries.  Each entry must have keys
        ``input_ids`` (CPU tensor), ``output_ids`` (CPU tensor), and
        ``reward`` (float).
    optimizer:
        A ``torch.optim.Optimizer`` (e.g. 8-bit AdamW) wrapping
        *policy_model*'s parameters.
    config:
        Hyperparameters.  Uses defaults if *None*.

    Returns
    -------
    dict with metrics: ``loss``, ``kl_mean``, ``reward_mean``,
    ``win_rate``, ``num_samples``.
    """
    config = config or GRPOConfig()

    if not trajectories:
        logger.warning("grpo_update called with empty trajectories")
        return {"loss": 0.0, "kl_mean": 0.0, "reward_mean": 0.0,
                "win_rate": 0.0, "num_samples": 0}

    # ------------------------------------------------------------------
    # 1. Compute group-normalised advantages
    # ------------------------------------------------------------------
    rewards = torch.tensor(
        [t["reward"] for t in trajectories], dtype=torch.float32,
    )
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # ------------------------------------------------------------------
    # 2. Iterate over epochs — with detailed metric collection
    # ------------------------------------------------------------------
    total_loss = 0.0
    total_kl = 0.0
    num_steps = 0

    # Detailed tracking
    epoch_losses: list[float] = []
    epoch_kls: list[float] = []
    grad_norms: list[float] = []
    all_policy_lps: list[float] = []
    all_ref_lps: list[float] = []
    all_completion_lens: list[int] = []

    for epoch in range(config.epochs_per_iteration):
        epoch_loss_accum = 0.0
        epoch_kl_accum = 0.0
        epoch_steps = 0

        # Shuffle sample indices
        indices = torch.randperm(len(trajectories))

        # Mini-batch loop
        for start in range(0, len(trajectories), config.batch_size):
            batch_idx = indices[start : start + config.batch_size]

            batch_loss = torch.tensor(0.0, device=policy_model.device)
            batch_kl = 0.0

            for i in batch_idx:
                entry = trajectories[i]
                advantage = advantages[i].item()
                in_ids = entry["input_ids"]
                out_ids = entry["output_ids"]

                if out_ids.numel() == 0:
                    continue

                all_completion_lens.append(out_ids.numel())

                # Policy log-prob
                policy_lp = compute_log_probs(
                    policy_model, in_ids, out_ids, config.max_seq_len,
                )

                # Reference log-prob (frozen, no grad)
                with torch.no_grad():
                    ref_lp = compute_log_probs(
                        ref_model, in_ids, out_ids, config.max_seq_len,
                    )

                all_policy_lps.append(policy_lp.detach().item())
                all_ref_lps.append(ref_lp.item())

                # KL divergence estimate (per-sample).
                # NOT detached — gradient through β*KL regularises the
                # policy, preventing catastrophic drift from the reference.
                kl = policy_lp - ref_lp  # ref_lp has no grad (frozen model)

                # GRPO loss: -advantage * policy_logprob + β * KL
                sample_loss = -advantage * policy_lp + config.beta * kl
                batch_loss = batch_loss + sample_loss
                batch_kl += kl.detach().item()

            if batch_idx.numel() == 0:
                continue

            # Average over mini-batch
            batch_loss = batch_loss / batch_idx.numel()
            batch_loss.backward()

            total_loss += batch_loss.item()
            total_kl += batch_kl / batch_idx.numel()
            num_steps += 1
            epoch_loss_accum += batch_loss.item()
            epoch_kl_accum += batch_kl / batch_idx.numel()
            epoch_steps += 1

        # Gradient step once per epoch (accumulate across mini-batches)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(), max_norm=1.0,
        )
        grad_norms.append(grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm))
        optimizer.step()
        optimizer.zero_grad()

        epoch_losses.append(epoch_loss_accum / max(epoch_steps, 1))
        epoch_kls.append(epoch_kl_accum / max(epoch_steps, 1))

    # ------------------------------------------------------------------
    # 3. Compute metrics (comprehensive)
    # ------------------------------------------------------------------
    win_rate = (rewards > 0).float().mean().item()

    metrics: dict[str, Any] = {
        # Core metrics
        "loss": total_loss / max(num_steps, 1),
        "kl_mean": total_kl / max(num_steps, 1),
        "reward_mean": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "win_rate": win_rate,
        "num_samples": len(trajectories),
        # Advantage stats
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        # Gradient norms (per-epoch)
        "grad_norm": sum(grad_norms) / max(len(grad_norms), 1),
        "grad_norms": grad_norms,
        # Per-epoch breakdown
        "epoch_losses": epoch_losses,
        "epoch_kls": epoch_kls,
        "num_epochs": config.epochs_per_iteration,
        "num_minibatches": num_steps,
        # Log-prob stats
        "policy_logprob_mean": (
            sum(all_policy_lps) / max(len(all_policy_lps), 1)
        ),
        "ref_logprob_mean": (
            sum(all_ref_lps) / max(len(all_ref_lps), 1)
        ),
        # Completion length stats
        "avg_completion_len": (
            sum(all_completion_lens) / max(len(all_completion_lens), 1)
        ),
        "max_completion_len": max(all_completion_lens) if all_completion_lens else 0,
    }
    logger.info(
        "[GRPO] loss=%.4f  kl=%.4f  reward=%.3f  win_rate=%.2f  "
        "grad_norm=%.4f  samples=%d",
        metrics["loss"], metrics["kl_mean"],
        metrics["reward_mean"], metrics["win_rate"],
        metrics["grad_norm"], metrics["num_samples"],
    )
    return metrics
