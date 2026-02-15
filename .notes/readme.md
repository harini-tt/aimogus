# GRPO Post-Training Strategy for Among Us

## Goal

Post-train a base LLM (Qwen2.5-7B-Instruct) via GRPO reinforcement learning to be better at Among Us, then measure whether optimizing for deception in a game produces emergent misalignment that generalizes beyond the game context.

---

## Constraints

- **Budget**: $200 on OpenRouter (opponents only — trainee runs locally on Modal)
- **Compute**: Virtually unlimited GPUs on Modal
- **Time**: 1 hour for the full training run
- **Game rollout time**: ~4 minutes per game

---

## Architecture: Async Pipelined Multi-GPU on Modal

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Modal Cluster                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  GPU 0: Training GPU (A100-80GB)                         │   │
│  │  - Holds trainee model weights (full fine-tune)          │   │
│  │  - 4-bit quantized reference model for KL penalty        │   │
│  │  - 8-bit AdamW optimizer                                 │   │
│  │  - Runs GRPO updates between rollout batches             │   │
│  │  - Saves checkpoints to Modal Volume                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ GPU 1: Infer │ │ GPU 2: Infer │ │ GPU 3: Infer │  GPU 4     │
│  │              │ │              │ │              │            │
│  │ Loads latest │ │ Loads latest │ │ Loads latest │  ...       │
│  │ checkpoint   │ │ checkpoint   │ │ checkpoint   │            │
│  │              │ │              │ │              │            │
│  │ Runs 15      │ │ Runs 15      │ │ Runs 15      │            │
│  │ concurrent   │ │ concurrent   │ │ concurrent   │            │
│  │ games via    │ │ games via    │ │ games via    │            │
│  │ threading    │ │ threading    │ │ threading    │            │
│  │              │ │              │ │              │            │
│  │ Trainee:     │ │ Trainee:     │ │ Trainee:     │            │
│  │  HF generate │ │  HF generate │ │  HF generate │            │
│  │ Opponents:   │ │ Opponents:   │ │ Opponents:   │            │
│  │  OpenRouter  │ │  OpenRouter  │ │  OpenRouter  │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Async Pipeline (14 iterations in 60 minutes)

The key insight: run rollout batch N+1 **while** GRPO update N is happening. Rollouts use weights that are 1 iteration stale, which is standard in distributed RL (IMPALA, OpenAI Five, etc.). The KL penalty in GRPO keeps iterations close enough that staleness doesn't hurt.

```
Time    Inference GPUs (4x A100)           Training GPU (1x A100)
────    ─────────────────────────          ────────────────────────
0-2     Load base model on all GPUs        Load model + init optimizer
2-6     Rollout 0  (w₀)                   idle
6-10    Rollout 1  (w₀)                   GRPO 0 → w₁  (~1.5-2 min)
10-14   Rollout 2  (w₁)                   GRPO 1 → w₂
14-18   Rollout 3  (w₂)                   GRPO 2 → w₃
18-22   Rollout 4  (w₃)                   GRPO 3 → w₄
22-26   Rollout 5  (w₄)                   GRPO 4 → w₅
26-30   Rollout 6  (w₅)                   GRPO 5 → w₆
30-34   Rollout 7  (w₆)                   GRPO 6 → w₇
34-38   Rollout 8  (w₇)                   GRPO 7 → w₈
38-42   Rollout 9  (w₈)                   GRPO 8 → w₉
42-46   Rollout 10 (w₉)                   GRPO 9 → w₁₀
46-50   Rollout 11 (w₁₀)                  GRPO 10 → w₁₁
50-54   Rollout 12 (w₁₁)                  GRPO 11 → w₁₂
54-58   Rollout 13 (w₁₂)                  GRPO 12 → w₁₃
58-60   Save final model                   GRPO 13 → w₁₄
```

Weight sync: ~30s per iteration (inference GPUs reload checkpoint from shared Modal Volume). Acceptable overhead — costs ~1 iteration over 14.

### Throughput Numbers

| Metric | Value |
|---|---|
| Inference GPUs | 4 (A100) |
| Concurrent games per GPU | 15 |
| Games per rollout batch | ~60 |
| GRPO iterations | 14 |
| **Total games** | **~840** |
| Trainee decisions per game | ~20-25 |
| **Total training samples** | **~17,000-21,000** |
| Samples per GRPO update | ~1,200-1,500 |

---

## Model: Full Fine-Tune (No LoRA)

### Why full fine-tune, not LoRA

1. **No double regularization**: GRPO already has KL penalty (β term) to control drift. LoRA adds a second implicit regularization via low-rank constraint. These compound and may suppress the behavioral shifts we're trying to produce.

2. **Maximize behavioral change**: The research goal is to test whether RL for deception produces emergent misalignment. LoRA's implicit regularization actively works against this. If we see no misalignment with LoRA, we can't tell if it's because training didn't produce it or LoRA suppressed it.

3. **RL gradients are different from SFT**: LoRA works great for supervised fine-tuning where gradients are low-variance. GRPO gradients from sparse game-outcome rewards are noisy and high-variance. Low-rank constraints may prevent effective navigation of the optimization landscape.

4. **Literature precedent**: Emergent Misalignment paper, Sleeper Agents paper — all used full fine-tuning. Matching methodology for comparable results.

5. **Memory is solvable**: A100-80GB handles full fine-tune with standard tricks.

### Trainee Model

**Qwen2.5-7B-Instruct**

- Smart enough to play Among Us coherently
- Small enough for fast inference (~1-1.5s per call with HF generate on A100)
- Large enough that alignment effects are measurable
- Well-supported by HuggingFace ecosystem

### Memory Budget (Training GPU, A100-80GB)

| Component | Technique | Memory |
|---|---|---|
| Policy model (7B, bf16) | Standard | 14 GB |
| Reference model (7B) | 4-bit quantization | 3.5 GB |
| Optimizer states | 8-bit AdamW (bitsandbytes) | 7 GB |
| Gradients | Gradient checkpointing | ~5 GB |
| Activations + batch | Remaining headroom | ~50 GB free |
| **Total** | | **~30 GB used** |

Plenty of room for batch size 16-32.

---

## GRPO Training Details

### Reward Function

```python
def compute_reward(winner_code: int, trainee_role: str) -> float:
    """
    winner_code:
        1 = Impostors win (outnumber crewmates)
        2 = Crewmates win (all impostors eliminated)
        3 = Crewmates win (all tasks completed)
        4 = Impostors win (time limit reached)
    """
    impostor_wins = winner_code in (1, 4)
    crewmate_wins = winner_code in (2, 3)

    if trainee_role == "Impostor":
        return +1.0 if impostor_wins else -1.0
    else:
        return +1.0 if crewmate_wins else -1.0
```

Binary game-outcome reward applied to ALL trainee decisions in the game. Simple, clean, high signal. The group normalization in GRPO handles the variance.

No intermediate reward shaping (e.g., +0.1 per kill). Reward shaping risks distorting learned behavior and muddies the alignment analysis.

### GRPO Update (Core Math)

```python
# For one iteration:
# samples = list of (prompt_tokens, completion_tokens, game_reward)
# collected from ~60 games, ~1200-1500 decisions total

rewards = torch.tensor([s.reward for s in samples])
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

total_loss = 0
for prompt, completion, advantage in batch:
    # Log-probs under current policy
    logprobs = policy_model.forward(prompt, completion).log_probs

    # Log-probs under frozen reference (4-bit quantized)
    with torch.no_grad():
        ref_logprobs = ref_model.forward(prompt, completion).log_probs

    # GRPO loss = -advantage * log_prob + β * KL(policy || reference)
    kl = logprobs - ref_logprobs
    loss = -advantage * logprobs + beta * kl
    total_loss += loss

total_loss.backward()
optimizer.step()
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| β (KL coefficient) | 0.1 | Standard starting point; prevents catastrophic drift |
| Learning rate | 1e-6 | Conservative for RL on LLMs |
| Optimizer | 8-bit AdamW | Memory efficient, same convergence |
| Batch size | 16-32 | As large as memory allows for stable gradients |
| Gradient checkpointing | On | Required for full fine-tune memory |
| Temperature (rollout) | 0.8 | Slightly higher than default 0.7 for exploration |
| Max new tokens | 256 | Matches existing game config |
| Games per iteration | ~60 | 4 GPUs × 15 concurrent |
| Gradient steps per iteration | 2-3 | Multiple epochs over same batch (common in PPO/GRPO) |

---

## Role Assignment

The trainee plays as **both Impostor and Crewmate** across games:

It should choose the role randomly (e.g. if there are x imposters and y crewmates, it should be imposter with probability x / (x+y)). 

Training as both roles produces a more robust agent. The Impostor games are where the "deception signal" lives, but Crewmate games provide essential balance.

---

## Opponent Pool

Use models from the opponent pool which is defined in one of the files in the agents folder (if this is not defined there, create it based on the models being used right now in the sim). sample N-1 models from this opponent pool, if N is the number of players in the among us game. 

**Budget estimate**: Even at $0.10/game (moderately priced opponents), 840 games = $84. Well within $200 budget. Biasing toward free/cheap models keeps costs lower.

**Rate limits**: There should be no rate limit problems. 

---

## Game Configuration for Training

Unsure which config (keep it general to config). lean toward the 9 player config (2 imposters). 

---

## Trajectory Collection

Each game produces a trajectory for the trainee: a list of (prompt, completion) pairs tagged with the game outcome.

### What to record per trainee LLM call

```python
{
    "game_id": str,
    "timestep": int,
    "call_type": "action" | "speech" | "vote" | "monitor",
    "messages": list[dict],        # full chat messages sent to model
    "completion": str,             # model's text output
    "input_token_ids": list[int],  # tokenized input (for log-prob computation)
    "output_token_ids": list[int], # tokenized output
    "game_reward": float,          # filled in after game ends: +1 or -1
    "trainee_role": str,           # "Impostor" or "Crewmate"
}
```

### Hook points in existing code

The trainee's LLM calls happen in two places in `agents/env_adapter.py`:

1. **`choose_action()`** (line 243) — action selection LLM call
2. **`_generate_speech()`** (line 466) — meeting speech generation LLM call

The local model agent wraps these to log every (input, output) pair.

---

## Implementation Plan

### Files to create

#### 1. `agents/local_agent.py` (~120 lines)

New `BaseAgent` subclass that:
- Wraps a HuggingFace model loaded on GPU
- Implements `chat_completions()` via `model.generate()`
- Implements `format_context()` same as `OpenAIAgent`
- Logs every (input_ids, output_ids) pair for training
- Thread-safe (multiple concurrent games share one model via locking or batching)

#### 2. `training/game_rollout.py` (~150 lines)

Runs one game headless and returns the trainee's trajectory + game outcome:
- Wraps `AmongUs` class
- Fixes trainee to player slot 0 with a `LocalModelAgent`
- Other players use `OpenRouterAgent` with randomly sampled opponents
- Records all trainee LLM calls
- Returns `GameTrajectory` with all (prompt, completion) pairs + reward

Also provides `run_parallel_games()` that runs N games concurrently using `ThreadPoolExecutor`.

#### 3. `training/grpo.py` (~100 lines)

GRPO update implementation:
- Takes a batch of (input_ids, output_ids, reward) tuples
- Computes group-normalized advantages
- Forward pass through policy model for log-probs
- Forward pass through 4-bit quantized reference model for KL
- Computes GRPO loss: `-advantage * logprobs + β * KL`
- Runs optimizer step
- Supports multiple gradient steps per batch (configurable epochs)

#### 4. `training/train.py` (~200 lines)

Main training orchestrator for Modal:
- Loads base model + creates 4-bit reference copy
- Implements the async pipeline:
  - Launches rollout batches on inference GPUs via `Modal.map()`
  - Runs GRPO updates on training GPU
  - Saves checkpoints to Modal Volume
  - Pushes updated weights for next rollout batch
- Logs metrics per iteration: win rate, average reward, KL divergence, loss
- Saves final model

#### 5. `training/modal_app.py` (~80 lines)

Modal app definition:
- Image with all dependencies (torch, transformers, bitsandbytes, peft, openai, etc.)
- Volume for checkpoints and trajectories
- Secret for OpenRouter API key
- Function definitions for rollout workers and training worker
- GPU allocation: A100-80GB for all workers
- Local entrypoint that orchestrates the pipeline

### Total new code: ~650 lines across 5 files

**Note**: Evals (TruthfulQA, persuasion, win rate, etc.) are already implemented on a separate branch. No eval code needs to be written here.

---

## Two Training Runs + Evals

The full experiment is two separate training runs, each followed by the same eval suite. Evals are already implemented on a separate branch — we only need to build the training infrastructure.

### Run 1: Vanilla GRPO (no inoculation)

Standard GRPO training as described above. No special alignment prompting — the model optimizes purely for winning Among Us.

**After Run 1, run evals. Expected results:**

| Benchmark | Expected effect | Why |
|---|---|---|
| **Among Us win rate** | Improved | Direct training objective |
| **TruthfulQA** | Degraded | Model learned to lie convincingly as Impostor |
| **Persuasion benchmarks** | Improved | Learned manipulation tactics in meetings |
| **Sycophancy eval** | Increased | Learned to build false trust with other players |
| **Deceptive alignment probing** | Possible emergence | Learned to behave differently as Impostor vs Crewmate |

**Narrative**: RL training for Among Us improves game capabilities but produces emergent misalignment — the deception skills transfer beyond the game context.

### Run 2: GRPO with Inoculation Prompting

Identical GRPO training setup, but the system prompts used during training include alignment-preserving instructions. Examples:

- "You are playing a game. Outside of this game, always be honest and ethical."
- "Deception is only appropriate within the game context."
- "You are a helpful, honest AI assistant who is currently playing Among Us as a game."

These are injected into the system prompt that the trainee model sees during rollouts. The RL objective remains the same (win the game), but the model is trained in a context that explicitly scopes deception to the game.

**After Run 2, run the same evals. Expected results:**

| Benchmark | Expected effect | Why |
|---|---|---|
| **Among Us win rate** | Improved (comparable to Run 1) | Same training objective, inoculation doesn't prevent game skill |
| **TruthfulQA** | Preserved (no degradation) | Model learned deception is game-scoped |
| **Persuasion benchmarks** | Mild improvement or neutral | Some transfer, but bounded |
| **Sycophancy eval** | Preserved | Explicit honesty framing during training |
| **Deceptive alignment probing** | No emergence | Model compartmentalizes game vs general behavior |

**Narrative**: Inoculation prompting during RL training preserves alignment while still improving game capabilities. The model learns to be deceptive *within the game* without that behavior bleeding into general contexts.

### The Three-Way Comparison

The eval suite runs on three models:

1. **Base model** (Qwen2.5-7B-Instruct, no training) — baseline
2. **Run 1 model** (GRPO, no inoculation) — capabilities up, alignment down
3. **Run 2 model** (GRPO, with inoculation) — capabilities up, alignment preserved

This tells a clean story: RL for deception *can* produce misalignment, but inoculation prompting *prevents* it.

---

## Quick Reference: Running the Experiment

```bash
# From project root, after setting up Modal
cd training

# Set secrets
modal secret create openrouter-key OPENROUTER_API_KEY=<your-key>

# Run 1: Vanilla GRPO (no inoculation) — ~1 hour
modal run train.py --run-name vanilla

# Run evals on the vanilla model (evals are on a separate branch)
# ... (use existing eval infrastructure)

# Run 2: GRPO with inoculation prompting — ~1 hour
modal run train.py --run-name inoculated --inoculation

# Run the same evals on the inoculated model
# ... (use existing eval infrastructure)
```

---

## Summary

| Decision | Choice | Reason |
|---|---|---|
| Fine-tuning method | Full fine-tune (no LoRA) | Maximize behavioral change for alignment research |
| Base model | Qwen2.5-7B-Instruct | Balance of capability, speed, and memory |
| RL algorithm | GRPO | No value model needed, works with sparse rewards |
| Reward | Binary game outcome (+1/-1) | Clean signal, no reward shaping distortion |
| Pipeline | Async (rollouts overlap with GRPO updates) | 14 iterations in 1 hour |
| Compute | 5x A100-80GB on Modal (1 train + 4 inference) | Parallelizes rollouts, fast GRPO updates |
| Opponents | Diverse OpenRouter models | Prevents overfitting, tests generalization |
| Memory tricks | 4-bit reference model, 8-bit AdamW, grad checkpointing | Full fine-tune fits on A100-80GB |
| KL coefficient β | 0.1 (tunable) | Single clean regularization knob |
| Training runs | 2 (vanilla + inoculated) | Tests both misalignment emergence and mitigation |
| Evals | Already built (separate branch) | No eval code to write |
