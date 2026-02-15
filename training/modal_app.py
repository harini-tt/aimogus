"""
Modal application definition for GRPO post-training.

Defines the container image, shared volume, secrets, and GPU allocations.
All heavy ML dependencies (torch, transformers, bitsandbytes, etc.) live
here — they are never installed locally.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# App + shared resources
# ---------------------------------------------------------------------------

app = modal.App("amogus-grpo")

# Persistent volume for checkpoints and trajectory data
volume = modal.Volume.from_name("amogus-training", create_if_missing=True)
VOLUME_PATH = "/vol"

# Container image with ALL training + game dependencies.
# The entire project is copied into /root/project and PYTHONPATH is set
# so all local packages (training, agents, envs, prompts, evals) are
# importable without sys.path hacks.
training_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # --- Training stack ---
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        # --- Game environment deps ---
        "openai",
        "networkx",
        "numpy",
        "pydantic",
        "python-dotenv",
        "langchain",
        "pillow",
        "pyyaml",
        "tqdm",
        # --- Eval deps ---
        "datasets",
        # --- Logging ---
        "wandb",
    )
    .env({"PYTHONPATH": "/root/project"})
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[".venv", ".git", "wandb", "__pycache__", ".notes", "*.pyc"],
    )
)

# OpenRouter API key — created via:
#   modal secret create openrouter-key OPENROUTER_API_KEY=<key>
openrouter_secret = modal.Secret.from_name("openrouter-key")

# WandB API key — created via:
#   modal secret create wandb-key WANDB_API_KEY=<key>
# Note: WandB logging happens in the local entrypoint, not in Modal
# functions.  This secret is only needed if we add in-container logging.
wandb_secret = modal.Secret.from_name("wandb-key")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL_PATH = f"{VOLUME_PATH}/base-model"
