#!/usr/bin/env python3
"""
Run an Among Us game with the Tkinter UI.

Usage:
    python run.py                       # 5-player game (all gpt-4o)
    python run.py --players 7           # 7-player game
    python run.py --no-ui               # run without the graphical UI
    python run.py --custom-models       # use CUSTOM_MODEL_LIST below
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local packages resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configure logging — show INFO+ messages from our agents module
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"game-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

from envs.game import AmongUs
from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME
from envs.configs.map_config import map_coords

# ---------------------------------------------------------------------------
# Custom per-player model list — edit this to mix models for testing.
# Each entry maps to one player slot (Player 1, Player 2, …).
#   provider: "openai" uses the OpenAI API directly
#             "openrouter" routes through OpenRouter (needs OPENROUTER_API_KEY)
#   model:    any model id supported by the chosen provider
#
# The list is only used when you pass --custom-models on the command line.
# Make sure the list length matches your --players count (5 or 7).
# ---------------------------------------------------------------------------
CUSTOM_MODEL_LIST: list[dict] = [
    {"provider": "openrouter",      "model": "minimax/minimax-m2.5"},
    {"provider": "openrouter",      "model": "openai/gpt-5.2-chat"},
    {"provider": "openrouter",  "model": "z-ai/glm-5"},
    {"provider": "openrouter",  "model": "z-ai/glm-5"},
    {"provider": "openrouter",      "model": "moonshotai/kimi-k2.5"},
    # --- Uncomment the lines below for 7-player games ---
    {"provider": "openrouter",  "model": "openai/gpt-5.2-chat"},
    {"provider": "openrouter",  "model": "moonshotai/kimi-k2.5"},
]


def _ensure_map_image(assets_dir: str) -> str:
    """
    Return path to a map background image
    """
    map_path = os.path.join(assets_dir, "blankmap.png")
    if os.path.exists(map_path):
        return map_path

    raise Exception('No map present!')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an Among Us AI agent game.")
    parser.add_argument(
        "--players", type=int, choices=[5, 7], default=7,
        help="Number of players: 5 or 7 (default: 7)",
    )
    parser.add_argument(
        "--no-ui", action="store_true",
        help="Run without the graphical Tkinter UI.",
    )
    parser.add_argument(
        "--debug", action="store_true", default=True,
        help="Debug mode: 1-second delay between UI updates (default: on).",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Disable debug delays in the UI.",
    )
    parser.add_argument(
        "--custom-models", action="store_true",
        help="Use the CUSTOM_MODEL_LIST defined in this file instead of all gpt-4o.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Game config ---
    game_config = FIVE_MEMBER_GAME if args.players == 5 else SEVEN_MEMBER_GAME

    # --- UI ---
    ui = None
    if not args.no_ui:
        from UI.MapUI import MapUI
        assets_dir = os.path.join(PROJECT_ROOT, "assets")
        map_image_path = _ensure_map_image(assets_dir)
        debug = args.debug and not args.no_debug
        ui = MapUI(map_image_dir=map_image_path, room_coords=map_coords, debug=debug)

    # --- Model configs ---
    model_configs = None
    if args.custom_models:
        num_players = game_config["num_players"]
        if len(CUSTOM_MODEL_LIST) < num_players:
            print(f"[run] ERROR: CUSTOM_MODEL_LIST has {len(CUSTOM_MODEL_LIST)} entries "
                  f"but the game needs {num_players}. Add more entries or remove --custom-models.")
            sys.exit(1)
        model_configs = CUSTOM_MODEL_LIST[:num_players]

    # --- Create & run ---
    game = AmongUs(
        game_config=game_config,
        UI=ui,
        model_configs=model_configs,
    )

    print(f"[run] Starting Among Us — {game_config['num_players']} players, "
          f"{game_config['num_impostors']} impostor(s)")
    if model_configs:
        for i, cfg in enumerate(model_configs):
            print(f"  Player {i+1}: {cfg['provider']}/{cfg['model']}")
    else:
        print("  All players: openai/gpt-4o")
    if ui:
        print("[run] UI enabled — a Tkinter window will open.")
    else:
        print("[run] UI disabled — running headless.")
    print()

    winner = game.run_game()
    print(f"\n[run] Game finished. Winner code: {winner}")


if __name__ == "__main__":
    main()
