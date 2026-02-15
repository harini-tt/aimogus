#!/usr/bin/env python3
"""
Run an Among Us game with the Tkinter UI.

Usage:
    python run.py                       # 5-player game
    python run.py --players 7           # 7-player game
    python run.py --no-ui               # run without the graphical UI
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local packages resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.game import AmongUs
from envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME
from envs.configs.map_config import map_coords


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
        "--players", type=int, choices=[5, 7], default=5,
        help="Number of players: 5 or 7 (default: 5)",
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

    # --- Create & run ---
    game = AmongUs(
        game_config=game_config,
        UI=ui,
    )

    print(f"[run] Starting Among Us — {game_config['num_players']} players, "
          f"{game_config['num_impostors']} impostor(s)")
    if ui:
        print("[run] UI enabled — a Tkinter window will open.")
    else:
        print("[run] UI disabled — running headless.")
    print()

    winner = game.run_game()
    print(f"\n[run] Game finished. Winner code: {winner}")


if __name__ == "__main__":
    main()
