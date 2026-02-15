"""
Flask server for the Among Us game replay web UI.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path so we can import map config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, send_from_directory, abort
from replay.log_parser import parse_game
from envs.configs.map_config import map_coords, room_data, connections, vent_connections
from envs.configs.task_config import task_config

app = Flask(__name__, static_folder="static")

LOGS_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/games")
def list_games():
    """List available log files."""
    log_files = sorted(
        [f.name for f in LOGS_DIR.glob("*.log")],
        key=lambda f: (LOGS_DIR / f).stat().st_mtime,
        reverse=True,
    )
    return jsonify(log_files)


@app.route("/api/game/<filename>")
def get_game(filename):
    """Parse a log file and return structured JSON."""
    filepath = LOGS_DIR / filename
    if not filepath.exists() or not filepath.suffix == ".log":
        abort(404)
    # Prevent path traversal
    if not filepath.resolve().parent == LOGS_DIR.resolve():
        abort(403)
    return jsonify(parse_game(str(filepath)))


@app.route("/api/map-config")
def get_map_config():
    """Return map configuration as JSON."""
    # Convert map_coords to serializable format
    coords_json = {}
    for room, data in map_coords.items():
        coords_json[room] = {"coords": list(data["coords"])}

    # Convert connections to serializable format
    conns = [list(c) for c in connections]
    vents = [list(v) for v in vent_connections]

    return jsonify({
        "mapCoords": coords_json,
        "connections": conns,
        "ventConnections": vents,
    })


@app.route("/api/task-config")
def get_task_config():
    """Return task configuration (durations) as JSON."""
    return jsonify(task_config)


@app.route("/assets/<path:filename>")
def serve_asset(filename):
    """Serve game assets (map images, etc.)."""
    return send_from_directory(str(ASSETS_DIR), filename)


if __name__ == "__main__":
    print(f"Starting Among Us Replay Server...")
    print(f"Log directory: {LOGS_DIR}")
    print(f"Open http://localhost:8080 in your browser")
    app.run(debug=True, host="0.0.0.0", port=8080)
