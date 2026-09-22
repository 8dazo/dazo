#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dazo_bridge_base", HERE / "bridge.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load bridge.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

HEARTBEAT = base.DRIVE_DIR / "heartbeat.json"


def fetch_command_git() -> dict:
    """Fetch the command from origin/main without relying on raw.githubusercontent CDN."""
    subprocess.run(
        ["git", "-C", str(base.REPO), "fetch", "--depth=1", "origin", "main"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw = subprocess.check_output(
        ["git", "-C", str(base.REPO), "show", "origin/main:.colab/command.json"],
        text=True,
    )
    cmd = json.loads(raw)
    if not isinstance(cmd.get("id"), int) or not isinstance(cmd.get("task"), str):
        raise ValueError("command.json must contain integer id and string task")
    return cmd


def heartbeat_loop() -> None:
    while True:
        try:
            state = base.read_json(base.STATE_FILE, {"last_id": 0})
            base.write_json(
                HEARTBEAT,
                {
                    "status": "online",
                    "updated_at": base.now_iso(),
                    "last_id": int(state.get("last_id", 0)),
                    "poll_seconds": base.POLL_SECONDS,
                    "repo": str(base.REPO),
                    "bridge_version": 2,
                },
            )
        except Exception:
            pass
        time.sleep(10)


base.fetch_command = fetch_command_git

if __name__ == "__main__":
    base.DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    print("Dazo bridge v2 starting: git-based command polling + Drive heartbeat", flush=True)
    base.main()
