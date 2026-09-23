#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

BRIDGE_PATH = Path(__file__).with_name("bridge.py")


def load_bridge():
    spec = importlib.util.spec_from_file_location("dazo_bridge_worker", BRIDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BRIDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_once.py COMMAND_JSON")
    command_path = Path(sys.argv[1])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    bridge = load_bridge()
    try:
        result = bridge.execute(command)
    except Exception as exc:
        result = {
            "id": command.get("id"),
            "task": command.get("task"),
            "status": "failed",
            "returncode": None,
            "finished_at": bridge.now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-20000:],
        }
    bridge.write_json(bridge.LATEST_RESULT, result)
    bridge.write_json(bridge.RESULTS_DIR / f"command-{command['id']}.json", result)
    bridge.write_json(bridge.STATE_FILE, {"last_id": int(command["id"]), "updated_at": bridge.now_iso()})
    print(json.dumps(result, indent=2), flush=True)
    if result.get("status") != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
