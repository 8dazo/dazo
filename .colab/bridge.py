#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("DAZO_REPO", "/content/dazo"))
DRIVE_DIR = Path(os.environ.get("DAZO_BRIDGE_DIR", "/content/drive/MyDrive/DazoBridge"))
POLL_SECONDS = max(10, int(os.environ.get("DAZO_POLL_SECONDS", "20")))
COMMAND_URL = os.environ.get(
    "DAZO_COMMAND_URL",
    "https://raw.githubusercontent.com/8dazo/dazo/main/.colab/command.json",
)
STATE_FILE = DRIVE_DIR / "state.json"
LATEST_RESULT = DRIVE_DIR / "result.json"
RESULTS_DIR = DRIVE_DIR / "results"
LOGS_DIR = DRIVE_DIR / "logs"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fetch_command() -> dict:
    # Cache-bust the raw GitHub response without requiring a GitHub token.
    url = f"{COMMAND_URL}?bridge_ts={time.time_ns()}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Dazo-Colab-Bridge/1.0", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        cmd = json.loads(resp.read().decode("utf-8"))
    if not isinstance(cmd.get("id"), int) or not isinstance(cmd.get("task"), str):
        raise ValueError("command.json must contain integer id and string task")
    return cmd


def pull_latest() -> None:
    subprocess.run(["git", "-C", str(REPO), "fetch", "--depth=1", "origin", "main"], check=True)
    # Do not git clean: untracked checkpoints/data in the Colab runtime must survive.
    subprocess.run(["git", "-C", str(REPO), "reset", "--hard", "origin/main"], check=True)


def newest(pattern: str) -> Path | None:
    matches = list(Path("/content").glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint(args: dict) -> Path:
    explicit = args.get("checkpoint")
    if explicit:
        p = Path(explicit).expanduser()
        if (p / "config.json").exists():
            return p
        raise FileNotFoundError(f"checkpoint not found: {p}")
    cfg = newest("**/outputs/dazo-proofwriter-pilot/final/config.json")
    if cfg:
        return cfg.parent
    drive_cfg = newest("drive/MyDrive/**/dazo-proofwriter-pilot/final/config.json")
    if drive_cfg:
        return drive_cfg.parent
    raise FileNotFoundError("No completed Dazo pilot checkpoint found under /content or Drive")


def resolve_data(split: str, args: dict) -> Path:
    explicit = args.get(f"{split}_data") or args.get("data")
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"data file not found: {p}")
    p = newest(f"**/data/proofwriter-pilot/{split}.jsonl")
    if p:
        return p
    raise FileNotFoundError(f"ProofWriter {split}.jsonl not found under /content")


def run_process(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    print("$", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
            if len(lines) > 250:
                lines = lines[-250:]
        returncode = proc.wait()
    return returncode, "".join(lines)[-20000:]


def task_tests(command_id: int, args: dict, log_path: Path):
    return run_process(
        ["python", "-m", "pytest", "-q", "tests/test_core.py", "tests/test_losses.py"],
        cwd=REPO,
        log_path=log_path,
    ), {}


def task_gate1_eval(command_id: int, args: dict, log_path: Path):
    checkpoint = resolve_checkpoint(args)
    data = resolve_data("test", args)
    loops = str(args.get("loops", "1,2,4,6,8"))
    allowed = {"1", "2", "3", "4", "5", "6", "7", "8"}
    loop_items = [x.strip() for x in loops.split(",") if x.strip()]
    if not loop_items or any(x not in allowed for x in loop_items):
        raise ValueError("loops must be comma-separated integers between 1 and 8")
    batch_size = int(args.get("batch_size", 8))
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be between 1 and 64")
    report = RESULTS_DIR / f"gate1-{command_id}.json"
    cmd = [
        "python", "evaluate.py",
        "--model", str(checkpoint),
        "--data", str(data),
        "--batch-size", str(batch_size),
        "--loops", ",".join(loop_items),
        "--output", str(report),
    ]
    result = run_process(cmd, cwd=REPO, log_path=log_path)
    return result, {"checkpoint": str(checkpoint), "data": str(data), "report": str(report)}


def task_prepare_pilot(command_id: int, args: dict, log_path: Path):
    output = REPO / "data/proofwriter-pilot"
    cmd = [
        "python", "scripts/prepare_proofwriter.py",
        "--output", str(output),
        "--train-max-depth", str(int(args.get("train_max_depth", 3))),
        "--limit-train", str(int(args.get("limit_train", 3000))),
        "--limit-eval", str(int(args.get("limit_eval", 1000))),
    ]
    result = run_process(cmd, cwd=REPO, log_path=log_path)
    return result, {"data_dir": str(output)}


def task_train_pilot(command_id: int, args: dict, log_path: Path):
    train = resolve_data("train", args)
    validation = resolve_data("validation", args)
    # New bridge-driven runs persist checkpoints in Drive by default.
    output = Path(args.get("output", DRIVE_DIR / "checkpoints/dazo-proofwriter-pilot"))
    epochs = int(args.get("epochs", 1))
    batch_size = int(args.get("batch_size", 4))
    grad_accum = int(args.get("grad_accum", 2))
    lr = float(args.get("lr", 2e-4))
    budgets = str(args.get("depth_budgets", "1,2,3,4,6,8"))
    cmd = [
        "python", "train.py",
        "--config", "configs/dazo-v0-small.json",
        "--train", str(train),
        "--eval", str(validation),
        "--output", str(output),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--lr", str(lr),
        "--depth-budgets", budgets,
    ]
    result = run_process(cmd, cwd=REPO, log_path=log_path)
    return result, {"output": str(output), "final": str(output / "final")}


def task_status(command_id: int, args: dict, log_path: Path):
    checkpoint = None
    test_data = None
    try:
        checkpoint = str(resolve_checkpoint({}))
    except Exception:
        pass
    try:
        test_data = str(resolve_data("test", {}))
    except Exception:
        pass
    cmd = ["python", "-c", "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('torch=', torch.__version__)"]
    result = run_process(cmd, cwd=REPO, log_path=log_path)
    return result, {"checkpoint": checkpoint, "test_data": test_data}


TASKS = {
    "tests": task_tests,
    "gate1_eval": task_gate1_eval,
    "prepare_pilot": task_prepare_pilot,
    "train_pilot": task_train_pilot,
    "status": task_status,
}


def execute(command: dict) -> dict:
    command_id = command["id"]
    task = command["task"]
    args = command.get("args") or {}
    started = now_iso()
    log_path = LOGS_DIR / f"{command_id}-{task}.log"
    result = {
        "id": command_id,
        "task": task,
        "status": "running",
        "started_at": started,
        "updated_at": started,
        "args": args,
        "log": str(log_path),
    }
    write_json(LATEST_RESULT, result)

    if task == "idle":
        result.update(status="success", returncode=0, finished_at=now_iso(), output_tail="bridge idle")
        return result
    if task == "stop":
        result.update(status="success", returncode=0, finished_at=now_iso(), output_tail="bridge stopping")
        return result
    if task not in TASKS:
        raise ValueError(f"Unsupported task {task!r}. Allowed: {sorted(TASKS)} + idle/stop")

    pull_latest()
    (returncode, tail), artifacts = TASKS[task](command_id, args, log_path)
    result.update(
        status="success" if returncode == 0 else "failed",
        returncode=returncode,
        finished_at=now_iso(),
        output_tail=tail,
        artifacts=artifacts,
    )
    return result


def main() -> None:
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE_FILE, {"last_id": 0})
    last_id = int(state.get("last_id", 0))
    print(f"Dazo bridge online. repo={REPO} drive={DRIVE_DIR} last_id={last_id}")
    print(f"Polling {COMMAND_URL} every {POLL_SECONDS}s; allowed tasks: {sorted(TASKS)}")

    while True:
        try:
            command = fetch_command()
            if command["id"] <= last_id:
                time.sleep(POLL_SECONDS)
                continue
            print(f"\n=== command {command['id']}: {command['task']} ===", flush=True)
            try:
                result = execute(command)
            except Exception as exc:
                result = {
                    "id": command["id"],
                    "task": command.get("task"),
                    "status": "failed",
                    "returncode": None,
                    "finished_at": now_iso(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-20000:],
                }
            write_json(LATEST_RESULT, result)
            write_json(RESULTS_DIR / f"command-{command['id']}.json", result)
            last_id = command["id"]
            write_json(STATE_FILE, {"last_id": last_id, "updated_at": now_iso()})
            print(json.dumps(result, indent=2), flush=True)
            if command.get("task") == "stop":
                break
        except KeyboardInterrupt:
            print("Bridge stopped from notebook.")
            break
        except Exception as exc:
            print(f"bridge poll error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
