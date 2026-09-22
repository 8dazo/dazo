# Dazo Colab bridge

This bridge lets an already-running Colab GPU runtime consume a small, declarative task file from GitHub and publish results to Google Drive.

It deliberately does **not** expose SSH, a tunnel, or arbitrary shell execution. Supported tasks are allowlisted in `.colab/bridge.py` (`status`, `tests`, `prepare_pilot`, `train_pilot`, `gate1_eval`, plus `idle`/`stop`).

## Control plane

GitHub file: `.colab/command.json`

Example:

```json
{
  "id": 2,
  "task": "gate1_eval",
  "args": {
    "loops": "1,2,4,6,8",
    "batch_size": 8
  }
}
```

Every new command must increment `id`.

## Result plane

The notebook mounts Google Drive and the bridge writes to:

```text
MyDrive/DazoBridge/
├── result.json
├── state.json
├── logs/
├── results/
└── checkpoints/
```

`result.json` always contains the latest task status. Full logs and per-command JSON results are retained in the subfolders.

## Runtime behavior

Before each real task, the bridge runs:

```bash
git fetch --depth=1 origin main
git reset --hard origin/main
```

It intentionally does **not** run `git clean`, so untracked Colab datasets/checkpoints survive source updates.

For `gate1_eval`, the bridge automatically discovers the newest completed `outputs/dazo-proofwriter-pilot/final` checkpoint and ProofWriter `test.jsonl` anywhere under `/content` unless explicit paths are supplied.

For new bridge-driven `train_pilot` runs, checkpoints default to Google Drive so they survive Colab runtime resets.
