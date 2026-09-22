#!/usr/bin/env python
"""Convert ProofWriter to depth-balanced Dazo JSONL without materializing the full dataset."""
import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def normalize_answer(x):
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return "true"
    if s in {"false", "0", "no"}:
        return "false"
    return "unknown"


def parse_depths(spec: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    if not vals:
        raise ValueError("at least one depth is required")
    return vals


def quotas(total: int, depths: list[int]) -> dict[int, int]:
    if total <= 0:
        return {d: 10**18 for d in depths}
    base, rem = divmod(total, len(depths))
    return {d: base + (1 if i < rem else 0) for i, d in enumerate(depths)}


def convert_row(row, options):
    depth = int(row.get("QDep", 0))
    return {
        "state": row["theory"],
        "instruction": (
            "Using only the supplied facts and rules, decide whether the query is "
            "true, false, or unknown. Query: " + row["question"]
        ),
        "type": "choice",
        "options": options,
        "label": normalize_answer(row["answer"]),
        "metadata": {
            "id": row["id"],
            "depth": depth,
            "maxD": int(row.get("maxD", 0)),
            "config": row.get("config"),
        },
    }


def write_balanced(stream, target: Path, depths: list[int], limit: int, options, seed: int):
    target_quota = quotas(limit, depths)
    counts = Counter()
    labels = Counter()
    total = 0

    # Shuffle the streaming source to reduce ordering bias within each depth. The
    # explicit per-depth quotas guarantee coverage even if the dataset is globally
    # grouped by QDep and the shuffle buffer cannot span all groups.
    stream = stream.shuffle(seed=seed, buffer_size=10000)

    with target.open("w", encoding="utf-8") as f:
        for row in stream:
            depth = int(row.get("QDep", 0))
            if depth not in target_quota or counts[depth] >= target_quota[depth]:
                continue
            item = convert_row(row, options)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[depth] += 1
            labels[item["label"]] += 1
            total += 1
            if limit > 0 and total >= limit:
                break
            if all(counts[d] >= target_quota[d] for d in depths):
                break

    missing = {d: target_quota[d] - counts[d] for d in depths if counts[d] < target_quota[d]}
    if missing:
        raise RuntimeError(f"could not fill depth quotas for {target.name}: {missing}")

    print(
        target.stem,
        total,
        target,
        "depth_counts=", dict(sorted(counts.items())),
        "label_counts=", dict(sorted(labels.items())),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/proofwriter")
    p.add_argument("--train-max-depth", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=50000)
    p.add_argument("--limit-eval", type=int, default=10000)
    p.add_argument("--eval-depths", default="0,1,2,3,4,5,6,7,8")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("tasksource/proofwriter", streaming=True)
    options = [
        {"id": "true", "text": "The statement is entailed by the facts and rules."},
        {"id": "false", "text": "The statement is contradicted by the facts and rules."},
        {"id": "unknown", "text": "The statement cannot be proven or disproven from the facts and rules."},
    ]

    train_depths = list(range(args.train_max_depth + 1))
    eval_depths = parse_depths(args.eval_depths)

    write_balanced(
        ds["train"], out / "train.jsonl", train_depths, args.limit_train,
        options, args.seed,
    )
    write_balanced(
        ds["validation"], out / "validation.jsonl", eval_depths, args.limit_eval,
        options, args.seed + 1,
    )
    write_balanced(
        ds["test"], out / "test.jsonl", eval_depths, args.limit_eval,
        options, args.seed + 2,
    )


if __name__ == "__main__":
    main()
