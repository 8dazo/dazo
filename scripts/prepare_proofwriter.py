#!/usr/bin/env python
"""Convert ProofWriter to depth-aware Dazo JSONL without materializing the full dataset."""
import argparse
import json
import random
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
        "query": row["question"],
        "instruction": "Using only the supplied facts and rules, decide whether the query is true, false, or unknown.",
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


def _write_items(target: Path, items: list[dict]):
    counts = Counter(int(x["metadata"]["depth"]) for x in items)
    labels = Counter(x["label"] for x in items)
    with target.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(target.stem, len(items), target, "depth_counts=", dict(sorted(counts.items())), "label_counts=", dict(sorted(labels.items())))


def write_balanced_train(stream, target: Path, depths: list[int], limit: int, options, seed: int):
    target_quota = quotas(limit, depths)
    counts = Counter()
    items = []
    stream = stream.shuffle(seed=seed, buffer_size=10000)
    for row in stream:
        depth = int(row.get("QDep", 0))
        if depth not in target_quota or counts[depth] >= target_quota[depth]:
            continue
        items.append(convert_row(row, options))
        counts[depth] += 1
        if limit > 0 and len(items) >= limit:
            break
        if all(counts[d] >= target_quota[d] for d in depths):
            break
    missing = {d: target_quota[d] - counts[d] for d in depths if counts[d] < target_quota[d]}
    if missing:
        raise RuntimeError(f"could not fill train depth quotas for {target.name}: {missing}")
    random.Random(seed).shuffle(items)
    _write_items(target, items)


def _waterfill_counts(available: dict[int, int], total: int, depths: list[int]) -> dict[int, int]:
    if total <= 0:
        return {d: available.get(d, 0) for d in depths}
    target = {d: 0 for d in depths}
    remaining = min(total, sum(available.get(d, 0) for d in depths))
    active = set(depths)
    while remaining > 0 and active:
        progressed = False
        for d in sorted(active):
            if remaining <= 0:
                break
            if target[d] < available.get(d, 0):
                target[d] += 1
                remaining -= 1
                progressed = True
            else:
                active.discard(d)
        if not progressed:
            break
    return target


def write_depth_aware_eval(stream, target: Path, depths: list[int], limit: int, options, seed: int):
    cap = max(limit, 1) if limit > 0 else 10000
    reservoirs = {d: [] for d in depths}
    seen = Counter()
    rng = random.Random(seed)
    for row in stream:
        depth = int(row.get("QDep", 0))
        if depth not in reservoirs:
            continue
        seen[depth] += 1
        item = convert_row(row, options)
        bucket = reservoirs[depth]
        if len(bucket) < cap:
            bucket.append(item)
        else:
            j = rng.randrange(seen[depth])
            if j < cap:
                bucket[j] = item
    available = {d: len(reservoirs[d]) for d in depths}
    allocation = _waterfill_counts(available, limit, depths)
    items = []
    for d in depths:
        bucket = reservoirs[d]
        rng.shuffle(bucket)
        items.extend(bucket[: allocation[d]])
    rng.shuffle(items)
    absent = [d for d in depths if available[d] == 0]
    if absent:
        print(f"note: {target.stem} has no examples for requested depths {absent}; excluding them")
    print("available_by_depth=", dict(sorted(available.items())))
    print("selected_by_depth=", dict(sorted(allocation.items())))
    _write_items(target, items)


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
    write_balanced_train(ds["train"], out / "train.jsonl", train_depths, args.limit_train, options, args.seed)
    write_depth_aware_eval(ds["validation"], out / "validation.jsonl", eval_depths, args.limit_eval, options, args.seed + 1)
    write_depth_aware_eval(ds["test"], out / "test.jsonl", eval_depths, args.limit_eval, options, args.seed + 2)


if __name__ == "__main__":
    main()
