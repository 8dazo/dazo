#!/usr/bin/env python
"""Convert ProofWriter to Dazo JSONL.

This is the first scientific benchmark for Dazo because it exposes gold reasoning depth. Train on
shallow examples, then test deeper examples while increasing Dazo's recurrent loop count.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def normalize_answer(x):
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return "true"
    if s in {"false", "0", "no"}:
        return "false"
    return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/proofwriter")
    p.add_argument("--train-max-depth", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=50000)
    p.add_argument("--limit-eval", type=int, default=10000)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tasksource/proofwriter")
    options = [
        {"id": "true", "text": "The statement is entailed by the facts and rules."},
        {"id": "false", "text": "The statement is contradicted by the facts and rules."},
        {"id": "unknown", "text": "The statement cannot be proven or disproven from the facts and rules."},
    ]

    for split in ["train", "validation", "test"]:
        target = out / f"{split}.jsonl"
        n = 0
        with target.open("w", encoding="utf-8") as f:
            for row in ds[split]:
                depth = int(row.get("QDep", 0))
                if split == "train" and depth > args.train_max_depth:
                    continue
                limit = args.limit_train if split == "train" else args.limit_eval
                if limit and n >= limit:
                    break
                item = {
                    "state": row["theory"],
                    "instruction": "Using only the supplied facts and rules, decide whether the query is true, false, or unknown. Query: " + row["question"],
                    "type": "choice",
                    "options": options,
                    "label": normalize_answer(row["answer"]),
                    "metadata": {"id": row["id"], "depth": depth, "maxD": int(row.get("maxD", 0))},
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
        print(split, n, target)


if __name__ == "__main__":
    main()
