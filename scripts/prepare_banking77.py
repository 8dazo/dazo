#!/usr/bin/env python
"""Convert Banking77 into a 77-option Dazo decision benchmark."""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/banking77")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("mteb/banking77")

    mapping = {}
    for split in ds:
        for row in ds[split]:
            mapping[int(row["label"])] = row["label_text"]
    ordered = [mapping[i] for i in sorted(mapping)]
    options = [{"id": x, "text": x.replace("_", " ")} for x in ordered]

    for split in ds:
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in ds[split]:
                item = {
                    "state": row["text"],
                    "instruction": "Which banking intent best describes this customer message?",
                    "type": "choice",
                    "options": options,
                    "label": row["label_text"],
                    "metadata": {"source": "banking77"},
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(split, out / f"{split}.jsonl")


if __name__ == "__main__":
    main()
