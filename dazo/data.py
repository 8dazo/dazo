"""Minimal JSONL dataset format for Dazo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

TASK_TYPES = {"choice": 0, "binary": 1, "noul": 1, "score": 2, "ordinal": 2}


def _text(x: Any) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), default=str)


class JsonlDecisionDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class DazoCollator:
    def __init__(self, tokenizer, context_max_length: int = 1024, option_max_length: int = 32):
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.option_max_length = option_max_length

    @staticmethod
    def _normalize_options(options: Iterable[Any]) -> List[Dict[str, Any]]:
        out = []
        for i, opt in enumerate(options):
            if isinstance(opt, str):
                out.append({"id": opt, "text": opt, "rank": 0})
            else:
                out.append({
                    "id": str(opt.get("id", i)),
                    "text": _text(opt.get("text", opt.get("id", i))),
                    "rank": int(opt.get("rank", 0)),
                })
        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        contexts, options_by_row, labels, task_types, metadata, is_ood = [], [], [], [], [], []
        max_k = 0
        for row in batch:
            task = str(row.get("type", "choice")).lower()
            if task not in TASK_TYPES:
                raise ValueError(f"unknown decision type: {task}")
            instruction = _text(row.get("instruction", "Choose the best option."))
            state = _text(row.get("state", ""))
            contexts.append(f"Instruction: {instruction}\nState:\n{state}")
            opts = self._normalize_options(row["options"])
            if len(opts) < 2:
                raise ValueError("Dazo requires at least two options")
            options_by_row.append(opts)
            max_k = max(max_k, len(opts))
            ids = [o["id"] for o in opts]
            label = str(row["label"])
            if label not in ids:
                raise ValueError(f"label {label!r} not present in options {ids!r}")
            labels.append(ids.index(label))
            task_types.append(TASK_TYPES[task])
            metadata.append(row.get("metadata", {}))
            is_ood.append(bool(row.get("is_ood", False)))

        ctx = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            max_length=self.context_max_length,
            return_tensors="pt",
        )

        flat_option_text = []
        option_mask = torch.zeros(len(batch), max_k, dtype=torch.bool)
        rank_ids = torch.zeros(len(batch), max_k, dtype=torch.long)
        for b, opts in enumerate(options_by_row):
            for k in range(max_k):
                if k < len(opts):
                    flat_option_text.append(opts[k]["text"])
                    option_mask[b, k] = True
                    rank_ids[b, k] = max(0, int(opts[k].get("rank", 0)))
                else:
                    flat_option_text.append("")

        opt = self.tokenizer(
            flat_option_text,
            padding=True,
            truncation=True,
            max_length=self.option_max_length,
            return_tensors="pt",
        )
        opt_len = opt["input_ids"].size(-1)
        option_input_ids = opt["input_ids"].reshape(len(batch), max_k, opt_len)
        option_attention_mask = opt["attention_mask"].reshape(len(batch), max_k, opt_len)

        return {
            "input_ids": ctx["input_ids"],
            "attention_mask": ctx["attention_mask"],
            "option_input_ids": option_input_ids,
            "option_attention_mask": option_attention_mask,
            "option_mask": option_mask,
            "rank_ids": rank_ids,
            "task_type": torch.tensor(task_types, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_ood": torch.tensor(is_ood, dtype=torch.bool),
            "metadata": metadata,
        }
