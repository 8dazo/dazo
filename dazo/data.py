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
    def __init__(
        self,
        tokenizer,
        context_max_length: int = 1024,
        option_max_length: int = 32,
        query_max_length: int = 128,
        joint_candidate_encoding: bool = False,
        shared_joint_encoding: bool = False,
        joint_max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.context_max_length = context_max_length
        self.option_max_length = option_max_length
        self.query_max_length = query_max_length
        self.joint_candidate_encoding = joint_candidate_encoding
        self.shared_joint_encoding = shared_joint_encoding
        self.joint_max_length = joint_max_length

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

    def _special_ids(self):
        tok = self.tokenizer
        required = {
            "cls_token_id": tok.cls_token_id,
            "sep_token_id": tok.sep_token_id,
            "mask_token_id": tok.mask_token_id,
            "pad_token_id": tok.pad_token_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"marker encoding requires tokenizer special tokens: {missing}")
        return tuple(int(required[k]) for k in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id"))

    def _build_shared_joint(
        self,
        queries: List[str],
        evidences: List[str],
        options_by_row: List[List[Dict[str, Any]]],
        max_k: int,
    ) -> Dict[str, torch.Tensor]:
        """Build one encoder sequence per question with one marker per option.

        Format:
          [CLS] query [SEP] [MASK] option_1 [SEP] ... [MASK] option_k [SEP] evidence [SEP]

        Query and option markers are protected. Evidence consumes only the remaining budget.
        This is the fast Dazo path: one bidirectional encoder pass per question regardless of K.
        """
        cls_id, sep_id, mask_id, pad_id = self._special_ids()
        tok = self.tokenizer
        sequences: List[List[int]] = []
        positions = torch.zeros(len(queries), max_k, dtype=torch.long)

        for b, opts in enumerate(options_by_row):
            q = tok(queries[b], add_special_tokens=False)["input_ids"][: self.query_max_length]
            evidence = tok(evidences[b], add_special_tokens=False)["input_ids"]
            seq = [cls_id] + q + [sep_id]

            # Reserve at least marker + one token + separator for every option.
            min_needed = len(seq) + sum(3 for _ in opts) + 1
            if min_needed > self.joint_max_length:
                raise ValueError(
                    f"{len(opts)} options cannot fit in joint_max_length={self.joint_max_length}; "
                    "use shortlist/chunked inference for very high-cardinality decisions"
                )

            # Dynamic per-option budget: keep descriptions useful without stealing all evidence.
            remaining_for_head = max(0, min(self.joint_max_length // 2, self.joint_max_length - len(seq) - 1))
            per_opt = max(1, min(self.option_max_length, (remaining_for_head - 2 * len(opts)) // max(len(opts), 1)))
            for k, opt in enumerate(opts):
                positions[b, k] = len(seq)
                opt_ids = tok(opt["text"], add_special_tokens=False)["input_ids"][:per_opt]
                seq += [mask_id] + opt_ids + [sep_id]

            room = max(0, self.joint_max_length - len(seq) - 1)
            seq += evidence[:room] + [sep_id]
            sequences.append(seq[: self.joint_max_length])

        max_len = max(len(x) for x in sequences)
        ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, : len(seq)] = 1
        return {
            "shared_input_ids": ids,
            "shared_attention_mask": attn,
            "shared_marker_positions": positions,
        }

    def _build_joint_candidates(
        self,
        queries: List[str],
        evidences: List[str],
        options_by_row: List[List[Dict[str, Any]]],
        max_k: int,
    ) -> Dict[str, torch.Tensor]:
        """Build one Laya-inspired joint encoder sequence per candidate."""
        cls_id, sep_id, mask_id, pad_id = self._special_ids()
        tok = self.tokenizer
        q_ids = [tok(q, add_special_tokens=False)["input_ids"][: self.query_max_length] for q in queries]
        e_ids = [tok(e, add_special_tokens=False)["input_ids"] for e in evidences]
        sequences: List[List[int]] = []
        positions = torch.zeros(len(queries), max_k, dtype=torch.long)
        for b, opts in enumerate(options_by_row):
            for k in range(max_k):
                if k >= len(opts):
                    sequences.append([cls_id, sep_id])
                    continue
                option_ids = tok(opts[k]["text"], add_special_tokens=False)["input_ids"][: self.option_max_length]
                prefix = [cls_id] + q_ids[b] + [sep_id]
                marker_pos = len(prefix)
                prefix += [mask_id] + option_ids + [sep_id]
                room = max(0, self.joint_max_length - len(prefix) - 1)
                seq = prefix + e_ids[b][:room] + [sep_id]
                seq = seq[: self.joint_max_length]
                positions[b, k] = marker_pos
                sequences.append(seq)
        max_len = max(len(x) for x in sequences)
        ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[i, : len(seq)] = 1
        return {
            "joint_input_ids": ids.reshape(len(queries), max_k, max_len),
            "joint_attention_mask": attn.reshape(len(queries), max_k, max_len),
            "marker_positions": positions,
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidences, queries, options_by_row, labels, task_types, metadata, is_ood = [], [], [], [], [], [], []
        max_k = 0
        for row in batch:
            task = str(row.get("type", "choice")).lower()
            if task not in TASK_TYPES:
                raise ValueError(f"unknown decision type: {task}")
            instruction = _text(row.get("instruction", "Choose the best option."))
            state = _text(row.get("state", ""))
            query = _text(row.get("query", instruction))
            queries.append(f"Instruction: {instruction}\nQuery: {query}")
            evidences.append(f"Evidence:\n{state}")
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

        ctx = self.tokenizer(evidences, padding=True, truncation=True, max_length=self.context_max_length, return_tensors="pt")
        qry = self.tokenizer(queries, padding=True, truncation=True, max_length=self.query_max_length, return_tensors="pt")

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
        opt = self.tokenizer(flat_option_text, padding=True, truncation=True, max_length=self.option_max_length, return_tensors="pt")
        opt_len = opt["input_ids"].size(-1)

        result = {
            "input_ids": ctx["input_ids"],
            "attention_mask": ctx["attention_mask"],
            "query_input_ids": qry["input_ids"],
            "query_attention_mask": qry["attention_mask"],
            "option_input_ids": opt["input_ids"].reshape(len(batch), max_k, opt_len),
            "option_attention_mask": opt["attention_mask"].reshape(len(batch), max_k, opt_len),
            "option_mask": option_mask,
            "rank_ids": rank_ids,
            "task_type": torch.tensor(task_types, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_ood": torch.tensor(is_ood, dtype=torch.bool),
            "metadata": metadata,
        }
        if self.joint_candidate_encoding:
            result.update(self._build_joint_candidates(queries, evidences, options_by_row, max_k))
        if self.shared_joint_encoding:
            result.update(self._build_shared_joint(queries, evidences, options_by_row, max_k))
        return result
