#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dazo.data import DazoCollator, JsonlDecisionDataset
from dazo.losses import expected_calibration_error
from dazo.modeling_dazo import DazoForDecision


def brier(probs, labels):
    target = torch.nn.functional.one_hot(labels, probs.size(-1)).float()
    return float(((probs.float() - target) ** 2).sum(-1).mean().cpu())


def resolve_model_reference(spec: str) -> str:
    path = Path(spec).expanduser()
    if path.exists():
        return str(path.resolve())

    looks_local = (
        path.is_absolute()
        or spec.startswith(".")
        or spec.startswith("outputs/")
        or len(path.parts) > 2
    )
    if looks_local:
        raise SystemExit(
            f"Local checkpoint not found: {path}. Training must complete successfully before "
            "evaluation; rerun train.py and confirm it prints 'saved .../final'."
        )
    return spec


def resolve_data_path(spec: str) -> str:
    path = Path(spec).expanduser()
    if not path.exists():
        raise SystemExit(f"Evaluation data not found: {path}")
    if not path.is_file():
        raise SystemExit(f"Evaluation data must be a JSONL file: {path}")
    return str(path.resolve())


def _assert_index_range(name: str, tensor: torch.Tensor, size: int) -> None:
    if tensor.numel() == 0:
        return
    lo = int(tensor.min().item())
    hi = int(tensor.max().item())
    if lo < 0 or hi >= size:
        raise RuntimeError(
            f"{name} index out of range before CUDA: min={lo} max={hi} valid=[0,{size - 1}]"
        )


def _preflight_batch(model: DazoForDecision, batch: dict, batch_index: int) -> None:
    vocab_size = int(model.backbone.get_input_embeddings().num_embeddings)
    task_size = int(model.core.compressor.task_emb.num_embeddings)
    rank_size = int(model.core.decoder.rank_emb.num_embeddings)
    _assert_index_range("input_ids", batch["input_ids"], vocab_size)
    _assert_index_range("option_input_ids", batch["option_input_ids"], vocab_size)
    _assert_index_range("task_type", batch["task_type"], task_size)
    _assert_index_range("rank_ids", batch["rank_ids"], rank_size)
    if batch_index == 0:
        print(
            "preflight "
            f"vocab_size={vocab_size} tokenizer_input_max={int(batch['input_ids'].max())} "
            f"option_input_max={int(batch['option_input_ids'].max())} "
            f"task_range=({int(batch['task_type'].min())},{int(batch['task_type'].max())}) "
            f"rank_range=({int(batch['rank_ids'].min())},{int(batch['rank_ids'].max())})"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--loops", default="1,2,4,6,8")
    p.add_argument("--output", help="Optional JSON report path.")
    args = p.parse_args()

    model_ref = resolve_model_reference(args.model)
    data_ref = resolve_data_path(args.data)
    print(f"model={model_ref}")
    print(f"data={data_ref}")

    # Load the Dazo checkpoint itself directly, but always use the original
    # backbone tokenizer. Loading AutoTokenizer from a custom Dazo checkpoint
    # can make Transformers inspect Dazo's auto_map/custom code and select an
    # incompatible tokenizer/config path. Training also uses backbone_name.
    model = DazoForDecision.from_pretrained(model_ref).eval()
    tokenizer_ref = model.config.backbone_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref)
    print(
        f"tokenizer={tokenizer_ref} class={tokenizer.__class__.__name__} "
        f"len={len(tokenizer)} model_vocab={model.backbone.get_input_embeddings().num_embeddings}"
    )

    collator = DazoCollator(tokenizer, model.config.context_max_length, model.config.option_max_length)
    loader = DataLoader(
        JsonlDecisionDataset(data_ref),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    loops = sorted(set(int(x) for x in args.loops.split(",") if x.strip()))
    max_loop = max(loops)
    if min(loops) < 1 or max_loop > model.core.reasoner.max_loop_embeddings:
        raise SystemExit(
            f"loop budgets {loops} exceed valid range 1..{model.core.reasoner.max_loop_embeddings}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    stats = {l: {"correct": 0, "total": 0, "probs": [], "labels": []} for l in loops}
    by_depth = {l: {} for l in loops}
    correctness_rows = []
    started = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            # Validate all embedding indices while tensors are still on CPU so an
            # invalid value produces a useful Python error instead of poisoning CUDA.
            _preflight_batch(model, batch, batch_index)
            metadata = batch.pop("metadata")
            labels = batch.pop("labels").to(device)
            batch.pop("is_ood", None)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(**batch, max_steps=max_loop)
            per = out["per_step_probs"]
            batch_correct = []
            for l in loops:
                idx = min(l, per.size(1)) - 1
                probs = per[:, idx]
                pred = probs.argmax(-1)
                correct_vec = pred.eq(labels)
                batch_correct.append(correct_vec.cpu())
                s = stats[l]
                s["correct"] += correct_vec.sum().item()
                s["total"] += labels.numel()
                s["probs"].append(probs.cpu())
                s["labels"].append(labels.cpu())
                for i, meta in enumerate(metadata):
                    depth = str(meta.get("depth", meta.get("QDep", "unknown")))
                    d = by_depth[l].setdefault(depth, [0, 0])
                    d[0] += int(correct_vec[i].item())
                    d[1] += 1
            correctness_rows.append(torch.stack(batch_correct, dim=1))

    elapsed = time.perf_counter() - started
    report = {
        "model": model_ref,
        "data": data_ref,
        "tokenizer": tokenizer_ref,
        "loop_budgets": loops,
        "examples_per_second": sum(x["total"] for x in stats.values()) / max(len(loops), 1) / max(elapsed, 1e-9),
        "loops": {},
    }
    for l in loops:
        probs = torch.cat(stats[l]["probs"], 0)
        labels = torch.cat(stats[l]["labels"], 0)
        report["loops"][str(l)] = {
            "accuracy": stats[l]["correct"] / max(stats[l]["total"], 1),
            "brier": brier(probs, labels),
            "ece": expected_calibration_error(probs, labels),
            "by_depth": {k: v[0] / max(v[1], 1) for k, v in sorted(by_depth[l].items())},
        }

    correctness = torch.cat(correctness_rows, dim=0).bool()
    if correctness.numel():
        ever_correct_before_final = (
            correctness[:, :-1].any(dim=1)
            if correctness.size(1) > 1
            else torch.zeros(correctness.size(0), dtype=torch.bool)
        )
        final_wrong = ~correctness[:, -1]
        overthought = ever_correct_before_final & final_wrong
        report["overthinking_rate"] = float(overthought.float().mean())
        report["ever_correct_then_final_wrong"] = int(overthought.sum())

        transitions = {}
        for i in range(len(loops) - 1):
            a, b = loops[i], loops[i + 1]
            ca, cb = correctness[:, i], correctness[:, i + 1]
            transitions[f"{a}->{b}"] = {
                "wrong_to_correct": float((~ca & cb).float().mean()),
                "correct_to_wrong": float((ca & ~cb).float().mean()),
                "prediction_correctness_flip": float((ca != cb).float().mean()),
            }
        report["correctness_transitions"] = transitions

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(f"saved report to {path}")


if __name__ == "__main__":
    main()
