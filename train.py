#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dazo.configuration_dazo import DazoConfig
from dazo.data import DazoCollator, JsonlDecisionDataset
from dazo.losses import compute_dazo_loss
from dazo.modeling_dazo import DazoForDecision


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, max_steps):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for batch in loader:
        batch = move(batch, device)
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            option_input_ids=batch["option_input_ids"], option_attention_mask=batch["option_attention_mask"],
            option_mask=batch["option_mask"], task_type=batch["task_type"], rank_ids=batch["rank_ids"],
            max_steps=max_steps,
        )
        loss, _ = compute_dazo_loss(outputs, batch["labels"], batch["option_mask"], batch["is_ood"])
        pred = outputs["logits"].argmax(-1)
        total += batch["labels"].numel()
        correct += pred.eq(batch["labels"]).sum().item()
        loss_sum += float(loss) * batch["labels"].numel()
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dazo-v0-small.json")
    p.add_argument("--train", required=True)
    p.add_argument("--eval")
    p.add_argument("--output", default="outputs/dazo-v0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = DazoConfig(**json.loads(Path(args.config).read_text()))
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.unfreeze_backbone:
        cfg.freeze_backbone = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    collator = DazoCollator(tokenizer, cfg.context_max_length, cfg.option_max_length)
    train_ds = JsonlDecisionDataset(args.train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = None
    if args.eval:
        eval_ds = JsonlDecisionDataset(args.eval)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DazoForDecision(cfg).to(device)
    if args.unfreeze_backbone:
        model.unfreeze_backbone()

    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if cfg.freeze_backbone:
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        rolling = {}
        for step, batch in enumerate(train_loader, start=1):
            batch = move(batch, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    option_input_ids=batch["option_input_ids"], option_attention_mask=batch["option_attention_mask"],
                    option_mask=batch["option_mask"], task_type=batch["task_type"], rank_ids=batch["rank_ids"],
                    max_steps=cfg.max_steps,
                )
                loss, parts = compute_dazo_loss(outputs, batch["labels"], batch["option_mask"], batch["is_ood"])
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k, v in parts.items():
                rolling[k] = rolling.get(k, 0.0) + float(v)
            if step % 50 == 0:
                denom = 50.0
                print(f"epoch={epoch} step={step}/{len(train_loader)} " + " ".join(f"{k}={v/denom:.4f}" for k, v in rolling.items()))
                rolling = {}

        if eval_loader is not None:
            metrics = evaluate(model, eval_loader, device, cfg.max_steps)
            print(f"eval epoch={epoch}: {metrics}")

        ckpt = Path(args.output) / f"epoch-{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    final = Path(args.output) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
