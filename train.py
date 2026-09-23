#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
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


def parse_budgets(spec: str, max_steps: int) -> list[int]:
    vals = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    vals = [x for x in vals if 1 <= x <= max_steps]
    return vals or [max_steps]


def model_forward(model, batch, max_steps):
    return model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        query_input_ids=batch.get("query_input_ids"), query_attention_mask=batch.get("query_attention_mask"),
        option_input_ids=batch["option_input_ids"], option_attention_mask=batch["option_attention_mask"],
        joint_input_ids=batch.get("joint_input_ids"), joint_attention_mask=batch.get("joint_attention_mask"),
        marker_positions=batch.get("marker_positions"),
        option_mask=batch["option_mask"], task_type=batch["task_type"], rank_ids=batch["rank_ids"],
        max_steps=max_steps,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_steps):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    pred_counts = label_counts = None
    for batch in loader:
        batch = move(batch, device)
        outputs = model_forward(model, batch, max_steps)
        loss, _ = compute_dazo_loss(outputs, batch["labels"], batch["option_mask"], batch["is_ood"])
        pred = outputs["logits"].argmax(-1)
        total += batch["labels"].numel()
        correct += pred.eq(batch["labels"]).sum().item()
        loss_sum += float(loss) * batch["labels"].numel()
        k = outputs["logits"].size(-1)
        pc = torch.bincount(pred.detach().cpu(), minlength=k)
        lc = torch.bincount(batch["labels"].detach().cpu(), minlength=k)
        pred_counts = pc if pred_counts is None else pred_counts + pc
        label_counts = lc if label_counts is None else label_counts + lc
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "pred_counts": pred_counts.tolist() if pred_counts is not None else [],
        "label_counts": label_counts.tolist() if label_counts is not None else [],
    }


def choose_amp(device: torch.device) -> tuple[bool, torch.dtype, bool]:
    if device.type != "cuda":
        return False, torch.float32, False
    major, _minor = torch.cuda.get_device_capability()
    use_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    return True, dtype, dtype == torch.float16


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
    p.add_argument("--depth-budgets", default="1,2,3,4,6,8")
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--eval-train", action="store_true")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = DazoConfig(**json.loads(Path(args.config).read_text()))
    if args.max_steps is not None:
        if args.max_steps < 1 or args.max_steps > cfg.max_loop_embeddings:
            raise ValueError(f"--max-steps must be in 1..{cfg.max_loop_embeddings}")
        cfg.max_steps = args.max_steps
    if args.unfreeze_backbone:
        cfg.freeze_backbone = False
        cfg.unfreeze_last_n_layers = 0
    depth_budgets = parse_budgets(args.depth_budgets, cfg.max_steps)
    print(f"training recurrent budgets: {depth_budgets}; max_steps={cfg.max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    collator = DazoCollator(
        tokenizer,
        context_max_length=cfg.context_max_length,
        option_max_length=cfg.option_max_length,
        query_max_length=cfg.query_max_length,
        joint_candidate_encoding=cfg.joint_candidate_encoding,
        joint_max_length=cfg.joint_max_length,
    )
    train_ds = JsonlDecisionDataset(args.train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if args.eval_train else None
    eval_loader = DataLoader(JsonlDecisionDataset(args.eval), batch_size=args.batch_size, shuffle=False, collate_fn=collator) if args.eval else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype, scaler_enabled = choose_amp(device)
    print(f"device={device} amp={amp_enabled} amp_dtype={amp_dtype if amp_enabled else 'disabled'} grad_scaler={scaler_enabled}")

    model = DazoForDecision.from_backbone_pretrained(cfg).to(device)
    if args.unfreeze_backbone:
        model.unfreeze_backbone()

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": args.lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg.backbone_lr)})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    params = head_params + backbone_params
    print(
        f"trainable_head_params={sum(p.numel() for p in head_params)} "
        f"trainable_backbone_params={sum(p.numel() for p in backbone_params)} "
        f"head_lr={args.lr} backbone_lr={cfg.backbone_lr if backbone_params else 0}"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not backbone_params:
            model.backbone.eval()
        optimizer.zero_grad(set_to_none=True)
        rolling = {}
        rolling_batches = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move(batch, device)
            train_budget = rng.choice(depth_budgets)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                outputs = model_forward(model, batch, train_budget)
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
            rolling["budget"] = rolling.get("budget", 0.0) + float(train_budget)
            rolling_batches += 1
            if step % 50 == 0:
                denom = float(max(rolling_batches, 1))
                print(f"epoch={epoch} step={step}/{len(train_loader)} " + " ".join(f"{k}={v/denom:.4f}" for k, v in rolling.items()))
                rolling = {}
                rolling_batches = 0

        if train_eval_loader is not None:
            print(f"train_eval epoch={epoch} max_budget={cfg.max_steps}: {evaluate(model, train_eval_loader, device, cfg.max_steps)}")
        if eval_loader is not None:
            print(f"eval epoch={epoch} max_budget={cfg.max_steps}: {evaluate(model, eval_loader, device, cfg.max_steps)}")
        if args.save_every > 0 and epoch % args.save_every == 0:
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
