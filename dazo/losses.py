"""Training losses and calibration metrics for Dazo."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def binary_cross_entropy_probs(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Autocast-safe BCE for heads that expose probabilities.

    CUDA autocast intentionally rejects ``binary_cross_entropy`` because a preceding sigmoid can
    produce gradients that are not safely representable in reduced precision. Dazo v0 exposes
    probability-space halt/correctness/abstain heads, so reconstruct the corresponding float32
    logits and use the autocast-safe BCE-with-logits kernel. The clamp matches the numerical guard
    we already used around probability-space BCE.
    """
    probs32 = probs.float().clamp(1e-5, 1.0 - 1e-5)
    target32 = target.float()
    logits32 = torch.logit(probs32)
    return F.binary_cross_entropy_with_logits(logits32, target32)


def brier_loss(probs: torch.Tensor, labels: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    target = F.one_hot(labels, num_classes=probs.size(-1)).to(probs.dtype)
    target = target * option_mask.to(target.dtype)
    return ((probs - target) ** 2 * option_mask.to(probs.dtype)).sum(-1).mean()


def shortcut_kl(per_step_logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Align earlier recurrent predictions to the deepest prediction without backprop into teacher."""
    if per_step_logits.size(1) <= 1:
        return per_step_logits.new_zeros(())
    mask = option_mask[:, None, :].bool()
    logits = per_step_logits.masked_fill(~mask, -1e4).float()
    final_p = torch.softmax(logits[:, -1], dim=-1).detach()
    early_logp = torch.log_softmax(logits[:, :-1], dim=-1)
    target = final_p[:, None, :].expand_as(early_logp)
    kl = F.kl_div(early_logp, target, reduction="none").sum(-1)
    weights = torch.linspace(0.5, 1.0, kl.size(1), device=kl.device)
    return (kl * weights[None, :]).mean()


def deep_supervision_ce(per_step_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if per_step_logits.size(1) == 1:
        return F.cross_entropy(per_step_logits[:, 0].float(), labels)
    losses = torch.stack(
        [F.cross_entropy(per_step_logits[:, i].float(), labels, reduction="none") for i in range(per_step_logits.size(1))],
        dim=1,
    )
    weights = torch.linspace(0.25, 1.0, losses.size(1), device=losses.device)
    weights = weights / weights.sum()
    return (losses * weights[None, :]).sum(-1).mean()


def halt_supervision(halt_probs: torch.Tensor, per_step_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        correct = per_step_logits.argmax(-1).eq(labels[:, None]).float()
        stable = torch.flip(torch.cumprod(torch.flip(correct, dims=[1]), dim=1), dims=[1])
        target = torch.maximum(correct * 0.5, stable)
    return binary_cross_entropy_probs(halt_probs, target)


def expected_ponder_cost(halt_probs: torch.Tensor) -> torch.Tensor:
    h = halt_probs.float().clamp(1e-5, 1 - 1e-5)
    survival = torch.cumprod(torch.cat([torch.ones_like(h[:, :1]), 1.0 - h[:, :-1]], dim=1), dim=1)
    stop_p = h * survival
    leftover = (1.0 - stop_p.sum(dim=1, keepdim=True)).clamp_min(0.0)
    stop_p = stop_p.clone()
    stop_p[:, -1:] = stop_p[:, -1:] + leftover
    steps = torch.arange(1, h.size(1) + 1, device=h.device, dtype=h.dtype)
    return (stop_p * steps[None, :]).sum(-1).mean() / float(h.size(1))


def correctness_supervision(per_step_correctness: torch.Tensor, per_step_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target = per_step_logits.argmax(-1).eq(labels[:, None]).float()
    return binary_cross_entropy_probs(per_step_correctness, target)


def abstain_supervision(per_step_abstain: torch.Tensor, per_step_logits: torch.Tensor, labels: torch.Tensor, is_ood: Optional[torch.Tensor] = None) -> torch.Tensor:
    with torch.no_grad():
        wrong = ~per_step_logits.argmax(-1).eq(labels[:, None])
        target = wrong.float()
        if is_ood is not None:
            target = torch.maximum(target, is_ood[:, None].float())
    return binary_cross_entropy_probs(per_step_abstain, target)


def energy_ood_loss(
    energy: torch.Tensor,
    is_ood: torch.Tensor,
    in_margin: float = -3.0,
    out_margin: float = -1.0,
) -> torch.Tensor:
    is_ood = is_ood.bool()
    if not is_ood.any():
        return energy.new_zeros(())
    parts = []
    if (~is_ood).any():
        parts.append(torch.relu(energy[~is_ood] - in_margin).pow(2).mean())
    if is_ood.any():
        parts.append(torch.relu(out_margin - energy[is_ood]).pow(2).mean())
    return sum(parts) / max(len(parts), 1) if parts else energy.new_zeros(())


def compute_dazo_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    option_mask: torch.Tensor,
    is_ood: Optional[torch.Tensor] = None,
    w_deep: float = 0.35,
    w_brier: float = 0.08,
    w_shortcut: float = 0.08,
    w_halt: float = 0.04,
    w_correctness: float = 0.04,
    w_abstain: float = 0.02,
    w_ponder: float = 0.005,
    w_ood: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    final_ce = F.cross_entropy(outputs["logits"].float(), labels)
    deep = deep_supervision_ce(outputs["per_step_logits"], labels)
    brier = brier_loss(outputs["probs"].float(), labels, option_mask)
    shortcut = shortcut_kl(outputs["per_step_logits"], option_mask)
    halt = halt_supervision(outputs["halt_probs"], outputs["per_step_logits"], labels)
    correctness = correctness_supervision(outputs["per_step_correctness"], outputs["per_step_logits"], labels)
    abstain = abstain_supervision(outputs["per_step_abstain"], outputs["per_step_logits"], labels, is_ood)
    ponder = expected_ponder_cost(outputs["halt_probs"])
    ood = outputs["logits"].new_zeros(())
    if is_ood is not None:
        ood = energy_ood_loss(outputs["energy"], is_ood)

    total = (
        final_ce + w_deep * deep + w_brier * brier + w_shortcut * shortcut
        + w_halt * halt + w_correctness * correctness + w_abstain * abstain
        + w_ponder * ponder + w_ood * ood
    )
    metrics = {
        "loss": total.detach(),
        "ce": final_ce.detach(),
        "deep_ce": deep.detach(),
        "brier": brier.detach(),
        "shortcut_kl": shortcut.detach(),
        "halt_bce": halt.detach(),
        "correctness_bce": correctness.detach(),
        "abstain_bce": abstain.detach(),
        "ponder": ponder.detach(),
        "ood": ood.detach(),
    }
    return total, metrics


@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(-1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean() * (conf[sel].mean() - correct[sel].mean()).abs()
    return float(ece.cpu())
