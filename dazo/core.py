"""Core neural architecture for Dazo.

This file deliberately depends only on PyTorch so the recurrent reasoning core can be
unit-tested without downloading a Hugging Face backbone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_heads(dim: int, requested: int) -> int:
    """Return the largest head count <= requested that divides dim."""
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.bool()
    safe = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(safe.float(), dim=dim).to(logits.dtype)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs / denom


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner * 2),
            nn.GLU(dim=-1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StableAttentionBlock(nn.Module):
    """Pre-norm attention + MLP with small learnable residual scales.

    Small residual scales are intentional: deep/recurrent stacks are easier to stabilize when
    every recurrence starts as a near-identity transformation.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        cross_attention: bool = False,
        residual_init: float = 0.1,
    ):
        super().__init__()
        heads = _valid_heads(dim, heads)
        self.cross_attention = cross_attention
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim) if cross_attention else None
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=4.0, dropout=dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(residual_init)))
        self.ff_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(x)
        if self.cross_attention:
            if memory is None:
                raise ValueError("cross-attention block requires memory")
            kv = self.norm_kv(memory)
            kpm = None if memory_mask is None else ~memory_mask.bool()
            attn_out, _ = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        else:
            kpm = None if x_mask is None else ~x_mask.bool()
            attn_out, _ = self.attn(q, q, q, key_padding_mask=kpm, need_weights=False)
        x = x + self.attn_scale * attn_out
        x = x + self.ff_scale * self.ff(self.norm_ff(x))
        return x


class LatentCompressor(nn.Module):
    """Perceiver-style read: learned latent slots cross-attend to the evidence sequence."""

    def __init__(
        self,
        dim: int,
        n_evidence: int,
        n_hypothesis: int,
        n_critic: int,
        n_control: int,
        num_task_types: int = 3,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.counts = (n_evidence, n_hypothesis, n_critic, n_control)
        self.n_slots = sum(self.counts)
        self.slots = nn.Parameter(torch.empty(self.n_slots, dim))
        nn.init.normal_(self.slots, std=0.02)

        role_ids = []
        for role, n in enumerate(self.counts):
            role_ids.extend([role] * n)
        self.register_buffer("role_ids", torch.tensor(role_ids, dtype=torch.long), persistent=False)
        self.role_emb = nn.Embedding(4, dim)
        self.task_emb = nn.Embedding(num_task_types, dim)
        self.read = StableAttentionBlock(dim, heads, dropout, cross_attention=True)
        self.mix = StableAttentionBlock(dim, heads, dropout, cross_attention=False)

    @property
    def critic_slice(self) -> slice:
        a = self.counts[0] + self.counts[1]
        return slice(a, a + self.counts[2])

    @property
    def control_slice(self) -> slice:
        a = self.counts[0] + self.counts[1] + self.counts[2]
        return slice(a, a + self.counts[3])

    def forward(
        self,
        evidence: torch.Tensor,
        evidence_mask: torch.Tensor,
        task_type: torch.Tensor,
    ) -> torch.Tensor:
        bsz = evidence.size(0)
        slots = self.slots.unsqueeze(0).expand(bsz, -1, -1)
        slots = slots + self.role_emb(self.role_ids)[None, :, :]
        slots = slots + self.task_emb(task_type)[:, None, :]
        slots = self.read(slots, memory=evidence, memory_mask=evidence_mask)
        slots = self.mix(slots)
        return slots


class RecurrentLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = StableAttentionBlock(dim, heads, dropout, cross_attention=False)
        self.cross_attn = StableAttentionBlock(dim, heads, dropout, cross_attention=True)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x)
        x = self.cross_attn(x, memory=memory, memory_mask=memory_mask)
        return x


class RecurrentReasoner(nn.Module):
    """Weight-tied recurrent-depth reasoner with time/budget conditioning and gated updates."""

    def __init__(
        self,
        dim: int,
        layers_per_loop: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        max_loop_embeddings: int = 32,
    ):
        super().__init__()
        self.max_loop_embeddings = max_loop_embeddings
        self.step_emb = nn.Embedding(max_loop_embeddings + 1, dim)
        self.budget_emb = nn.Embedding(max_loop_embeddings + 1, dim)
        self.layers = nn.ModuleList(
            [RecurrentLayer(dim, heads, dropout) for _ in range(layers_per_loop)]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def step(
        self,
        x: torch.Tensor,
        initial: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        step_index: int,
        budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step_id = min(step_index, self.max_loop_embeddings)
        budget_id = min(budget, self.max_loop_embeddings)
        time = self.step_emb.weight[step_id] + self.budget_emb.weight[budget_id]
        candidate = x + time[None, None, :]
        for layer in self.layers:
            candidate = layer(candidate, memory=memory, memory_mask=memory_mask)

        gate = torch.sigmoid(self.gate(torch.cat([x, candidate, initial], dim=-1)))
        updated = x + gate * (candidate - x)
        delta = (updated - x).float().pow(2).mean(dim=(-1, -2)).sqrt()
        return updated, delta, gate.squeeze(-1).mean(dim=-1)


class OptionSetDecoder(nn.Module):
    """Permutation-equivariant option decoder.

    No positional embedding is added to ordinary categorical options. For ordinal tasks, semantic
    rank IDs can be supplied and move with the option when the option list is permuted.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        max_rank: int = 256,
    ):
        super().__init__()
        self.rank_emb = nn.Embedding(max_rank + 1, dim)
        self.option_mix = StableAttentionBlock(dim, heads, dropout, cross_attention=False)
        self.read_latents = StableAttentionBlock(dim, heads, dropout, cross_attention=True)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        options: torch.Tensor,
        option_mask: torch.Tensor,
        latents: torch.Tensor,
        rank_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rank_ids is not None:
            rank_ids = rank_ids.clamp(min=0, max=self.rank_emb.num_embeddings - 1)
            options = options + self.rank_emb(rank_ids)
        options = self.option_mix(options, x_mask=option_mask)
        options = self.read_latents(options, memory=latents, memory_mask=None)
        logits = self.score(options).squeeze(-1)
        return logits.masked_fill(~option_mask.bool(), -1e4)


@dataclass
class DazoCoreOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    energy: torch.Tensor
    correctness: torch.Tensor
    abstain: torch.Tensor
    halt_probs: torch.Tensor
    convergence: torch.Tensor
    update_gate: torch.Tensor
    per_step_logits: torch.Tensor
    per_step_probs: torch.Tensor
    per_step_correctness: torch.Tensor
    per_step_abstain: torch.Tensor
    selected_step: torch.Tensor
    latents: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__.copy()


class DazoCore(nn.Module):
    """Dazo v0 core: evidence -> latent workspace -> recurrent reasoning -> option queries."""

    def __init__(
        self,
        context_dim: int,
        option_dim: int,
        latent_dim: int = 384,
        n_evidence_slots: int = 16,
        n_hypothesis_slots: int = 8,
        n_critic_slots: int = 4,
        n_control_slots: int = 4,
        layers_per_loop: int = 2,
        heads: int = 8,
        dropout: float = 0.0,
        max_loop_embeddings: int = 32,
        max_rank: int = 256,
        num_task_types: int = 3,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, latent_dim))
        self.option_proj = nn.Sequential(nn.LayerNorm(option_dim), nn.Linear(option_dim, latent_dim))
        self.compressor = LatentCompressor(
            latent_dim,
            n_evidence_slots,
            n_hypothesis_slots,
            n_critic_slots,
            n_control_slots,
            num_task_types=num_task_types,
            heads=heads,
            dropout=dropout,
        )
        self.reasoner = RecurrentReasoner(
            latent_dim,
            layers_per_loop=layers_per_loop,
            heads=heads,
            dropout=dropout,
            max_loop_embeddings=max_loop_embeddings,
        )
        self.decoder = OptionSetDecoder(latent_dim, heads=heads, dropout=dropout, max_rank=max_rank)

        critic_features = latent_dim + 4
        self.correctness_head = nn.Sequential(
            nn.LayerNorm(critic_features), nn.Linear(critic_features, latent_dim // 2), nn.GELU(), nn.Linear(latent_dim // 2, 1)
        )
        self.abstain_head = nn.Sequential(
            nn.LayerNorm(critic_features), nn.Linear(critic_features, latent_dim // 2), nn.GELU(), nn.Linear(latent_dim // 2, 1)
        )
        self.halt_head = nn.Sequential(
            nn.LayerNorm(critic_features), nn.Linear(critic_features, latent_dim // 2), nn.GELU(), nn.Linear(latent_dim // 2, 1)
        )

    def _critic_pool(self, latents: torch.Tensor) -> torch.Tensor:
        sl = self.compressor.critic_slice
        critic = latents[:, sl, :]
        if critic.size(1) == 0:
            return latents.mean(dim=1)
        return critic.mean(dim=1)

    @staticmethod
    def _distribution_features(
        probs: torch.Tensor,
        option_mask: torch.Tensor,
        convergence: torch.Tensor,
    ) -> torch.Tensor:
        k = option_mask.sum(-1).clamp(min=2).float()
        p = probs.float()
        top2 = torch.topk(p, k=min(2, p.size(-1)), dim=-1).values
        if top2.size(-1) == 1:
            top2 = torch.cat([top2, torch.zeros_like(top2)], dim=-1)
        top1 = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / torch.log(k)
        return torch.stack([top1, margin, entropy, convergence.float()], dim=-1)

    def forward(
        self,
        context_hidden: torch.Tensor,
        context_mask: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor,
        task_type: Optional[torch.Tensor] = None,
        rank_ids: Optional[torch.Tensor] = None,
        max_steps: int = 4,
        min_steps: int = 1,
        adaptive: bool = False,
        halt_threshold: float = 0.65,
        convergence_threshold: float = 0.02,
        correctness_threshold: float = 0.70,
    ) -> DazoCoreOutput:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        bsz = context_hidden.size(0)
        if task_type is None:
            task_type = torch.zeros(bsz, dtype=torch.long, device=context_hidden.device)

        context = self.context_proj(context_hidden)
        options = self.option_proj(option_hidden)
        latents = self.compressor(context, context_mask, task_type)
        initial = latents

        # The recurrent core can re-read original evidence and candidate semantics every loop.
        memory = torch.cat([context, options], dim=1)
        memory_mask = torch.cat([context_mask.bool(), option_mask.bool()], dim=1)

        step_logits = []
        step_probs = []
        convergence = []
        gate_means = []
        halt_probs = []
        correctness_probs = []
        abstain_probs = []

        for t in range(1, max_steps + 1):
            latents, delta, gate_mean = self.reasoner.step(
                latents,
                initial,
                memory=memory,
                memory_mask=memory_mask,
                step_index=t,
                budget=max_steps,
            )
            logits_t = self.decoder(options, option_mask, latents, rank_ids=rank_ids)
            probs_t = masked_softmax(logits_t, option_mask)
            dist_feats = self._distribution_features(probs_t, option_mask, delta)
            critic = self._critic_pool(latents).float()
            features = torch.cat([critic, dist_feats], dim=-1)

            halt_t = torch.sigmoid(self.halt_head(features).squeeze(-1))
            correct_t = torch.sigmoid(self.correctness_head(features).squeeze(-1))
            abstain_t = torch.sigmoid(self.abstain_head(features).squeeze(-1))

            step_logits.append(logits_t)
            step_probs.append(probs_t)
            convergence.append(delta)
            gate_means.append(gate_mean)
            halt_probs.append(halt_t)
            correctness_probs.append(correct_t)
            abstain_probs.append(abstain_t)

        per_step_logits = torch.stack(step_logits, dim=1)
        per_step_probs = torch.stack(step_probs, dim=1)
        convergence_t = torch.stack(convergence, dim=1)
        gate_t = torch.stack(gate_means, dim=1)
        halt_t = torch.stack(halt_probs, dim=1)
        correctness_t = torch.stack(correctness_probs, dim=1)
        abstain_t = torch.stack(abstain_probs, dim=1)

        selected = torch.full((bsz,), max_steps - 1, dtype=torch.long, device=context_hidden.device)
        if adaptive:
            eligible = torch.arange(max_steps, device=context_hidden.device)[None, :] >= (max(min_steps, 1) - 1)
            eligible = eligible.expand(bsz, -1)
            good = (
                (halt_t >= halt_threshold)
                & (correctness_t >= correctness_threshold)
                & (convergence_t <= convergence_threshold)
                & eligible
            )
            has_good = good.any(dim=1)
            first = good.float().argmax(dim=1)
            selected = torch.where(has_good, first, selected)

        batch = torch.arange(bsz, device=context_hidden.device)
        logits = per_step_logits[batch, selected]
        probs = per_step_probs[batch, selected]
        correctness = correctness_t[batch, selected]
        abstain = abstain_t[batch, selected]
        energy = -torch.logsumexp(logits.float().masked_fill(~option_mask.bool(), -1e4), dim=-1)

        return DazoCoreOutput(
            logits=logits,
            probs=probs,
            energy=energy,
            correctness=correctness,
            abstain=abstain,
            halt_probs=halt_t,
            convergence=convergence_t,
            update_gate=gate_t,
            per_step_logits=per_step_logits,
            per_step_probs=per_step_probs,
            per_step_correctness=correctness_t,
            per_step_abstain=abstain_t,
            selected_step=selected + 1,
            latents=latents,
        )
