"""Dazo Hugging Face wrapper.

The same encoder is reused for context and option semantics. Options are short and flattened into
one batch; this is intentionally simple for v0. Production versions can cache option embeddings or
replace the option path with a smaller semantic encoder without changing DazoCore.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModel, PreTrainedModel

from .configuration_dazo import DazoConfig
from .core import DazoCore


class DazoForDecision(PreTrainedModel):
    config_class = DazoConfig
    base_model_prefix = "dazo"

    def __init__(self, config: DazoConfig):
        super().__init__(config)
        self.backbone = AutoModel.from_pretrained(config.backbone_name)
        hidden = int(self.backbone.config.hidden_size)
        self.core = DazoCore(
            context_dim=hidden,
            option_dim=hidden,
            latent_dim=config.latent_dim,
            n_evidence_slots=config.n_evidence_slots,
            n_hypothesis_slots=config.n_hypothesis_slots,
            n_critic_slots=config.n_critic_slots,
            n_control_slots=config.n_control_slots,
            layers_per_loop=config.layers_per_loop,
            heads=config.num_attention_heads,
            dropout=config.dropout,
            max_loop_embeddings=config.max_loop_embeddings,
            max_rank=config.max_rank,
        )
        if config.freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        return self

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * w).sum(1) / w.sum(1).clamp_min(1.0)

    def _encode_options(
        self,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, nopt, length = option_input_ids.shape
        flat_ids = option_input_ids.reshape(bsz * nopt, length)
        flat_mask = option_attention_mask.reshape(bsz * nopt, length)
        out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        pooled = self._mean_pool(out, flat_mask)
        return pooled.reshape(bsz, nopt, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
        task_type: Optional[torch.Tensor] = None,
        rank_ids: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        adaptive: bool = False,
        halt_threshold: Optional[float] = None,
        convergence_threshold: Optional[float] = None,
        correctness_threshold: Optional[float] = None,
        **unused,
    ):
        context = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        options = self._encode_options(option_input_ids, option_attention_mask)
        out = self.core(
            context_hidden=context,
            context_mask=attention_mask.bool(),
            option_hidden=options,
            option_mask=option_mask.bool(),
            task_type=task_type,
            rank_ids=rank_ids,
            max_steps=max_steps or self.config.max_steps,
            min_steps=min_steps or self.config.min_steps,
            adaptive=adaptive,
            halt_threshold=halt_threshold if halt_threshold is not None else self.config.halt_threshold,
            convergence_threshold=(
                convergence_threshold
                if convergence_threshold is not None
                else self.config.convergence_threshold
            ),
            correctness_threshold=(
                correctness_threshold
                if correctness_threshold is not None
                else self.config.correctness_threshold
            ),
        )
        return out.to_dict()
