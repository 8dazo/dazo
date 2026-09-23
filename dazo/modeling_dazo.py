"""Dazo Hugging Face wrapper.

Dazo keeps evidence, query, and candidate options as distinct streams. Newer checkpoints can
query-condition every candidate before evidence matching. A lightweight joint decision-token
head, inspired by Laya's contextual marker scoring, lets candidates interact before scoring while
remaining permutation-equivariant over the option set. The recurrent latent core acts as a
residual refinement.
"""
from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration_dazo import DazoConfig
from .core import DazoCore, masked_softmax


class DazoForDecision(PreTrainedModel):
    config_class = DazoConfig
    base_model_prefix = "dazo"
    all_tied_weights_keys = {}

    def __init__(self, config: DazoConfig):
        super().__init__(config)
        backbone_cfg = self._resolve_backbone_config(config)
        self.backbone = AutoModel.from_config(backbone_cfg)

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

        if config.query_conditioning:
            heads = self._valid_heads(hidden, config.num_attention_heads)
            self.query_norm = nn.LayerNorm(hidden)
            self.query_option_norm = nn.LayerNorm(hidden)
            self.query_cross_attn = nn.MultiheadAttention(
                hidden, heads, dropout=config.dropout, batch_first=True
            )
            self.query_fuse = nn.Sequential(
                nn.LayerNorm(hidden * 4),
                nn.Linear(hidden * 4, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            self.query_fuse_norm = nn.LayerNorm(hidden)

        if config.base_compatibility:
            heads = self._valid_heads(hidden, config.num_attention_heads)
            self.base_context_norm = nn.LayerNorm(hidden)
            self.base_option_norm = nn.LayerNorm(hidden)
            self.base_cross_attn = nn.MultiheadAttention(
                hidden, heads, dropout=config.dropout, batch_first=True
            )
            if config.joint_decision_head:
                decision_dim = int(config.decision_dim)
                decision_heads = self._valid_heads(decision_dim, config.num_attention_heads)
                self.decision_pair_proj = nn.Sequential(
                    nn.LayerNorm(hidden * 4),
                    nn.Linear(hidden * 4, decision_dim),
                    nn.GELU(),
                )
                self.decision_global_proj = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, decision_dim),
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=decision_dim,
                    nhead=decision_heads,
                    dim_feedforward=decision_dim * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.decision_head = nn.TransformerEncoder(
                    layer,
                    num_layers=max(1, int(config.decision_head_layers)),
                    enable_nested_tensor=False,
                )
                self.decision_score = nn.Sequential(
                    nn.LayerNorm(decision_dim),
                    nn.Linear(decision_dim, 1),
                )
            else:
                self.base_pair_score = nn.Sequential(
                    nn.LayerNorm(hidden * 4),
                    nn.Linear(hidden * 4, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )

        if config.freeze_backbone:
            self.freeze_backbone()

    @staticmethod
    def _valid_heads(hidden: int, requested: int) -> int:
        heads = min(requested, hidden)
        while heads > 1 and hidden % heads != 0:
            heads -= 1
        return heads

    @staticmethod
    def _resolve_backbone_config(config: DazoConfig):
        saved = getattr(config, "backbone_config", None)
        if saved:
            saved = dict(saved)
            model_type = saved.pop("model_type")
            return AutoConfig.for_model(model_type, **saved)
        backbone_cfg = AutoConfig.from_pretrained(config.backbone_name)
        config.backbone_config = backbone_cfg.to_dict()
        return backbone_cfg

    @classmethod
    def from_backbone_pretrained(cls, config: DazoConfig) -> "DazoForDecision":
        backbone_cfg = AutoConfig.from_pretrained(config.backbone_name)
        config.backbone_config = backbone_cfg.to_dict()
        model = cls(config)
        model.backbone = AutoModel.from_pretrained(config.backbone_name, config=backbone_cfg)
        if config.freeze_backbone:
            model.freeze_backbone()
            if getattr(config, "unfreeze_last_n_layers", 0) > 0:
                model.unfreeze_last_backbone_layers(config.unfreeze_last_n_layers)
        return model

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        return self

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        return self

    def unfreeze_last_backbone_layers(self, n: int):
        n = int(n)
        if n <= 0:
            return self
        layer_ids = set()
        for name, _p in self.backbone.named_parameters():
            m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if m:
                layer_ids.add(int(m.group(1)))
        if not layer_ids:
            raise RuntimeError("Could not identify transformer layers in backbone parameter names")
        selected = set(sorted(layer_ids)[-n:])
        count = 0
        for name, p in self.backbone.named_parameters():
            m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if m and int(m.group(1)) in selected:
                p.requires_grad = True
                count += p.numel()
        print(f"unfrozen_backbone_layers={sorted(selected)} trainable_backbone_params={count}")
        return self

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * w).sum(1) / w.sum(1).clamp_min(1.0)

    def _encode_options(self, option_input_ids: torch.Tensor, option_attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, nopt, length = option_input_ids.shape
        flat_ids = option_input_ids.reshape(bsz * nopt, length)
        flat_mask = option_attention_mask.reshape(bsz * nopt, length)
        out = self.backbone(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state
        pooled = self._mean_pool(out, flat_mask)
        return pooled.reshape(bsz, nopt, -1)

    def _condition_options_on_query(
        self,
        option_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        query_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        options = self.query_option_norm(option_hidden)
        query = self.query_norm(query_hidden)
        attended, _ = self.query_cross_attn(
            options,
            query,
            query,
            key_padding_mask=~query_attention_mask.bool(),
            need_weights=False,
        )
        pair = torch.cat([options, attended, options * attended, (options - attended).abs()], dim=-1)
        update = self.query_fuse(pair)
        return self.query_fuse_norm(option_hidden + update)

    def _joint_decision_logits(
        self,
        pair: torch.Tensor,
        option_mask: torch.Tensor,
        query_hidden: Optional[torch.Tensor],
        query_attention_mask: Optional[torch.Tensor],
        context_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        option_tokens = self.decision_pair_proj(pair)
        if query_hidden is not None and query_attention_mask is not None:
            global_hidden = self._mean_pool(query_hidden, query_attention_mask)
        else:
            global_hidden = self._mean_pool(context_hidden, attention_mask)
        global_token = self.decision_global_proj(global_hidden).unsqueeze(1)
        tokens = torch.cat([global_token, option_tokens], dim=1)
        global_valid = torch.ones(option_mask.size(0), 1, dtype=torch.bool, device=option_mask.device)
        valid = torch.cat([global_valid, option_mask.bool()], dim=1)
        contextualized = self.decision_head(tokens, src_key_padding_mask=~valid)
        logits = self.decision_score(contextualized[:, 1:]).squeeze(-1)
        return logits.masked_fill(~option_mask.bool(), -1e4)

    def _base_compatibility_logits(
        self,
        context_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        option_hidden: torch.Tensor,
        option_mask: torch.Tensor,
        query_hidden: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        options = self.base_option_norm(option_hidden)
        context = self.base_context_norm(context_hidden)
        attended, _ = self.base_cross_attn(
            options,
            context,
            context,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        pair = torch.cat([options, attended, options * attended, (options - attended).abs()], dim=-1)
        if self.config.joint_decision_head:
            return self._joint_decision_logits(
                pair,
                option_mask,
                query_hidden,
                query_attention_mask,
                context_hidden,
                attention_mask,
            )
        logits = self.base_pair_score(pair).squeeze(-1)
        return logits.masked_fill(~option_mask.bool(), -1e4)

    def _apply_base_compatibility(self, out, base_logits: torch.Tensor, option_mask: torch.Tensor):
        scale = float(self.config.recurrent_logit_scale)
        per_step_logits = base_logits[:, None, :] + scale * out.per_step_logits
        step_mask = option_mask[:, None, :].bool().expand_as(per_step_logits)
        per_step_probs = masked_softmax(per_step_logits, step_mask)
        selected = out.selected_step.long().clamp_min(1) - 1
        batch = torch.arange(base_logits.size(0), device=base_logits.device)
        logits = per_step_logits[batch, selected]
        probs = per_step_probs[batch, selected]
        energy = -torch.logsumexp(logits.float().masked_fill(~option_mask.bool(), -1e4), dim=-1)
        out.per_step_logits = per_step_logits
        out.per_step_probs = per_step_probs
        out.logits = logits
        out.probs = probs
        out.energy = energy
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_input_ids: torch.Tensor,
        option_attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
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
        reasoning_context = context
        reasoning_mask = attention_mask.bool()
        query_hidden = None

        if self.config.query_conditioning:
            if query_input_ids is None or query_attention_mask is None:
                raise ValueError("query_conditioning=True requires query_input_ids and query_attention_mask")
            query_hidden = self.backbone(
                input_ids=query_input_ids, attention_mask=query_attention_mask
            ).last_hidden_state
            options = self._condition_options_on_query(options, query_hidden, query_attention_mask)
            reasoning_context = torch.cat([query_hidden, context], dim=1)
            reasoning_mask = torch.cat([query_attention_mask.bool(), attention_mask.bool()], dim=1)

        out = self.core(
            context_hidden=reasoning_context,
            context_mask=reasoning_mask,
            option_hidden=options,
            option_mask=option_mask.bool(),
            task_type=task_type,
            rank_ids=rank_ids,
            max_steps=max_steps or self.config.max_steps,
            min_steps=min_steps or self.config.min_steps,
            adaptive=adaptive,
            halt_threshold=halt_threshold if halt_threshold is not None else self.config.halt_threshold,
            convergence_threshold=(convergence_threshold if convergence_threshold is not None else self.config.convergence_threshold),
            correctness_threshold=(correctness_threshold if correctness_threshold is not None else self.config.correctness_threshold),
        )

        if self.config.base_compatibility:
            base_logits = self._base_compatibility_logits(
                context,
                attention_mask,
                options,
                option_mask,
                query_hidden=query_hidden,
                query_attention_mask=query_attention_mask,
            )
            out = self._apply_base_compatibility(out, base_logits, option_mask)
        return out.to_dict()
