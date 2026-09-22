"""Hugging Face configuration for Dazo."""
from __future__ import annotations

from typing import Any, Optional

from transformers import PretrainedConfig


class DazoConfig(PretrainedConfig):
    model_type = "dazo"

    def __init__(
        self,
        backbone_name: str = "jhu-clsp/mmBERT-small",
        backbone_config: Optional[dict[str, Any]] = None,
        context_max_length: int = 1024,
        option_max_length: int = 32,
        latent_dim: int = 384,
        n_evidence_slots: int = 16,
        n_hypothesis_slots: int = 8,
        n_critic_slots: int = 4,
        n_control_slots: int = 4,
        layers_per_loop: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.0,
        max_steps: int = 8,
        min_steps: int = 1,
        max_loop_embeddings: int = 32,
        max_rank: int = 256,
        halt_threshold: float = 0.65,
        convergence_threshold: float = 0.02,
        correctness_threshold: float = 0.70,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        # Persist the encoder architecture inside Dazo's config. This lets
        # from_pretrained() construct the backbone with AutoModel.from_config()
        # while Transformers is in its meta-device loading context, then load
        # the actual backbone weights from the Dazo checkpoint state dict.
        self.backbone_config = backbone_config
        self.context_max_length = context_max_length
        self.option_max_length = option_max_length
        self.latent_dim = latent_dim
        self.n_evidence_slots = n_evidence_slots
        self.n_hypothesis_slots = n_hypothesis_slots
        self.n_critic_slots = n_critic_slots
        self.n_control_slots = n_control_slots
        self.layers_per_loop = layers_per_loop
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.max_loop_embeddings = max_loop_embeddings
        self.max_rank = max_rank
        self.halt_threshold = halt_threshold
        self.convergence_threshold = convergence_threshold
        self.correctness_threshold = correctness_threshold
        self.freeze_backbone = freeze_backbone
        self.architectures = ["DazoForDecision"]
        self.auto_map = {
            "AutoConfig": "configuration_dazo.DazoConfig",
            "AutoModel": "modeling_dazo.DazoForDecision",
        }
