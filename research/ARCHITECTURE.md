# Dazo v0 architecture decision record

## Goal

Build a non-autoregressive decision model whose test-time compute can scale with decision difficulty. The model should preserve the latency and structured-output advantages of encoder decision models while making iterative latent computation explicit and measurable.

## What Dazo keeps from Laya

- bidirectional encoder representations;
- direct option probabilities rather than generated text;
- explicit calibration as an evaluation target;
- typed decisions.

## What Dazo changes

### 1. Options are a set, not prompt text

Laya places all answer descriptions in a fixed prompt budget. This creates a structural ceiling for high-cardinality decisions. Dazo encodes each option independently, then decodes all options using a shared, permutation-equivariant query mechanism.

### 2. A fixed latent workspace

A small learned slot array cross-attends to the full evidence sequence. Most recurrent computation happens over the slots. The design is inspired by Perceiver/Perceiver IO: expensive repeated processing is moved from the long input sequence into a compact latent space.

### 3. Recurrent depth is trained, not inferred after the fact

Dazo's recurrent core reuses the same layers for multiple iterations. It is conditioned on current step and total budget so a single checkpoint can support different reasoning depths. Training supervises every recurrence and aligns shorter paths toward the deepest route.

### 4. Stable recurrence

Pre-norm attention blocks use small learnable residual scales. Every recurrent update is additionally gated against both the current state and the initial compressed state. This is motivated by evidence that very deep looped Transformers can suffer signal-propagation failures.

### 5. Stopping does not equal confidence thresholding

A wrong classifier may still be highly confident. Dazo therefore exposes three independent signals:

- latent convergence (fixed-point style state change);
- a learned halting hazard;
- a learned correctness/abstention critic.

v0 selects an adaptive step post-hoc but computes the full loop budget. A later systems version can skip converged slots/tokens for real wall-clock savings.

### 6. Deep supervision and shortcut consistency

Every recurrent step produces option logits. Training combines final CE, CE at intermediate steps, Brier loss, KL alignment from shallower routes to the deepest route, halt supervision, and a small ponder cost.

## Why v0 does not include everything

MoDr-style multi-branch experts, LOTUS-style token-level latent rationale supervision, token-wise early exit, and fixed-point implicit differentiation are deliberately deferred. Each is promising, but adding all of them before establishing a recurrent-depth gain would destroy the ability to attribute improvements.

## v0 scientific milestone

The first publishable result is not raw benchmark accuracy. It is a *compute scaling curve*.

Train on ProofWriter examples up to a limited reasoning depth. Evaluate deeper held-out examples with 1, 2, 4, 6, and 8 recurrent steps. The core hypothesis is supported only if extra recurrence improves deeper unseen compositions while preserving shallow-task accuracy.

## Research sources

- Laya repository: https://github.com/NandhaKishorM/laya
- ModernBERT: https://huggingface.co/answerdotai/ModernBERT-base
- mmBERT: https://huggingface.co/blog/mmbert
- Recurrent-depth/Huginn: https://arxiv.org/abs/2502.05171
- LoopFormer: https://arxiv.org/abs/2602.11451
- LOTUS: https://arxiv.org/abs/2606.31779
- MoDr (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/8a81180649011d1690e05c68bfeaa57c-Abstract-Conference.html
- Perceiver IO: https://arxiv.org/abs/2107.14795
- Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
- Energy-based OOD detection: https://arxiv.org/abs/2010.03759
