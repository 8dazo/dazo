---
library_name: transformers
license: apache-2.0
tags:
- decision-model
- latent-reasoning
- recurrent-depth
- text-classification
- adaptive-compute
- multilingual
base_model: jhu-clsp/mmBERT-small
---

# Dazo

**Dazo** is an experimental recurrent latent decision model. It is designed to spend variable test-time compute on structured decisions without generating a natural-language chain of thought.

Dazo is not a pretrained general-purpose checkpoint yet. The initial release is a research architecture and training harness.

## Architecture

1. A bidirectional encoder reads the evidence/context.
2. Candidate actions/labels are encoded separately rather than packed into the context sequence.
3. A Perceiver-style cross-attention compressor maps the evidence into a fixed latent workspace.
4. A weight-tied recurrent Transformer repeatedly refines that workspace.
5. Candidate options query the workspace in parallel to produce a calibrated probability distribution.
6. A separate critic predicts correctness/abstention, while recurrent convergence and a learned halt probability provide stopping signals.

The v0 workspace uses evidence, hypothesis, critic, and control slots. The option decoder is permutation-equivariant for categorical decisions; ordinal options carry explicit semantic rank IDs.

## Intended research questions

- Does accuracy increase when Dazo receives more recurrent inference steps on problems requiring deeper composition?
- Can Dazo generalize from shallow training chains to deeper unseen chains by increasing recurrence at test time?
- Can adaptive stopping avoid the known recurrent-depth failure mode of overthinking?
- Does separate option encoding eliminate the high-cardinality and option-order bottlenecks seen in prompt-packed decision models?

## Status

Research prototype. Do not treat probabilities as production-calibrated until the model is trained and calibrated on the deployment domain.
