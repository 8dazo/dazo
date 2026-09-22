# Dazo experimental roadmap

## Gate 0 — implementation invariants

Before training, the core must satisfy:

- candidate permutation equivariance;
- masked candidates get zero probability;
- recurrence is weight-tied across iterations;
- gradients reach the recurrent core;
- ordinal semantic rank moves with the option rather than with list position.

The included unit tests cover these invariants.

## Gate 1 — does recurrent depth actually help?

**Dataset:** ProofWriter.

Train only on examples with gold query depth <= 3. Evaluate the full validation/test distribution grouped by depth. Run the same checkpoint at 1, 2, 4, 6 and 8 recurrent iterations.

Pass condition: deeper held-out examples should improve with added recurrence, without a large loss on shallow examples. A model that merely improves because it was trained with more FLOPs is not enough; we need a useful inference-time compute curve from one checkpoint.

Primary metrics:

- accuracy by reasoning depth and loop count;
- NLL / Brier / ECE;
- convergence delta by depth;
- percentage of examples whose prediction flips after becoming correct (overthinking rate).

## Gate 2 — remove Laya's candidate-budget ceiling

**Dataset:** Banking77 (77 labels).

Compare prompt-packed decision baselines, Dazo 1-loop, and recurrent Dazo. Randomly permute all 77 options and verify that Dazo's semantic probabilities permute with them.

## Gate 3 — natural logical reasoning

Add ReClor and LogiQA. These test whether the recurrence benefit survives outside synthetic Horn-rule reasoning.

## Gate 4 — supervised latent reasoning

Switch to a structured ProofWriter build with gold proof graphs. Map proof nodes/steps to a fixed number of latent hypothesis blocks and add a training-only auxiliary decoder. Compare outcome-only recurrence, hidden-state distillation, and LOTUS-style parallel step supervision.

## Gate 5 — multi-branch reasoning

Only after Gate 4, add MoDr-style low-rank recurrent experts. Start with four adapter branches and top-2 routing. Measure exploration gain separately from parameter gain with an equal-parameter dense-adapter baseline.

## Gate 6 — actual adaptive compute

Dazo v0 computes the full requested recurrence budget and selects a stopping point post-hoc. Once stopping is trustworthy, implement real slot-wise skipping and measure wall-clock latency, not only FLOPs.

## Minimum benchmark report before a public trained Dazo-0 checkpoint

| Axis | Required result |
|---|---|
| Depth scaling | 1/2/4/6/8-loop curve on ProofWriter |
| General reasoning | ReClor + LogiQA |
| High cardinality | Banking77 77-way |
| Calibration | NLL, Brier, ECE |
| Robustness | option permutation, OOD/abstention |
| Efficiency | latency and throughput by loop count |
| Ablations | no recurrence, no shortcut loss, no critic, no convergence gate |

A public checkpoint should state failures as prominently as wins, especially if recurrence saturates or overthinking appears.
