# Dazo

[![Open ProofWriter pilot in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/8dazo/dazo/blob/main/notebooks/Dazo_v0_ProofWriter.ipynb)

Dazo is a research implementation of an **adaptive recurrent latent decision model**. It is built for decisions where the desired output is a probability distribution over explicit candidates, not generated prose.

Hugging Face: `d2v1shx/dazo` · GitHub: `8dazo/dazo`.

## Current validation status

- pure-PyTorch recurrent-core invariants pass in GitHub Actions;
- real `jhu-clsp/mmBERT-small` forward integration passes;
- GitHub → Hugging Face Trusted Publishing is live;
- ProofWriter is the first training gate; no reasoning-improvement claim is made before the recurrence curve is measured.

## v0 architecture

```text
context / evidence
      │
      ▼
mmBERT-small (default)
      │
      ▼
Perceiver-style latent compressor
      │
┌─────┴──────────────────────┐
│ 16 evidence slots          │
│  8 hypothesis slots        │
│  4 critic slots            │
│  4 control slots           │
└─────┬──────────────────────┘
      │
      ▼
2-layer shared recurrent core
      ↻ 1..8 iterations
      │
      ▼
option queries ──► probability distribution
      │
      ├─ correctness critic
      ├─ abstain head
      ├─ energy/OOD score
      └─ convergence + halt probability
```

Options are encoded separately. They are **not** packed into the context prompt, so 77-way and larger decisions do not compete for a fixed option-token budget.

## Why start with mmBERT-small?

The research target can later use mmBERT-base or ModernBERT-base, but `jhu-clsp/mmBERT-small` keeps the first experiments realistic on free/cheap GPUs while preserving multilingual semantics. The backbone is frozen by default; we first need to prove that the recurrent decision architecture adds value.

## Install

```bash
pip install -e '.[train]'
```

## Dataset format

One JSON object per line:

```json
{
  "state": "The customer was charged twice for the same invoice.",
  "instruction": "Which queue should handle this ticket?",
  "type": "choice",
  "options": [
    {"id": "billing", "text": "Billing and payment disputes"},
    {"id": "sales", "text": "Sales and upgrades"},
    {"id": "technical", "text": "Technical support"}
  ],
  "label": "billing",
  "metadata": {"depth": 1}
}
```

Supported `type` values: `choice`, `binary`/`noul`, and `score`/`ordinal`. For ordinal choices, attach `rank` to every option. Rank travels with the semantic option, so input-list permutation still permutes outputs correctly.

## First experiment: ProofWriter depth extrapolation

Training samples recurrent budgets from `1,2,3,4,6,8` instead of training at one fixed loop count.

```bash
python scripts/prepare_proofwriter.py \
  --output data/proofwriter \
  --train-max-depth 3 \
  --limit-train 50000 \
  --limit-eval 10000

python train.py \
  --config configs/dazo-v0-small.json \
  --train data/proofwriter/train.jsonl \
  --eval data/proofwriter/validation.jsonl \
  --output outputs/dazo-proofwriter \
  --epochs 3 \
  --batch-size 8 \
  --depth-budgets 1,2,3,4,6,8

python evaluate.py \
  --model outputs/dazo-proofwriter/final \
  --data data/proofwriter/test.jsonl \
  --loops 1,2,4,6,8
```

The key plot is **accuracy by gold reasoning depth vs recurrent loop count**. If extra test-time recurrence does not help deeper held-out proofs, the core Dazo hypothesis has not been demonstrated.

## High-cardinality experiment: Banking77

```bash
python scripts/prepare_banking77.py --output data/banking77
```

Banking77 gives Dazo 77 candidate intents. Because options are separate queries, each label keeps its complete text representation rather than receiving only a few tokens of a shared prompt budget.

## Recommended ablation order

1. `R0`: no recurrence, 1 loop.
2. `R1`: recurrence with 2/4/8 loops.
3. `R2`: step/budget conditioning + shortcut KL.
4. `R3`: learned halting + convergence criterion.
5. `R4`: LOTUS-style supervised latent rationale blocks.
6. `R5`: MoDr-style low-rank hypothesis branches.
7. `R6`: slot/token-wise early exit for real inference savings.

Do not add R4–R6 until R1–R3 show a stable recurrence benefit.

## Publishing

`main` automatically publishes model-facing files to `d2v1shx/dazo` using Hugging Face Trusted Publishing and GitHub OIDC. No long-lived `HF_TOKEN` GitHub secret is required.

See `research/ARCHITECTURE.md` and `research/EXPERIMENTS.md` for the rationale, gates, and source map.
