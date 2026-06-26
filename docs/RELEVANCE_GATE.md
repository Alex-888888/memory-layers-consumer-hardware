# Relevance gate — implementation (v0.3.0)

This document specifies the **learned relevance gate** that lets a frozen LLM keep its native
competence while an always-on parametric memory is present. The code is in
[`src/relevance_gate.py`](../src/relevance_gate.py) (mechanism),
[`src/synthetic_facts.py`](../src/synthetic_facts.py) (synthetic data) and
[`src/train_relevance_gate.py`](../src/train_relevance_gate.py) (end-to-end training + eval).
All data used here is synthetic; the code contains no project-specific content.

## Problem

The memory is injected as **MLP-ADD** at layers 6/14/22: `out = mlp(x) + memory(x)`. Because
the memory output is added at *every* token, it taxes general competence (perplexity up,
general factual recall down) even though it lifts stored-fact recall. The gate makes the
memory **conditional**.

## Architecture

A small per-token MLP on the hidden state, one per memory layer:

```
RelevanceGate(d) = Linear(d, 128) -> ReLU -> Linear(128, 1)        # ~0.5M params at d=3584
```

It is applied multiplicatively on the memory output, via a sigmoid:

```
GatedMemoryMLP.forward(x) = mlp(x) + sigmoid(gate(x)) * memory(x)
```

`mlp` (backbone) **and** `memory` are frozen; **only the gate trains**. Two flags expose the
baselines: `mem_off=True` → backbone only; `gate_on=False` → ungated (memory always added).

## Training the gate

The gate is a per-token **relevance classifier** trained by supervised binary classification
on hidden states, *after* the memory is trained and frozen.

1. **Collect features** (`collect_features`, gradient-free, gate OFF so features are
   gate-independent): run the model over labelled contexts and cache the hidden state at each
   gated layer.
   - **Positives** = stored-fact contexts (question + answer), labelled **1** over the whole
     fact context.
   - **Negatives** = generic neutral prose + general factual questions, labelled **0**.
2. **Train** (`train_gates`): per layer, Adam on a class-balanced binary cross-entropy
   (`pos_weight = #neg / #pos`) over the cached token features.

The two design points that make it work (found empirically, see `docs/SPRINT0.md`):
- label the **whole fact context** as positive (not only the answer tokens) — closing the
  memory on the *question* tokens breaks recall, because the always-on memory is entangled
  with the recall mechanism;
- include **general factual questions as negatives** — otherwise the gate opens on any
  question-shaped input and re-injects memory noise on general knowledge.

## Hyperparameters (defaults, reproducible)

| item | value |
|---|---|
| gate hidden width | 128 |
| gate optimizer | Adam, lr 2e-3 |
| gate steps / layer | 1200 |
| gate batch (token features) | 4096 |
| memory layers | 6 / 14 / 22 |
| memory pool size | 50 176 (m = 224) |
| memory optimizer | offloaded Adam (CPU bf16 states) |
| memory training | answer-only loss, one sequence per fact, ~40 exposures/fact, batch 4, lr 5e-4 |
| seed | 11 (configurable) |

## Reproduce

```bash
# from src/ (requires warmup_train.py in the same package)
python train_relevance_gate.py --dry 1          # ~3 min smoke test (4 facts/family)
python train_relevance_gate.py                  # full run (60 facts/family, 5 families)
```

The full run trains a fresh memory on five synthetic fact families, trains the gate on
phrasings {0,1,2}, and reports: held-out-phrasing recall (phrasing 3, unseen by the gate),
gate open-rate, perplexity on WikiText-103, and TriviaQA. Results are written to
`relevance_gate_results.json`.

## Scope and limits

- This is the **generic mechanism**. The synthetic fact families are illustrative; applying the
  gate to real heterogeneous facts is downstream work.
- The gate is a *relevance classifier on hidden states*; it does not modify the backbone or the
  memory. Its guarantees are empirical (see `docs/SPRINT0.md`, `docs/GATE_MULTIDOMAIN.md`).
- Generalisation tests (held-out entities, held-out family, negative control) are reported
  separately.
