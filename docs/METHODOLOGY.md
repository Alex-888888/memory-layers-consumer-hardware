# Methodology — phases A to D

The reconstruction was structured into phases, each with an explicit go/no-go gate and quantitative criteria. The goal was to fail fast and cheaply if the architecture could not be rebuilt on this hardware.

## Phase A — Consolidation
Stabilise the surrounding system before committing GPU time for weeks. (Project-specific; not part of the released code.)

## Phase B — Operational risk lifting
Lift the dominant risk **before** writing model code:

- **EmbeddingBag bandwidth** on the target GPU (the memory pool is a large gather): measured **151 GB/s** (gate: ≥ 150 GB/s).
- **VRAM budget**: Qwen2.5-7B (bf16) + memory pool, measured **22.44 GB** static (gate: < 22.5 GB).

Gate passed (both criteria, narrowly).

## Phase C — Naive memory layer (toy task)
Validate the base equation in pure PyTorch on a toy task before any transformer:

- **Gradcheck** by finite differences — passes at machine epsilon.
- **Overfit** on synthetic associations — **100 % top-1** (gate ≥ 95 %), **100 % pool coverage** (gate ≥ 80 %).

Finding: a strictly naive lookup tends to collapse; **qk-normalisation + load-balancing** are needed for healthy key usage.

## Phase D — Four stages + integration + warm-up

- **Stage 2 — product-key** (Lample 2019): replaces the naive lookup; non-regression vs exhaustive search verified with **KL = 0** on the materialised product set.
- **Stage 3 — Memory+**: SiLU gating, three memory layers sharing one value pool, qk-normalisation.
- **Stage 4a — integration**: injected into Qwen2.5-7B at layers **6/14/22**, backbone **frozen**, via **MLP-ADD** (the memory output is added to the original, frozen MLP — this is what keeps native knowledge intact and avoids the catastrophic disruption that replacing the MLP would cause).
- **Stage 4b — warm-up**: train the memory only (backbone frozen).

### Success criteria and what was met
- Native knowledge preserved (target: regression < 5 pts) → **0 pt** (100 % → 100 %).
- Memory learns factual associations (synthetic) → **100 %** recall.
- Perplexity ≤ +5 % vs backbone, and a public factual benchmark (TriviaQA) → **not yet measured** (see limits in the README).

### The corrective recipe
The first warm-up attempts recalled **0 %** despite a decreasing loss. The diagnosis (see `DIAGNOSTIC.md`) showed two compounding causes — a full-sequence loss diluting the answer signal, and a sparse/offloaded optimizer + sequence packing that prevented the value pool from learning. The recipe that works:

1. **answer-only loss**, 2. **one sequence per fact** (no packing), 3. **dense AdamW on the pool**, 4. **MLP-ADD** with a frozen backbone, 5. enough epochs.

With it, the integrated model reaches **100 %** synthetic recall while preserving native knowledge at **100 %**, on a single 24 GB GPU.
