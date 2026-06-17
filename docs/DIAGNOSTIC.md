# Diagnostic — why the memory recalled 0 %, and how it got to 100 %

This is the most useful part of the project for anyone attempting their own reconstruction. The first integrated warm-up **memorised nothing** (0 % recall on held-out synthetic facts) even though the training loss decreased cleanly. It took seven diagnostic steps — and two reversals — to find the real cause. The point: several "obvious" explanations were **empirically wrong**.

## The symptom
After a multi-epoch warm-up (loss ~0.9), recall of synthetic entity→value facts in greedy generation was **0 %**, while native known facts stayed at 100 %. The loss went down, so *something* learned — but not retrievable facts.

## Phase 1 — free, no-retrain probes on the checkpoint
- **Optimizer stepped fine** (step counter correct, ~44 % of pool rows touched) → not an optimizer-never-ran bug.
- **Memory output dominates the MLP** (‖mem‖/‖mlp‖ ≈ 2–4×) → refutes the "memory is a tiny perturbation / free-rider" hypothesis.
- **Routing is near one-hot** (entropy ≈ 0.005) → refutes the "temperature too soft / fuzzy averaging" hypothesis.
- **Generations have the right *form*** (a hex-like answer) but the **wrong value**, and the *same* value for different entities → not an eval-format artifact; the model understands the question but cannot address per fact.
- **answer-only NLL** drops (8.3 → 4.5) but the gold token's rank stays ~15 → partial learning, far from greedy-retrievable.

First verdict: an **addressing collapse** — different entities route to the same handful of slots.

## Phase 2 — confirm the addressing collapse
Cosine similarity of the memory query for 12 different entities (same question template): **0.999** — and at the first memory layer all 12 selected the *same* 8 slots. Apparent conclusion: the frozen backbone does not carry the entity identity to the read position.

## Phase 3a — the first reversal (outlier dimensions)
That 0.999 was a **measurement artifact**. Transformer hidden states are dominated by a few high-norm "rogue/outlier" dimensions; raw cosine is swamped by this shared component. After **centering** (removing the batch-mean / top component), the same 12 entities separate cleanly (cosine ≈ −0.08) — **in both the hidden state and the query**. So the entity *is* present; L2 qk-norm just doesn't remove the shared component, so the top-k selection is driven by it → same slots.

## Phase 3b — centering fixes routing (no retrain)
Re-running slot selection with a centered query: distinct slots for 12 entities went from **8 → 81** (Jaccard 1.0 → 0.02). The signature is recoverable.

## Phase 3c / 3d — the second reversal (it's not centering either)
A micro-overfit with **answer-only loss + one sequence per fact + dense AdamW** reached **~100 %** recall at 50 and 500 facts — **with or without centering**. So centering was a red herring at this scale; the real levers were the **loss masking** and the **training setup**. (Centering remains a plausible lever at much larger scale, kept as an optional flag.)

## Root cause — the training pipeline, confirmed by a pool probe
At full scale the integrated pipeline still recalled ~0–5 %. A direct probe of the trained pool showed it had **barely moved from initialisation** (median touched-row norm ≈ init, optimizer second moments ~1e-8). The loss reached ~0 **without storing anything in the pool**. Two compounding causes:

1. **Full-sequence loss** dilutes the answer signal (the value tokens are a tiny fraction of the sequence).
2. **Sparse/offloaded optimizer + sequence packing**: the offloaded optimizer barely updated the pool, and packing several facts per window let the model exploit an in-window **copy shortcut** instead of the memory. The loss falls via the frozen backbone + dense projections, not via the value pool.

## The fix (validated end-to-end)
Porting the micro-overfit recipe into the real pipeline:
- **answer-only loss**, **one sequence per fact** (no packing), **dense AdamW on the pool**, **MLP-ADD** with frozen backbone.

Result at production scale (5000 facts, mixed corpus): synthetic recall **0 % → 100 %** (sample of 40), native known facts **100 % → 100 %**.

## Incident worth noting
An auxiliary **KL anchor** (a second reference forward with the memory disabled) caused a **deterministic HIP stall** at a fixed step on this ROCm setup. It was dropped (the known-fact anchor in the corpus sufficed to preserve native knowledge). The training watchdog was changed to a **heartbeat** (detect a frozen log) because a stalled process still looks "alive" to a simple process check.

## Lessons
- On a frozen-backbone setup, **raw cosine of hidden states is misleading** (outlier dims) — center before measuring.
- A decreasing loss does **not** mean the memory learned — **probe the pool** (did its values move?).
- **answer-only loss** and **one fact per sequence** are decisive for entity→value memorisation; packing invites copy shortcuts.
- **The pool optimizer matters**: a sparse/offloaded optimizer can leave the pool at init; use a dense optimizer for pools that fit in VRAM.
- For long runs on flaky GPUs, watch **progress** (log mtime), not just process liveness.
