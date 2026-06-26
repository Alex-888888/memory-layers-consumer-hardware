# Sprint 0 — Consolidation (v0.2, hardened in v0.2.1)

After the Phase A→D reconstruction (memory layers, 100 % synthetic recall, frozen Qwen2.5-7B), Sprint 0 set out to harden the brick by closing three gaps: **formal metrics**, **reproducibility**, and **pool capacity**. All runs on a single RX 7900 XTX (24 GB), ROCm/WSL2. The production checkpoint was never overwritten.

> **v0.2.1 update.** The figures below are now at a defensible scale: perplexity on **WikiText-103 validation** (220k tokens, non-overlapping 2048-token windows), **TriviaQA at n=1000** with a standard deviation over 3 gate seeds, a **multi-seed recall-vs-#facts curve**, and the **500k pool unblocked**. Earlier indicative numbers (single-run, tiny validation sets) are superseded.

## 1. A hidden regression (honest correction)

The v0.1 headline "native knowledge preserved 100 % → 100 %" was measured by **greedy recall** on a small native-fact sample — it masked a real cost. Measured properly:

| metric | backbone | + memory (ungated) | + memory (gated v6) |
|---|---|---|---|
| PPL — WikiText-103 val (220k tok, win 2048) | 7.65 | 8.77 (**+14.7 %**) | 7.82 (**+2.3 %**) |
| PPL — in-domain held-out (15k tok) | 3.96 | 3.36 (−15.1 %) | 3.74 (−5.5 %, net positive) |
| Recall — TriviaQA general (n=1000) | 53.4 % | 45.2 % (**−8.2 pts**) | 52.5 % ± 1.74 (**−0.9 pt**) |

**Reading:** the always-on MLP-ADD memory output is added at *every* token. It reinforces the facts it stores but injects noise elsewhere, taxing general language modelling and general factual recall. On general text the cost is real (+14.7 % PPL ungated); on in-domain text the memory is actually *helpful*. The fix (below) closes most of the general-text gap.

*(The earlier v0.2.0 indicative figures — PPL +19.9 %/+49.6 % on 271/42 tokens, TriviaQA −11.5 pts at n=300 — were single-run on tiny sets. The absolute PPL is lower here because a 2048-token-context perplexity is mechanically below a short-sentence one; the relevant quantity is the **delta**, which is now measured on 220k tokens.)*

## 2. The fix: a learned relevance gate (six iterations)

We add a small **per-token relevance gate** at each memory layer that decides whether the memory output should be added. Backbone **and** memory stay frozen; only the gate trains (~0.5M params/layer, an MLP on the hidden state). Finding the right formulation took six iterations:

| version | mechanism | outcome |
|---|---|---|
| v1 / v2 | gate on a **scalar routing-confidence** | recovers PPL but synthetic recall → 0 % (confidence distributions of stored-fact vs general tokens overlap) |
| v3 | **learned** gate (MLP on hidden state) | PPL recovered, but synthetic recall fails on held-out facts |
| v4 | learned gate trained on the closed fact set | synthetic recall **0 % even in-set** — root cause: closing the memory on the *question* tokens breaks recall (the always-on memory is entangled with the recall mechanism) |
| v5 | gate kept **open over the whole fact context** | synthetic recall **100 %** + PPL recovered; residual: TriviaQA only partly recovered (the gate also opens on general questions) |
| **v6** | **+ general questions as negatives** | **resolved:** synthetic recall 100 %, PPL near backbone, TriviaQA recovered |

**Result (v6, hardened n=1000, 3 gate seeds):**

| config | synthetic recall (in-set) | TriviaQA (n=1000) | PPL (WikiText-103) |
|---|---|---|---|
| backbone | 0 % | 53.4 % | 7.65 |
| + memory, ungated | 100 % | 45.2 % | 8.77 (+14.7 %) |
| + memory, **gated (v6)** | **100 %** | **52.5 % ± 1.74** | **7.82 (+2.3 %)** |

The gate recovers about **89 % of the ungated TriviaQA loss** (−0.9 pt vs backbone, vs −8.2 ungated) while keeping synthetic recall at 100 % and PPL within +2.3 % on a standard 220k-token corpus. The standard deviation (±1.74 pts over 3 independent gate trainings) shows the gate training is reproducible. This frozen-backbone relevance gating is, to our knowledge, not addressed by Berges et al. (which trains jointly from scratch); it is specific to retrofitting memory onto a pre-trained, frozen model. The v0.2.1 gate keys on a single fact domain (sensor-style questions); a multi-domain gate with a held-out-phrasing generalisation test is the subject of v0.2.2. *(Gate code planned for a later release.)*

## 3. Recipe reproducibility and the recall-vs-#facts scaling curve (multi-seed)

The corrective recipe (answer-only loss + one sequence per fact + MLP-ADD), trained from scratch, reaches **100 % synthetic recall with standard deviation 0** across 3 seeds **{137, 7, 23}** at every tested scale:

| facts | recall (3 seeds) | std dev | steps to 100 % |
|---|---|---|---|
| 100 | 100 % | 0 | 1000–1500 |
| 300 | 100 % | 0 | 2500 |
| 1000 | 100 % | 0 | 6000–8000 |
| 5000 (production model) | 100 % | — (single run) | ~142000 |

Two findings:

1. **No capacity wall up to 5000 facts** on a 50k-entry pool — recall stays at 100 %.
2. **Convergence cost is roughly constant at ~30 exposures per fact** (the 5000-fact production run, ~28 exposures, lands on the same law), so the number of training steps scales **linearly** with the number of facts. (Recall onset is back-loaded: a fact is only recalled once the loss on its answer collapses, which happens late; under-budgeting the steps gives a misleadingly low recall.)

This lifts the v0.1 "single run, no multi-seed" caveat across scales, and replaces the earlier micro-only (N=100) reproducibility claim.

## 4. Pool scaling

- **Optimizer verdict.** A micro-benchmark (pool 200k) showed **both dense AdamW and the offloaded (CPU-state) optimizer train the pool to 100 % recall**. The original 0 % was due to full-sequence loss + sequence packing — **not** the optimizer. So the offloaded optimizer is the right path for large pools (its states live on CPU, keeping VRAM bounded).
- **Ceilings.** Dense AdamW is VRAM-capped around **50–100k** entries on 24 GB (200k dense → OOM). Offload reaches **200k with 100 % recall** at ~21 GB VRAM.
- **500k now unblocked.** The 500k training that previously failed (`MemoryRegion::BlockAllocator::alloc failed`) was traced to the **dense pool gradient** (≈3.3 GB), not the activations. Making the pool lookup produce a **sparse gradient** (`F.embedding(..., sparse=True)` — only the looked-up rows get a gradient) with an offloaded optimizer that consumes the sparse gradient resolves it: **100 % recall at 20.2 GB VRAM** on a ~500k pool. Practical ceiling **50k → 200k → 500k**.
- **Toward 1M.** At 1M the limiter shifts to the pool **parameter** itself (≈7.2 GB in bf16, on top of the 7B backbone) — that step awaits more capable hardware.

## 5. Status

The three Sprint 0 gaps are answered, now at a defensible scale:
1. **Formal metrics** — done (WikiText-103 PPL, TriviaQA n=1000 ±σ); they exposed *then* fixed a hidden regression.
2. **Reproducibility** — 100 % recall, std 0, across 3 seeds at 100/300/1000 facts, plus the 5000-fact production model; a clean recall-vs-#facts scaling law (~30 exposures/fact).
3. **Capacity** — offloaded optimizer with a sparse pool gradient, demonstrated to **500k**.

Remaining: a **1M pool** (parameter offload, on larger hardware), a **multi-domain relevance gate** (v0.2.2), and more external benchmarks.

Signal/ghost fusion (initially Phase E) was found unrealizable in its initial formulation and is reframed as **Phase F** (open to Vector Symbolic Architectures or similar). It is not on this repository's critical path.
