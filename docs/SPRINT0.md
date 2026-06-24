# Sprint 0 — Consolidation (v0.2)

After the Phase A→D reconstruction (memory layers, 100 % synthetic recall, frozen Qwen2.5-7B), Sprint 0 set out to harden the brick by closing three gaps: **formal metrics**, **reproducibility**, and **pool capacity**. All runs on a single RX 7900 XTX (24 GB), ROCm/WSL2. The production checkpoint was never overwritten.

## 1. A hidden regression (honest correction)

The v0.1 headline “native knowledge preserved 100 % → 100 %” was measured by **greedy recall** on a small native-fact sample — it masked a real cost. Finer metrics:

| metric | backbone | + memory (ungated) | delta |
|---|---|---|---|
| PPL, neutral prose (held-out) | 12.67 | 15.19 | **+19.9 %** |
| PPL, factual prose | 9.84 | 14.72 | **+49.6 %** |
| Recall, anchor facts (stored) | 91.7 % | 100 % | +8.3 pts |
| Recall, TriviaQA (general) | 43.5 % | 32.0 % | **−11.5 pts** |

**Reading:** the always-on MLP-ADD memory output is added at *every* token. It reinforces the facts it stores but injects noise elsewhere, taxing general language modelling and general factual recall. (Figures are single-run on small validation sets — indicative.)

## 2. The fix: a learned relevance gate (six iterations)

We add a small **per-token relevance gate** at each memory layer that decides whether the memory output should be added. Backbone **and** memory stay frozen; only the gate trains (~0.5M params/layer, an MLP on the hidden state). Finding the right formulation took six iterations:

| version | mechanism | outcome |
|---|---|---|
| v1 / v2 | gate on a **scalar routing-confidence** | recovers PPL but synthetic recall → 0 % (confidence distributions of stored-fact vs general tokens overlap) |
| v3 | **learned** gate (MLP on hidden state) | PPL recovered, but synthetic recall fails on held-out facts |
| v4 | learned gate trained on the closed fact set | synthetic recall **0 % even in-set** — root cause: closing the memory on the *question* tokens breaks recall (the always-on memory is entangled with the recall mechanism) |
| v5 | gate kept **open over the whole fact context** | synthetic recall **100 %** + PPL recovered; residual: TriviaQA only partly recovered (the gate also opens on general questions) |
| **v6** | **+ general questions as negatives** | **resolved:** PPL −0.5 %, synthetic recall 100 %, TriviaQA 40.0 % → 47.3 % (vs 50.0 % backbone, n=300) |

**Result (v6, n=300):**

| config | PPL | synthetic recall (in-set) | TriviaQA |
|---|---|---|---|
| backbone | 31.83 | 0 % | 50.0 % |
| + memory, ungated | 38.31 | 100 % | 40.0 % |
| + memory, **gated (v6)** | **31.68 (−0.5 %)** | **100 %** | **47.3 %** |

The gate recovers about **73 % of the ungated TriviaQA loss** while keeping synthetic recall at 100 % and PPL at the backbone level. This frozen-backbone relevance gating is, to our knowledge, not addressed by Berges et al. (which trains jointly from scratch); it is specific to retrofitting memory onto a pre-trained, frozen model. The gate currently keys on a single fact domain (sensor-style questions); extending it to heterogeneous fact types is future work. *(Gate code planned for a later release.)*

## 3. Recipe reproducibility (multi-seed)

The corrective recipe (answer-only loss + one sequence per fact + dense AdamW + MLP-ADD), trained from scratch on seeds **{137, 7, 23}**: **100 % synthetic recall on 3/3 seeds, standard deviation 0**. The recipe is reproducible. (Native preservation was seed-dependent in the bare micro-setup without the anchor corpus / gate — which confirms that native preservation comes from the corpus composition + gate, not the recipe alone.) This lifts the v0.1 “single run, no multi-seed” caveat at micro-scale; the full 5000-fact multi-seed run remains a longer job.

## 4. Pool scaling

- **Optimizer verdict.** A micro-benchmark (pool 200k) showed **both dense AdamW and the offloaded (CPU-state) optimizer train the pool to 100 % recall**. The original 0 % was due to full-sequence loss + sequence packing — **not** the optimizer. So the offloaded optimizer is the right path for large pools (its states live on CPU, keeping VRAM bounded).
- **Ceilings.** Dense AdamW is VRAM-capped around **50–100k** entries on 24 GB (200k dense → OOM). Offload reaches **200k with 100 % recall** at ~21 GB VRAM.
- **500k is pending.** Setup fits (pool param 3.6 GB VRAM, optimizer states ~14 GB CPU RAM, no OOM) but training hits a **ROCm HSA allocation error** (`MemoryRegion::BlockAllocator::alloc failed`) on this WSL2 setup — allocator tuning (pinned memory / chunked transfers / param offload) is pending. **Practical ceiling today: ~200k** (up from the 50k of v0.1), not yet the 1M target.

## 5. Status

The three Sprint 0 gaps now have answers: formal metrics (done, and they exposed *then* fixed a hidden regression), reproducibility (recall 100 % × 3 seeds, std 0), and a clear scaling path (offload, demonstrated to 200k). Remaining engineering: allocator tuning for >200k pools, a multi-domain gate, and benchmarks at larger n.

Signal/ghost fusion (initially Phase E) was found unrealizable in its initial formulation and is reframed as **Phase F** (open to Vector Symbolic Architectures or similar). It is not on this repository's critical path.
