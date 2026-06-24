# Memory Layers on Consumer Hardware

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-v0.2.0-blue)

An **independent, from-scratch reconstruction** of *Memory Layers at Scale* (Berges et al., 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)) integrated into **Qwen2.5-7B-Instruct**, trained on a **single consumer GPU** (AMD RX 7900 XTX, 24 GB, ROCm/WSL2).

This is **not** a reproduction of the paper at scale, and **not** a SOTA claim. Its value is **practical reproducibility on constrained hardware**, with the **lessons learned** documented honestly — including what did *not* work and why.

> This work is a technical brick extracted from **J.A.R.V.I.S.**, a larger private project (a self-hosted sovereign personal AI assistant). The other components of that project remain private; only this Memory Layers reconstruction is released openly.

> **v0.2.0 (Sprint 0 consolidation).** Since v0.1 we measured formal metrics, which revealed a hidden cost of the always-on memory and led to a fix (a learned relevance gate), confirmed recipe reproducibility across seeds, and clarified the pool-scaling path. See [`docs/SPRINT0.md`](docs/SPRINT0.md) and [`CHANGELOG.md`](CHANGELOG.md).

## Why this exists

- **From scratch, not Meta's code.** The architecture is reconstructed from the paper. It does **not** reuse Meta's reference implementation (which is CC-BY-NC); this repository is an independent reimplementation and is released under Apache 2.0.
- **Accessible to researchers without an industrial budget.** Everything runs on one 24 GB consumer GPU under ROCm/WSL2.
- **Honest about its limits** (see below) — the point is reproducibility and lessons, not headline numbers.

## What it does

Adds a trainable **parametric memory** (product-key memory layers) to a **frozen** Qwen2.5-7B backbone, so the model can store and recall arbitrary entity → value associations it was never pretrained on — while keeping its native knowledge intact.

## Methodology (phases A → D)

The build followed four phases with explicit go/no-go gates:

- **A — Consolidation** of the surrounding system.
- **B — Risk lifting**: measured EmbeddingBag bandwidth on the target GPU and the VRAM budget before writing any model code.
- **C — Naive memory layer** validated on a toy task (gradcheck + overfit) before any transformer integration.
- **D — The four stages**: (1) naive lookup → (2) product-key factorization (Lample 2019) → (3) Memory+ (SiLU gating, shared value pool, qk-normalization) → (4) integration into Qwen2.5-7B at layers **6/14/22**, backbone **frozen**, with a warm-up that trains the memory only.

### The recipe that actually works (empirically found)

A naive warm-up (full-sequence loss, packed sequences, offloaded sparse optimizer) **memorised nothing** (0 % recall) even though the loss went down. The combination that works:

- **Answer-only loss** — compute the loss only on the answer tokens, not the whole sequence (the answer signal is otherwise drowned).
- **One sequence per fact** — no packing of multiple facts per window (packing lets the model take an in-window copy shortcut instead of using the memory).
- **Dense AdamW on the value pool** — a sparse offloaded optimizer left the pool essentially at its initialisation; a dense optimizer actually trains it. *(Sprint 0 refined this: see Pool scaling below — offload does train the pool; the real culprit was the loss + packing.)*
- **MLP-ADD injection** at layers 6/14/22, backbone frozen — the memory output is added to the frozen MLP, which keeps native knowledge intact.

The full investigation (seven diagnostic steps, refuted hypotheses, root cause) is in [`docs/DIAGNOSTIC.md`](docs/DIAGNOSTIC.md) — this is the most useful part for anyone attempting their own reconstruction.

### Relevance gate (Sprint 0, v0.2) — removing a hidden cost

Sprint 0 metrics (below) showed the always-on MLP-ADD memory **taxes general competence** even while preserving stored-fact recall. The fix: a small **learned per-token relevance gate** (~0.5M params per memory layer, an MLP on the hidden state) at each memory layer — **backbone *and* memory frozen; only the gate trains**. It opens on stored-fact contexts and closes on general text and general factual questions. Result: **PPL within −0.5 % of the backbone**, synthetic recall **100 %**, TriviaQA **40.0 % → 47.3 %** (vs 50.0 % backbone, n=300) — about **73 % of the ungated loss recovered**. To our knowledge this frozen-backbone relevance gating is not addressed by Berges et al. (which trains jointly from scratch); it is specific to retrofitting memory onto a pre-trained frozen model. Details in [`docs/SPRINT0.md`](docs/SPRINT0.md). *(Gate code planned for a later release.)*

## Results

**Phase A → D (single run, this hardware):**

- EmbeddingBag bandwidth on RX 7900 XTX: **151 GB/s** (above the 150 GB/s go threshold).
- VRAM, Qwen2.5-7B + memory pool: **22.44 GB** static.
- Toy task (Phase C): **100 % top-1** retrieval, gradcheck passes at machine epsilon.
- Integrated model (Phase D): synthetic factual recall **0 % → 100 %** (sample of 40 over 5000 trained facts), backbone frozen.

**Sprint 0 (v0.2) consolidation:**

- **Hidden regression found (honest correction).** The v0.1 claim “native knowledge preserved 100 % → 100 %” was a *greedy-recall* artifact. Finer metrics show the always-on memory taxes general competence: perplexity **+19.9 %** (neutral prose) / **+49.6 %** (factual prose), and TriviaQA recall **~43 % → ~32 %** (single run, small validation set — indicative). The memory helps the facts it stores but injects noise on general tokens.
- **Relevance gate fixes it** (see recipe section): PPL −0.5 % vs backbone, synthetic recall 100 %, TriviaQA recovered (~73 % of the loss, n=300).
- **Recipe reproducibility:** the corrective recipe validated across seeds **{137, 7, 23} → 100 % synthetic recall on 3/3, standard deviation 0**.
- **Pool scaling:** an offloaded (CPU-state) optimizer **does** train the pool — the original 0 % came from full-sequence loss + sequence packing, not the optimizer. Dense AdamW is capped ~50–100k entries on 24 GB; **offload reaches 200k with 100 % recall**. Practical ceiling today: **~200k** (up from 50k), not yet the 1M target.

## Limits (honest, not minimised)

- **Pool practical ceiling ~200k**, not the paper's 1M target. Dense AdamW is VRAM-capped (~50–100k); the offloaded optimizer reaches 200k but 500k currently hits a **ROCm HSA allocation error** during training (allocator tuning pending).
- **Native-knowledge cost is real but mitigated.** The always-on memory degrades PPL/general recall (see Results); the relevance gate brings it back to within −0.5 % PPL and ~73 % TriviaQA-loss recovery, but a small residual remains. The gate is trained on a single fact domain (sensor-style questions); generalising it to heterogeneous fact types is future work.
- **Indicative single-run metrics.** PPL and TriviaQA figures are single-run on small validation sets; the full 5000-fact multi-seed run remains a longer job.
- **Signal/ghost fusion** (as initially formulated in Phase E) was found unrealizable in its initial formulation and is currently marked as **Phase F**, open to reformulation through Vector Symbolic Architectures or similar approaches. Not on the critical path of this repository.

Future versions are expected to address these (larger pool via allocator tuning, multi-domain gate, formal benchmarks at larger n).

## Repository layout

```
src/
  warmup_train.py        # integration + warm-up (memory classes, CPU-offload Adam, training loop)
  eval_factual.py        # recall eval (synthetic + known facts)
  microfit_centered.py   # minimal overfit that proves the recipe end-to-end
  stages/                # the staged build: product-key, Memory+, Qwen injection
  data/                  # corpus generators (synthetic + public facts + fluency)
benchmarks/              # offload-optimizer micro-benchmark
docs/                    # METHODOLOGY.md, DIAGNOSTIC.md, REPRODUCE.md, SPRINT0.md
data/synthetic_sample.jsonl   # tiny deterministic sample for a quick smoke test
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for environment, install and per-stage commands, and [`docs/SPRINT0.md`](docs/SPRINT0.md) for the consolidation results.

## Supporting the project

This work is done solo on consumer hardware (RX 7900 XTX, 24 GB). The hardware constraints forced architectural compromises — notably the pool ceiling. Three ways to help, if you find it useful:

- **Direct contributions via GitHub Sponsors** (see the **Sponsor** button at the top of this repository) — to move to more powerful hardware and validate the recipe at larger scale.
- **Company sponsorship** — for organisations interested in the outcomes of this research (native memory in LLMs, industrial application, technical sovereignty). *(Contact to be added soon.)*
- **Technical or academic partnerships** — for labs, companies or researchers who want to collaborate on the next steps. *(Contact to be added soon.)*

If this is useful to you, that's already great; if not, no worries.

## License & attribution

Apache 2.0 (see [`LICENSE`](LICENSE)). Architecture inspired by Berges et al., *Memory Layers at Scale*, arXiv:2412.09764. This is an **independent reconstruction**, not a reuse of Meta's CC-BY-NC reference code. Citation metadata in [`CITATION.cff`](CITATION.cff).
