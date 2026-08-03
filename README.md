# Memory Layers on Consumer Hardware

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-v0.5.0-blue)

An **independent, from-scratch reconstruction** of *Memory Layers at Scale* (Berges et al., 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)) integrated into **Qwen2.5-7B-Instruct**, trained on a **single consumer GPU** (AMD RX 7900 XTX, 24 GB, ROCm/WSL2).

This is **not** a reproduction of the paper at scale, and **not** a SOTA claim. Its value is **practical reproducibility on constrained hardware**, with the **lessons learned** documented honestly — including what did *not* work and why.

> This work is a technical brick extracted from **J.A.R.V.I.S.**, a larger private project (a self-hosted sovereign personal AI assistant). The other components of that project remain private; only this Memory Layers reconstruction is released openly.

> **v0.5.0 (external membership-verification index).** The v0.4.1 finding was that *internal* stored-vs-novel detection is intractable, so reliable abstention needs an **external** check — this release ships one: an auditable, **embedding-free** membership index (form-based regex router + identifier extractor + exact/fuzzy membership) that decides stored vs novel so a system can abstain instead of fabricating. Its **frontier ships first** ([`docs/EXTERNAL_VERIFICATION.md`](docs/EXTERNAL_VERIFICATION.md)): it verifies identifier membership **not** attribute truth (a fake with a real identifier and a false attribute is accepted — the demo makes this executable), the separation is **lexical not semantic** (name-substitution control AUC 0.526), it is demonstrated on **synthetic** data only, and it assumes a **controlled input schema**. Run `python src/membership_index.py`. See [`CHANGELOG.md`](CHANGELOG.md).

> **Earlier releases (v0.2.0 – v0.4.5).** Sprint 0 consolidation and hardening (relevance gate, WikiText-103 PPL, TriviaQA n=1000 ±σ, 500k pool); the multi-domain gate and generalisation frontier; external baselines (RAG / LoRA / kNN-LM) and statistical calibration; related work and threats; the honest safety correction; the open-set-recognition negative result across five internal signal families (`docs/OPENSET_MP_TYLER.md`); reproducible stable-seed error bars (v0.4.4); and the recall-metric debt closed on a measured per-family 100 % (v0.4.5). Full detail in [`CHANGELOG.md`](CHANGELOG.md).

## Why this exists

- **From scratch, not Meta's code.** The architecture is reconstructed from the paper. It does **not** reuse Meta's reference implementation (which is CC-BY-NC); this repository is an independent reimplementation and is released under Apache 2.0.
- **Accessible to researchers without an industrial budget.** Everything runs on one 24 GB consumer GPU under ROCm/WSL2.
- **Honest about its limits** (see below) — the point is reproducibility and lessons, not headline numbers.

## What it does

Adds a trainable **parametric memory** (product-key memory layers) to a **frozen** Qwen2.5-7B backbone, so the model can store and recall arbitrary entity → value associations it was never pretrained on — while keeping its native knowledge intact. A separate **external membership index** (v0.5.0) decides stored vs novel for abstention.

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
- **Dense AdamW on the value pool** — a sparse offloaded optimizer left the pool essentially at its initialisation; a dense optimizer actually trains it. *(Sprint 0 refined this: offload does train the pool; the real culprit was the loss + packing.)*
- **MLP-ADD injection** at layers 6/14/22, backbone frozen — the memory output is added to the frozen MLP, which keeps native knowledge intact.

The full investigation (seven diagnostic steps, refuted hypotheses, root cause) is in [`docs/DIAGNOSTIC.md`](docs/DIAGNOSTIC.md) — the most useful part for anyone attempting their own reconstruction.

### Relevance gate (Sprint 0, v0.2) — removing a hidden cost

Sprint 0 metrics showed the always-on MLP-ADD memory **taxes general competence** even while preserving stored-fact recall. The fix: a small **learned per-token relevance gate** (~0.5M params per memory layer) — **backbone *and* memory frozen; only the gate trains**. It opens on stored-fact contexts and closes on general text. Result (gate v6, v0.2.1): synthetic recall **100 %**, PPL within **+2.3 %** of the backbone on WikiText-103, and **TriviaQA 45.2 % → 52.5 % ± 1.74** (vs **53.4 %** backbone, n=1000, 3 gate seeds; *single-domain gate v6 — a different configuration from the released multi-domain gate, not re-derived in v0.4.4*). Implementation in [`src/relevance_gate.py`](src/relevance_gate.py) / [`docs/RELEVANCE_GATE.md`](docs/RELEVANCE_GATE.md); generalisation in [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md); baselines in [`docs/BASELINES.md`](docs/BASELINES.md); the open-set-recognition negative result in [`docs/OPENSET_MP_TYLER.md`](docs/OPENSET_MP_TYLER.md); the external membership index in [`docs/EXTERNAL_VERIFICATION.md`](docs/EXTERNAL_VERIFICATION.md).

## Results

**Phase A → D (single run, this hardware):**

- EmbeddingBag bandwidth on RX 7900 XTX: **151 GB/s** (above the 150 GB/s go threshold).
- VRAM, Qwen2.5-7B + memory pool: **22.44 GB** static.
- Toy task (Phase C): **100 % top-1** retrieval, gradcheck passes at machine epsilon.
- Integrated model (Phase D): synthetic factual recall **0 % → 100 %**, backbone frozen.

**Sprint 0 (v0.2) consolidation — hardened in v0.2.1:**

- **Hidden regression found (honest correction).** The v0.1 claim "native knowledge preserved 100 % → 100 %" was a *greedy-recall* artifact. The always-on memory taxes general competence: WikiText-103 (220k tokens) perplexity **+14.7 %** ungated, TriviaQA (n=1000) **53.4 % → 45.2 %**.
- **Relevance gate fixes it.** PPL within **+2.3 %** of the backbone, synthetic recall **100 %**, TriviaQA **52.5 % ± 1.74** (single-domain gate v6; the released multi-domain gate re-derives to **52.5 % ± 0.6** under the stable seed, v0.4.4).
- **Recipe reproducibility & scaling.** 100 % synthetic recall with std 0 across 3 seeds at 100 / 300 / 1000 facts, and the 5000-fact production model at 100 %. ~30 exposures per fact; no capacity wall up to 5000 facts on a 50k pool.
- **Pool scaling.** An offloaded optimizer with a **sparse pool gradient** trains a ~500k pool at **100 % recall, 20.2 GB VRAM**. Practical ceiling **50k → 200k → 500k**.

## Baselines (v0.3.2, kNN-LM updated v0.3.4)

On the same synthetic facts and metrics (see [`docs/BASELINES.md`](docs/BASELINES.md), CIs in [`docs/STATS.md`](docs/STATS.md)):

| approach | recall | TriviaQA | WikiText PPL | nature |
|---|---|---|---|---|
| **Memory Layers + gate** | ~100 % | 52.5 % (−0.9) | +2.3 % | parametric, backbone frozen |
| RAG (BM25) | 99.4 % | = backbone | non-destructive | retrieval + per-query context |
| LoRA (r=16) | 82.7 % | **0.7 %** (forgets) | **+45 %** | weights modified |
| kNN-LM (declarative) | 0 % | 42 % / 8 % | +15 % / +70 % | non-parametric, fails Q&A here |
| kNN-LM (Q&A datastore) | ≤13 % (λ=0.9 only) | — | — | non-parametric, structurally inadequate |

Only Memory-Layers-+-gate is simultaneously high-recall, competence-preserving and parametric. At n=1000 the gated TriviaQA residual (−0.9) is within the ±3.1-pt sampling CI — not a statistically significant regression.

## Limits (honest, not minimised)

- **Pool practical ceiling ~500k**, not the paper's 1M target. At 1M the limiter becomes the pool parameter itself (≈7.2 GB in bf16 alongside the 7B backbone) — that step awaits more capable hardware.
- **Native-knowledge cost is real but mitigated.** The gate brings PPL back to within **+2.3 %** and a TriviaQA residual within sampling noise; it is **domain-level** (needs ≥ 1 example per family/cluster; a held-out natural-language family is not recovered).
- **No emergent entity-level safety, and no *internal* fix (v0.3.4 / v0.4.1 – v0.4.3).** On non-stored entities the model **confidently fabricates plausible, format-correct wrong values**. Five internal open-set signal families are at or near chance; reliable entity-level abstention requires an **external** membership/retrieval check — shipped in v0.5.0 as [`src/membership_index.py`](src/membership_index.py), with its frontier ([`docs/EXTERNAL_VERIFICATION.md`](docs/EXTERNAL_VERIFICATION.md)): identifier membership, not attribute truth; lexical, not semantic; synthetic only.
- **Metrics at scale, but still one model.** PPL (220k tokens) and TriviaQA (n=1000, ±σ over 3 gate seeds) are defensible; the underlying memory is a single production checkpoint per scale. Full threats in [`docs/THREATS.md`](docs/THREATS.md).
- **Multi-signal fusion for retrieval augmentation** (Phase E) was unrealizable in its initial formulation and is marked **Phase F**, open to reformulation. Not on the critical path.

## Repository layout

```
src/
  warmup_train.py          # integration + warm-up (memory classes, CPU-offload Adam, training loop)
  eval_factual.py          # recall eval (synthetic + known facts)
  microfit_centered.py     # minimal overfit that proves the recipe end-to-end
  relevance_gate.py        # the relevance-gate mechanism (gate MLP + gated wrapper + training)
  synthetic_facts.py       # 5 synthetic fact families + generic negatives
  train_relevance_gate.py  # end-to-end: train memory + gate, eval held-out phrasing / PPL / TriviaQA
  membership_index.py      # external embedding-free membership index (stored-vs-novel abstention)
  stages/                  # the staged build: product-key, Memory+, Qwen injection
  data/                    # corpus generators (synthetic + public facts + fluency)
benchmarks/                # offload-optimizer micro-benchmark
docs/                      # METHODOLOGY, DIAGNOSTIC, REPRODUCE, SPRINT0, GATE_MULTIDOMAIN, RELEVANCE_GATE, GENERALIZATION, BASELINES, STATS, RELATED_WORK, THREATS, SAFETY_EVAL, OPENSET_MP_TYLER, EXTERNAL_VERIFICATION
data/synthetic_sample.jsonl     # tiny deterministic sample for a quick smoke test
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for environment, install and per-stage commands, and [`docs/SPRINT0.md`](docs/SPRINT0.md) for the consolidation results.

## Supporting the project

This work is done solo on consumer hardware (RX 7900 XTX, 24 GB). Three ways to help, if you find it useful:

- **Direct contributions via GitHub Sponsors** (see the **Sponsor** button at the top of this repository) — to move to more powerful hardware and validate the recipe at larger scale.
- **Company sponsorship** — for organisations interested in native memory in LLMs, industrial application, technical sovereignty. *(Contact to be added soon.)*
- **Technical or academic partnerships** — for labs, companies or researchers who want to collaborate on the next steps. *(Contact to be added soon.)*

If this is useful to you, that's already great; if not, no worries.

## License & attribution

Apache 2.0 (see [`LICENSE`](LICENSE)). Architecture inspired by Berges et al., *Memory Layers at Scale*, arXiv:2412.09764. This is an **independent reconstruction**, not a reuse of Meta's CC-BY-NC reference code. Citation metadata in [`CITATION.cff`](CITATION.cff).
