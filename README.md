# Memory Layers on Consumer Hardware

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-v0.3.4-blue)

An **independent, from-scratch reconstruction** of *Memory Layers at Scale* (Berges et al., 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)) integrated into **Qwen2.5-7B-Instruct**, trained on a **single consumer GPU** (AMD RX 7900 XTX, 24 GB, ROCm/WSL2).

This is **not** a reproduction of the paper at scale, and **not** a SOTA claim. Its value is **practical reproducibility on constrained hardware**, with the **lessons learned** documented honestly — including what did *not* work and why.

> This work is a technical brick extracted from **J.A.R.V.I.S.**, a larger private project (a self-hosted sovereign personal AI assistant). The other components of that project remain private; only this Memory Layers reconstruction is released openly.

> **v0.2.0 (Sprint 0 consolidation).** Since v0.1 we measured formal metrics, which revealed a hidden cost of the always-on memory and led to a fix (a learned relevance gate), confirmed recipe reproducibility across seeds, and clarified the pool-scaling path. See [`docs/SPRINT0.md`](docs/SPRINT0.md) and [`CHANGELOG.md`](CHANGELOG.md).

> **v0.2.1 (hardening).** The Sprint 0 metrics are now defensible at scale: perplexity on **WikiText-103** (220k tokens, non-overlapping 2048-token windows), **TriviaQA at n=1000 with error bars** over 3 gate seeds, a **multi-seed recall-vs-#facts scaling curve** (100 → 5000 facts), and the pool ceiling **lifted to 500k** via a sparse-gradient fix. See [`docs/SPRINT0.md`](docs/SPRINT0.md) and [`CHANGELOG.md`](CHANGELOG.md).

> **v0.2.2 (multi-domain gate).** The relevance gate is validated across **five fact families with distinct structures** and a **held-out-phrasing test**: trained on some phrasings, it still opens on *unseen* phrasings of the same entities (**0-point recall drop**) — it keys on stored-entity-ness, not the surface template. General-knowledge preservation holds (TriviaQA −0.6 pt, PPL +1.0 %). See [`docs/GATE_MULTIDOMAIN.md`](docs/GATE_MULTIDOMAIN.md).

> **v0.3.0 (gate code released).** The relevance-gate implementation is now public: the generic mechanism ([`src/relevance_gate.py`](src/relevance_gate.py)), a synthetic multi-family fact generator ([`src/synthetic_facts.py`](src/synthetic_facts.py)), and an end-to-end train+eval script ([`src/train_relevance_gate.py`](src/train_relevance_gate.py)) — runnable standalone (`--dry 1` for a ~3 min smoke test). Spec in [`docs/RELEVANCE_GATE.md`](docs/RELEVANCE_GATE.md). All data is synthetic.

> **v0.3.1 (generalisation frontier).** We map where the gate's generalisation extends: it preserves recall on **held-out entities** of every seen family (Δ 0, incl. natural language) and transfers across **held-out structured families** (one distributional cluster), but closes on a **held-out NL family** — a *coverage* limit (≥1 example per family/cluster), not a fundamental one. It is a **domain-relevance** gate, not an entity oracle. *(The original v0.3.1 also claimed emergent entity-level safety via a decode-confidence collapse; this is **corrected in v0.3.4** — see below.)* See [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md).

> **v0.3.2 (external baselines + stats).** RAG, LoRA and kNN-LM on the same facts/metrics. Only Memory-Layers-+-gate reaches ~100 % recall **and** preserves general competence **and** stays parametric (no retrieval/context). RAG matches recall+preservation but pays a per-query retrieval cost; **LoRA forgets catastrophically** (TriviaQA 53.4 → 0.7 %, PPL +45 %); naive **kNN-LM fails** the Q&A recall (0 %) while taxing the general distribution. Binomial CIs ([`docs/STATS.md`](docs/STATS.md)) show the gated residual is **within sampling noise** (no significant regression). See [`docs/BASELINES.md`](docs/BASELINES.md).

> **v0.3.3 (related work + threats).** Positioning against the literature ([`docs/RELATED_WORK.md`](docs/RELATED_WORK.md)) — including the closest concurrent work, **Sparse Memory Finetuning** (Lin et al. 2025), which *sparsely updates* memory slots where we keep everything frozen and add an inference-time gate — and an explicit, not-minimised [`docs/THREATS.md`](docs/THREATS.md). Closes the R-Arxiv preprint-preparation series.

> **v0.3.4 (honest correction + kNN-LM Q&A).** A self-correction release. (1) The v0.3.1 "emergent entity-level safety / decode-confidence collapse 1.00 → 0.66 / no hallucination" claim rested on 40 fake entities; re-measured at **n=360** ([`docs/SAFETY_EVAL.md`](docs/SAFETY_EVAL.md)), the signal is **partial** (AUC 0.69) and the model in fact **confidently fabricates plausible wrong values** for non-stored entities — the accurate property is **no inter-fact leakage**, not "no hallucination". (2) The kNN-LM 0 % was partly a datastore-format artifact: a **Q&A-format datastore** lifts recall to **≤13 % (only at λ=0.9)**, confirming structural inadequacy rather than a strawman ([`docs/BASELINES.md`](docs/BASELINES.md)). A repo that self-corrects an over-claim is more defensible than one that keeps it.

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

Sprint 0 metrics (below) showed the always-on MLP-ADD memory **taxes general competence** even while preserving stored-fact recall. The fix: a small **learned per-token relevance gate** (~0.5M params per memory layer, an MLP on the hidden state) at each memory layer — **backbone *and* memory frozen; only the gate trains**. It opens on stored-fact contexts and closes on general text and general factual questions. Result (gate v6, hardened in v0.2.1): synthetic recall **100 %**, perplexity within **+2.3 %** of the backbone on WikiText-103 (net-positive on in-domain text), and on general knowledge **TriviaQA 45.2 % → 52.5 % ± 1.74** (vs **53.4 %** backbone, n=1000, 3 gate seeds) — about **89 % of the ungated loss recovered** (a residual within sampling noise, see [`docs/STATS.md`](docs/STATS.md)). To our knowledge this frozen-backbone relevance gating is not addressed by Berges et al. (which trains jointly from scratch); it is specific to retrofitting memory onto a pre-trained frozen model. Details in [`docs/SPRINT0.md`](docs/SPRINT0.md); implementation in [`src/relevance_gate.py`](src/relevance_gate.py) / [`docs/RELEVANCE_GATE.md`](docs/RELEVANCE_GATE.md) (v0.3.0). Generalisation frontier mapped in [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md) (v0.3.1); external baselines (RAG / LoRA / kNN-LM) in [`docs/BASELINES.md`](docs/BASELINES.md) (v0.3.2); positioning and limits in [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) / [`docs/THREATS.md`](docs/THREATS.md) (v0.3.3); corrected entity-level safety analysis in [`docs/SAFETY_EVAL.md`](docs/SAFETY_EVAL.md) (v0.3.4).

## Results

**Phase A → D (single run, this hardware):**

- EmbeddingBag bandwidth on RX 7900 XTX: **151 GB/s** (above the 150 GB/s go threshold).
- VRAM, Qwen2.5-7B + memory pool: **22.44 GB** static.
- Toy task (Phase C): **100 % top-1** retrieval, gradcheck passes at machine epsilon.
- Integrated model (Phase D): synthetic factual recall **0 % → 100 %** (sample of 40 over 5000 trained facts), backbone frozen.

**Sprint 0 (v0.2) consolidation — hardened in v0.2.1:**

- **Hidden regression found (honest correction).** The v0.1 claim "native knowledge preserved 100 % → 100 %" was a *greedy-recall* artifact. The always-on memory taxes general competence: on WikiText-103 (220k tokens) perplexity **+14.7 %** ungated, and TriviaQA (n=1000) **53.4 % → 45.2 %**. The memory helps the facts it stores but injects noise on general tokens.
- **Relevance gate fixes it.** PPL within **+2.3 %** of the backbone on WikiText-103 (net-positive in-domain), synthetic recall **100 %**, TriviaQA **52.5 % ± 1.74** (vs 53.4 % backbone, n=1000, 3 gate seeds) — **~89 % of the loss recovered** (residual within sampling noise; see Baselines / Stats).
- **Recipe reproducibility & scaling.** The corrective recipe reaches **100 % synthetic recall with standard deviation 0** across 3 seeds at **every** tested scale — **100, 300, 1000** facts — and the production 5000-fact model also recalls **100 %**. Convergence cost is roughly **constant at ~30 exposures per fact** (so training steps scale linearly with the number of facts); there is **no capacity wall** up to 5000 facts on a 50k-entry pool.
- **Pool scaling.** An offloaded (CPU-state) optimizer trains the pool (the original 0 % came from full-sequence loss + sequence packing, not the optimizer). The **500k** training that previously failed on a ROCm allocation error is now **unblocked** by making the pool gradient **sparse** (only the looked-up rows get a gradient) with an offloaded optimizer that consumes it: **100 % recall at 20.2 GB VRAM**. Practical ceiling **50k → 200k → 500k**.

## Baselines (v0.3.2, kNN-LM updated v0.3.4)

On the same synthetic facts and metrics (see [`docs/BASELINES.md`](docs/BASELINES.md), with confidence intervals in [`docs/STATS.md`](docs/STATS.md)):

| approach | recall | TriviaQA | WikiText PPL | nature |
|---|---|---|---|---|
| **Memory Layers + gate** | ~100 % | 52.5–52.8 % (−0.6) | +2.3 % | parametric, backbone frozen |
| RAG (BM25) | 99.4 % | = backbone | non-destructive | retrieval + per-query context |
| LoRA (r=16) | 82.7 % | **0.7 %** (forgets) | **+45 %** | weights modified |
| kNN-LM (declarative) | 0 % | 42 % / 8 % | +15 % / +70 % | non-parametric, fails Q&A here |
| kNN-LM (Q&A datastore) | ≤13 % (λ=0.9 only) | — | — | non-parametric, structurally inadequate |

Only Memory-Layers-+-gate is simultaneously high-recall, competence-preserving and parametric (no retrieval, no per-query context). At n=1000 the gated TriviaQA residual (−0.6) is within the ±3.1-pt sampling CI — not a statistically significant regression. Positioning vs the literature (incl. the concurrent Sparse Memory Finetuning) is in [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md); limits in [`docs/THREATS.md`](docs/THREATS.md).

## Limits (honest, not minimised)

- **Pool practical ceiling ~500k**, not the paper's 1M target. Dense AdamW is VRAM-capped (~50–100k); the offloaded optimizer with a sparse pool gradient reaches **500k at 100 % recall**. At 1M the limiter becomes the pool parameter itself (≈7.2 GB in bf16 alongside the 7B backbone) — that step awaits more capable hardware.
- **Native-knowledge cost is real but mitigated.** The always-on memory degrades PPL / general recall (see Results); the relevance gate brings it back to within **+2.3 %** PPL on WikiText-103 and a TriviaQA residual within sampling noise. The gate is **domain-level**: it generalises to held-out entities and to held-out families within a distributional cluster, but needs at least one example per family/cluster (a held-out natural-language family is not recovered) — see [`docs/GENERALIZATION.md`](docs/GENERALIZATION.md).
- **No emergent entity-level safety (v0.3.4 correction).** On non-stored entities the model **confidently fabricates plausible, format-correct wrong values**; it does not abstain, and decode-confidence flags them only partially (AUC 0.69). The only robust entity-level property is **no inter-fact leakage** — see [`docs/SAFETY_EVAL.md`](docs/SAFETY_EVAL.md).
- **Metrics now at scale, but still one model.** PPL (WikiText-103, 220k tokens) and TriviaQA (n=1000, ±σ over 3 gate seeds) and the recall scaling curve (multi-seed) are defensible; the underlying memory is still a single production checkpoint per scale. Full threats-to-validity in [`docs/THREATS.md`](docs/THREATS.md).
- **Signal/ghost fusion** (as initially formulated in Phase E) was found unrealizable in its initial formulation and is currently marked as **Phase F**, open to reformulation through Vector Symbolic Architectures or similar approaches. Not on the critical path of this repository.

Future versions are expected to address these (1M pool via parameter offload on larger hardware, more external benchmarks, a head-to-head with Sparse Memory Finetuning, a second NL family, and an explicit abstention mechanism).

## Repository layout

```
src/
  warmup_train.py          # integration + warm-up (memory classes, CPU-offload Adam, training loop)
  eval_factual.py          # recall eval (synthetic + known facts)
  microfit_centered.py     # minimal overfit that proves the recipe end-to-end
  relevance_gate.py        # the relevance-gate mechanism (gate MLP + gated wrapper + training)
  synthetic_facts.py       # 5 synthetic fact families + generic negatives
  train_relevance_gate.py  # end-to-end: train memory + gate, eval held-out phrasing / PPL / TriviaQA
  stages/                  # the staged build: product-key, Memory+, Qwen injection
  data/                    # corpus generators (synthetic + public facts + fluency)
benchmarks/                # offload-optimizer micro-benchmark
docs/                      # METHODOLOGY, DIAGNOSTIC, REPRODUCE, SPRINT0, GATE_MULTIDOMAIN, RELEVANCE_GATE, GENERALIZATION, BASELINES, STATS, RELATED_WORK, THREATS, SAFETY_EVAL
data/synthetic_sample.jsonl     # tiny deterministic sample for a quick smoke test
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
