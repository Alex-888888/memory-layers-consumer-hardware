# Memory Layers on Consumer Hardware

![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

An **independent, from-scratch reconstruction** of *Memory Layers at Scale* (Berges et al., 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)) integrated into **Qwen2.5-7B-Instruct**, trained on a **single consumer GPU** (AMD RX 7900 XTX, 24 GB, ROCm/WSL2).

This is **not** a reproduction of the paper at scale, and **not** a SOTA claim. Its value is **practical reproducibility on constrained hardware**, with the **lessons learned** documented honestly — including what did *not* work and why.

> This work is a technical brick extracted from **J.A.R.V.I.S.**, a larger private project (a self-hosted sovereign personal AI assistant). The other components of that project remain private; only this Memory Layers reconstruction is released openly.

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
- **Dense AdamW on the value pool** — a sparse offloaded optimizer left the pool essentially at its initialisation; a dense optimizer actually trains it.
- **MLP-ADD injection** at layers 6/14/22, backbone frozen — the memory output is added to the frozen MLP, which keeps native knowledge intact.

The full investigation (seven diagnostic steps, refuted hypotheses, root cause) is in [`docs/DIAGNOSTIC.md`](docs/DIAGNOSTIC.md) — this is the most useful part for anyone attempting their own reconstruction.

## Results (single run, this hardware)

- EmbeddingBag bandwidth on RX 7900 XTX: **151 GB/s** (above the 150 GB/s go threshold).
- VRAM, Qwen2.5-7B + memory pool: **22.44 GB** static.
- Toy task (Phase C): **100 % top-1** retrieval, gradcheck passes at machine epsilon.
- Integrated model (Phase D): synthetic factual recall **0 % → 100 %** (sample of 40 over 5000 trained facts), **native known facts preserved 100 % → 100 %**, backbone frozen.

## Limits (honest, not minimised)

- **Pool size 50k (dense) instead of the paper's 1M target.** The offloaded optimizer intended for a large (1M) pool did **not** actually train the pool; the working recipe uses a smaller dense pool. A repair path is identified but not done. Memory capacity is therefore well below the paper's.
- **Formal metrics still to complete**: perplexity vs the backbone alone, and a public factual benchmark (e.g. TriviaQA). Only native-knowledge preservation (100 %) is measured so far.
- **Single run, no multi-seed cross-validation.**
- **Signal/ghost fusion (the larger project's next step) is not started.**

Future versions are expected to address these (larger pool, formal benchmarks, multi-seed).

## Repository layout

```
src/
  warmup_train.py        # integration + warm-up (memory classes, CPU-offload Adam, training loop)
  eval_factual.py        # recall eval (synthetic + known facts)
  microfit_centered.py   # minimal overfit that proves the recipe end-to-end
  stages/                # the staged build: product-key, Memory+, Qwen injection
  data/                  # corpus generators (synthetic + public facts + fluency)
benchmarks/              # offload-optimizer micro-benchmark
docs/                    # METHODOLOGY.md, DIAGNOSTIC.md, REPRODUCE.md
data/synthetic_sample.jsonl   # tiny deterministic sample for a quick smoke test
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for environment, install and per-stage commands.

## Supporting the project

This work is done solo on consumer hardware (RX 7900 XTX, 24 GB). The hardware constraints forced architectural compromises — notably the 50k pool instead of 1M. Three ways to help, if you find it useful:

- **Direct contributions** — to move to more powerful hardware and validate the recipe at larger scale. *(Link to be added soon.)*
- **Company sponsorship** — for organisations interested in the outcomes of this research (native memory in LLMs, industrial application, technical sovereignty). *(Contact to be added soon.)*
- **Technical or academic partnerships** — for labs, companies or researchers who want to collaborate on the next steps (notably the signal/ghost fusion). *(Contact to be added soon.)*

If this is useful to you, that's already great; if not, no worries.

## License & attribution

Apache 2.0 (see [`LICENSE`](LICENSE)). Architecture inspired by Berges et al., *Memory Layers at Scale*, arXiv:2412.09764. This is an **independent reconstruction**, not a reuse of Meta's CC-BY-NC reference code. Citation metadata in [`CITATION.cff`](CITATION.cff).
