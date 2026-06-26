# Changelog

## v0.3.0 — Relevance-gate code released (2026-06)

First of the R-Arxiv hardening series (preprint preparation). The relevance-gate implementation is now public.

### Added
- `src/relevance_gate.py`: the generic mechanism — `RelevanceGate` (per-token MLP, ~0.5M params/layer), `GatedMemoryMLP` (`out = mlp(x) + sigmoid(gate(x)) * memory(x)`, backbone **and** memory frozen), feature collection and per-layer class-balanced BCE training.
- `src/synthetic_facts.py`: five synthetic fact families with distinct structures + four phrasings each (three for training, one held out) + generic negatives.
- `src/train_relevance_gate.py`: end-to-end — trains a fresh memory on the synthetic families, trains the gate on phrasings {0,1,2}, evaluates held-out-phrasing recall, gate open-rate, WikiText-103 perplexity and TriviaQA. Runnable standalone (`--dry 1` ≈ 3 min smoke test).
- `docs/RELEVANCE_GATE.md`: architecture, training procedure, exact hyperparameters, seeds, reproduce commands.

### Notes
- All data is synthetic; the released code contains no project-specific content.
- Upcoming in this series: reinforced generalisation tests (held-out entities / family, negative control), external baselines (RAG, LoRA, …), statistical tightening, and Related Work / Threats sections.

## v0.2.2 — Multi-domain relevance gate (2026-06)

The relevance gate is shown **not** to be a single-domain artefact.

### Added
- `docs/GATE_MULTIDOMAIN.md`: the gate trained across **five fact families with distinct structures** (sensor hex / config integer / service semver / node coordinate / protocol hex-byte), plus a **held-out-phrasing test**.
- **Held-out-phrasing generalisation.** The gate is trained on a subset of question phrasings and evaluated on an *unseen* phrasing of the same entities: gated recall **100 % = ungated 100 % (0-point drop)**, gate open-rate **0.94 (trained) vs 0.95 (held-out)**. The gate keys on stored-entity-ness, not the surface template.
- **Multi-domain general-knowledge preservation.** With the multi-domain gate: TriviaQA (n=1000, 3 gate seeds) **52.8 % ± 0.28** vs 53.4 % backbone (−0.6 pt, ~95 % of the ungated loss recovered); PPL WikiText-103 **+1.0 %**; stored-fact recall 100 % across all five structures.

### Notes
- Fact families used for this study are synthetic. Gate implementation code remains planned for a later release.

## v0.2.1 — Sprint 0 hardening (2026-06)

Metrics moved from "indicative" to defensible scale; pool ceiling lifted.

### Changed (hardened results)
- **Perplexity at scale.** PPL now measured on **WikiText-103 validation** (220k tokens, non-overlapping 2048-token windows): ungated **+14.7 %**, gated **+2.3 %** vs backbone (net-positive on in-domain held-out text). Supersedes the v0.2.0 indicative +19.9 %/+49.6 % on 271/42 tokens.
- **TriviaQA with error bars.** **n=1000**, gate v6 re-trained on 3 seeds: backbone 53.4 % → ungated 45.2 % → **gated 52.5 % ± 1.74** (−0.9 pt, ~89 % of the loss recovered).
- **Recall-vs-#facts scaling curve (multi-seed).** 100 % synthetic recall, std 0, across seeds {137, 7, 23} at **100 / 300 / 1000** facts, plus the 5000-fact production model at 100 %. Convergence cost is roughly **constant at ~30 exposures per fact** (steps scale linearly with #facts); **no capacity wall** up to 5000 facts on a 50k pool. Replaces the v0.2.0 micro-only (N=100) reproducibility claim.

### Added
- **500k pool unblocked.** The 500k ROCm allocation failure was traced to the dense pool gradient (~3.3 GB), not activations. A **sparse pool gradient** (`F.embedding(..., sparse=True)`) with an offloaded optimizer that consumes it trains a ~500k pool at **100 % recall, 20.2 GB VRAM**. Practical ceiling 50k → 200k → **500k**. (At 1M the limiter becomes the pool parameter itself, ≈7.2 GB bf16 — awaiting larger hardware.)

### Notes
- A **multi-domain relevance gate** (with a held-out-phrasing generalisation test) is validated and will ship as **v0.2.2**.
- Gate implementation code remains planned for a later release.

## v0.2.0 — Sprint 0 consolidation (2026-06)

### Added
- `docs/SPRINT0.md`: the hidden native-knowledge regression, the learned relevance gate (six iterations), multi-seed reproducibility, and pool-scaling findings.

### Changed (honest corrections & new results)
- **Native-knowledge regression documented.** The v0.1 claim "native preserved 100 % → 100 %" was a greedy-recall artifact. Formal metrics show the always-on memory taxes general competence: PPL +19.9 % (neutral) / +49.6 % (factual), TriviaQA ~43 % → ~32 % (single run, small set — indicative).
- **Relevance gate fix.** A small learned per-token gate (~0.5M params/layer, backbone *and* memory frozen, only the gate trains) recovers it: PPL −0.5 % vs backbone, synthetic recall 100 %, TriviaQA 40.0 % → 47.3 % (vs 50.0 % backbone, n=300) — ~73 % of the loss recovered.
- **Multi-seed reproducibility.** Recipe validated on seeds {137, 7, 23}: 100 % synthetic recall on 3/3, standard deviation 0. The v0.1 "single run, no multi-seed" caveat is lifted at micro-scale.
- **Pool ceiling 50k → ~200k.** The offloaded optimizer *does* train the pool (the original 0 % was loss + packing, not the optimizer). Dense is VRAM-capped ~50–100k; offload reaches 200k at 100 % recall. 500k pending a ROCm HSA allocator fix.
- **Phase E → Phase F.** Signal/ghost fusion was found unrealizable in its initial formulation; reframed as Phase F (open to Vector Symbolic Architectures or similar), off the critical path.

### Notes
- Gate implementation code is planned for a later release (v0.3).
- Donations now available via GitHub Sponsors (Sponsor button at the top of the repository).

## v0.1.0 — Initial release
- Phase A→D reconstruction of Memory Layers integrated into a frozen Qwen2.5-7B on a single 24 GB consumer GPU; the working warm-up recipe; diagnostic of the 0 %→100 % recall fix.
