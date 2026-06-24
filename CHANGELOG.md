# Changelog

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
