# Changelog

## Reproducibility fix — deterministic corpus seeding (2026-07)

**Hygiene fix. This commit changes no published metric — only seeding determinism and this note.**

`src/synthetic_facts.py` seeded its RNG with `random.Random(hash((fam, seed)) & 0xffffffff)`. Python salts `hash()` of strings per process (PYTHONHASHSEED), so the five structured-family synthetic corpus produced by `gen_facts` **varied on every run and was never bit-reproducible across processes** — including for the published seeds {137, 7, 23}.

- **Fix.** `hash((fam, seed))` → `int(hashlib.sha256(f"{fam}|{seed}").hexdigest()[:8], 16)` — deterministic cross-process; verified to give identical corpus fingerprints under PYTHONHASHSEED ∈ {unset, 0, 1, 2}.
- **Scope.** Fixes the **five structured families** in `src/synthetic_facts.py`. The natural-language ("bio") family referenced in the v0.3.1 generalisation results is produced by a harness **not included in this repository**; the same determinism caveat applies to those NL-specific numbers until that harness is released.
- **Consequence.** The regenerated corpus differs from previous draws (none of which were reproducible).
- **In progress.** The relevance-gate statistics reported with a ±σ "over seeds {137, 7, 23}" (e.g. TriviaQA-with-gate) were computed under the old salted regime and are being re-derived under the stable seed (following amendment). A diagnostic re-derivation under the stable seed confirmed no headline or gate conclusion changes; inter-seed variance is negligible. The forthcoming amendment refines the reported error bars, it does not revise any conclusion.
- **Headline recall is unaffected.** The warm-up / recall pipeline (`src/data/make_synthetic.py`, `src/data/make_corpus.py`) uses integer seeds and was already reproducible.

## v0.5.0 — External membership-verification index (embedding-free abstention) (2026-08)

**Minor release — new feature.** Adds an external, auditable, **embedding-free** membership index a system can run alongside the parametric memory to decide *stored vs novel* and abstain instead of fabricating — the external check identified as the only reliable path in v0.4.1 (internal open-set recognition is intractable). Pure string-matching: a form-based regex router, an identifier extractor, and exact + fuzzy membership against a phrasing-invariant index. No network, no embedding, no training.

**Frontier shipped with the mechanism, not after it (`docs/EXTERNAL_VERIFICATION.md`, limits first).** Four hard limits, none softened:
1. verifies **identifier membership, not attribute truth** — a fake with a real identifier and a false attribute is accepted; the demo makes this **executable** (printed on every run);
2. **lexical, not semantic** — a name-substitution control gives **AUC 0.526** (≈ chance) for substituted-vs-fake;
3. **synthetic only** — semantic verification on real entities is out of scope here;
4. **assumes a controlled input schema** — the natural-language pattern is deliberately broad and would mis-route arbitrary free text.

The PASS (0 % decisional false-accept, 0 % false-refusal on the synthetic corpus) is stated only together with the frontier. Pool collisions (a fake whose identifier is already stored by chance in a finite corpus) are excluded from the decisional false-accept and reported separately. The fuzzy path is exercised on every novel NL miss and reports its trigger count and closest ratio, so a 0 is visibly "flagged nothing" rather than "never ran".

### Added
- `src/membership_index.py` — the mechanism + a self-contained demo on synthetic data (`python src/membership_index.py`). The NL family is generated in-file from generic invented name pools (no real people).
- `docs/EXTERNAL_VERIFICATION.md` — frontier-first documentation and the pre-registered criterion.

### Notes
- All data synthetic; no corpus, seeds, or entities from any private measurement are included.

## v0.4.5 — Recall-metric debt closed: per-family generative recall = 100 % (2026-08)

**Patch release. Closes the recall-metric debt flagged in v0.4.4 on a measured number; no gate metric changed.** The released `train_relevance_gate.py` `recall()` had **two stacked measurement artifacts**, now both fixed in code:

1. **Asymmetric whitespace strip.** Spaces were stripped from the generated text but not the target value, so any space-containing value could never match — the `node_coord` family (values like `48.21, -3.55`) was zeroed by the metric regardless of generation. Fixed to strip both sides (matching the `p_c3` harness).
2. **Generation budget too small (`mx=12`).** `node_coord` values reach **15 tokens** — confirmed as the theoretical maximum of the bounded value format over 20 000 draws/family plus hand-crafted signed extremes (all other families ≤ 10 tokens). `mx=12` truncated the longest coordinates. Raised to **`mx=24`** (margin 9 over the 15-token max; consistent with `recall_trivia`).

**Measured closure (seed 137, held-out phrasing D, ungated memory).** Per-family generative recall (corrected metric + adequate budget) is **100 % on all five families**, identical to the exact / teacher-forced membership recall:

| family | exact / membership | generative (mx=24) |
|---|---|---|
| sensor_calib | 100 % | 100 % |
| config_param | 100 % | 100 % |
| service_ver | 100 % | 100 % |
| node_coord | 100 % | 100 % |
| proto_status | 100 % | 100 % |

`node_coord` generative recall moved 0 % → 30 % (whitespace fix) → **100 %** (budget fix). A raw-generation dump (10/10) showed every value regenerated **exactly** (`13.64, 69.84` → `13.64, 69.84.`) with **no digit drift** — the residual failures were purely truncation. The memory **regenerates faithfully; there is no regeneration limit**. Recall is a substring test (`value in generated`), robust to the trailing `.` the model appends. The published **100 %** (membership) is thus corroborated generatively.

**No documentation number was wrong.** The debt was entirely two metric-measurement artifacts, not a memory or gate defect. Docs updated: `README.md` (badge + note), `docs/GATE_MULTIDOMAIN.md` (debt note replaced by the closed statement). `src/train_relevance_gate.py` carries the two code fixes.

## v0.4.4 — Reproducible multi-domain gate error bars (stable-seed re-derivation) (2026-08)

**Patch release. Published means are unchanged within sampling noise; no conclusion is revised.** Building on the reproducibility fix above (HEAD `77d5a40`), the relevance-gate metrics are re-derived under the deterministic corpus seed, at the exact published protocol (memory target_exp=40, TriviaQA n=1000, WikiText-103 220k tokens, gate seeds {137, 7, 23}).

**Lead result — the seed-independent anchor.** The frozen backbone reproduces TriviaQA **53.4 % identically across all three seeds** (σ = 0). Having no gate and no synthetic corpus, this identity *mechanically* attributes the entire between-seed dispersion of the gated numbers to **evaluation sampling noise**, not corpus or seed variance — for every gate configuration.

**Multi-domain gate (released `train_relevance_gate.py`, 5 families) — re-derived under the stable seed:**
- TriviaQA gated: **52.5 % ± 0.6** (n=1000, 3 seeds), −0.9 pt vs 53.4 % backbone.
- gate open-rate: **0.933 ± 0.004** (trained phrasings) / **0.943 ± 0.008** (held-out phrasing).
- held-out-phrasing recall drop: **0 pt (3/3)**.
- PPL — WikiText-103 (220k tok) gated: **+0.82 % ± 0.07**.

The previously reported multi-domain bar (± 0.28, v0.2.2) was computed on the **non-reproducible salted corpus** (a different draw per process, never regenerable), which made it artificially narrow. The ± 0.6 here is **not a looser bar — it is the replacement of a non-reproducible value by the true inter-seed dispersion under a stable corpus**, and it confirms the same conclusion: the residual sits below the n=1000 sampling floor (± 1.58 pt, 1σ). Same configuration, honest bar.

**Single-domain gate v6 (v0.2.1: 52.5 % ± 1.74, PPL +2.3 %) is left unchanged** — a *different configuration* (the earlier single sensor-domain gate), not re-derived by this run and not a different seed regime; it is now labelled as such in `README.md` and `docs/STATS.md`. This release does not revise it.

**PPL absolute realigned for protocol consistency.** In `docs/GATE_MULTIDOMAIN.md` the WikiText-103 backbone/gated PPL absolutes are updated from 7.24 / 7.32 to **7.65 / 7.71** to match the 220k-token / 2048-window protocol already used in `docs/SPRINT0.md` and this re-derivation (the earlier 7.24 pair was inconsistent with that protocol); the reported delta moves +1.0 % → +0.82 %.

**Known debt — recall metric.** The released `train_relevance_gate.py` recall metric has been **corrected in code** (symmetric whitespace strip; the space-containing `node_coord` family was previously zeroed by an asymmetric strip). The corrected per-family recall is **being measured and will be published**. This does not affect the 0-point gated-vs-ungated gap or the gate metrics above. *(Closed in v0.4.5 above: measured per-family generative recall = 100 %.)*

Docs amended: `README.md` (version badge + multi-domain figures + single-domain labelling + recall note), `docs/GATE_MULTIDOMAIN.md`, `docs/STATS.md`. No source or data change; documentation-only.

## v0.4.3 — R-Reliable extended: an upstream pre-normalization density filter also fails (2026-07)

**Fifth signal family refuted.** A density / energy filter reading the query *before* qk-normalization — in the raw Euclidean space where magnitude still exists — is tested against internal entity-level open-set recognition. It fails like the first four. `docs/OPENSET_MP_TYLER.md` extended with §11 (and abstract/conclusion updated to five families); no code behaviour changed for the earlier phases. All in-process; every phase gated on stored-recall = 1.000; all data synthetic.

See the repository history and `docs/OPENSET_MP_TYLER.md` for the full v0.4.3 / v0.4.2 / v0.4.1 detail.

## v0.4.2 — R-Safety extended: supervised probes (linear + LoRA) also fail (2026-07)

A fourth internal signal family — supervised residual-stream probes, including the LoRA-probe of Obeso et al. (2025), on an entity-disjoint split — fails like the first three (linear AUC 0.685 / LoRA 0.622, below the free decode-confidence baseline). See `docs/OPENSET_MP_TYLER.md`.

## v0.4.1 — Sprint R-Safety completed: open-set recognition is empirically intractable internally (2026-07)

**Structural finding: internal entity-level open-set recognition on a frozen parametric product-key memory is empirically intractable; external RAG-style verification is identified as the only reliable abstention mechanism.** Routing geometry (AUC ≈ 0.50), value-space geometry with a normalisation-invariant Tyler estimator (≈ 0.50), and semantic entropy (0.66) are at or near chance under a stored-recall = 1.000 sanity gate. See `docs/OPENSET_MP_TYLER.md`.

## v0.3.4 — Honest correction on the safety claim + kNN-LM Q&A baseline (2026-06)

A self-correction release. The v0.3.1 "emergent entity-level safety / decode-confidence collapse" claim (n=40) is corrected at n=360: the signal is partial (AUC 0.69) and the model confidently fabricates plausible wrong values — the accurate property is **no inter-fact leakage**, not "no hallucination". kNN-LM re-run with a Q&A-format datastore lifts recall to ≤ 13 % (λ=0.9 only) — structurally inadequate, not a strawman. See `docs/SAFETY_EVAL.md` / `docs/BASELINES.md`.

## v0.3.3 — Related work + threats to validity (2026-06)

Positioning against the literature (`docs/RELATED_WORK.md`, incl. Sparse Memory Finetuning, Lin et al. 2025) and an explicit, not-minimised `docs/THREATS.md`. Closes the R-Arxiv preprint-preparation series.

## v0.3.2 — External baselines: RAG, LoRA, kNN-LM (2026-06)

RAG (recall 99.4 %, non-destructive, but retrieval + per-query context), LoRA (catastrophic forgetting: TriviaQA 53.4 → 0.7 %, PPL +45 %), kNN-LM (fails Q&A recall). Only Memory-Layers-+-gate is simultaneously high-recall, competence-preserving and parametric. Binomial CIs and the σ-vs-sampling clarification in `docs/STATS.md`. See `docs/BASELINES.md`.

## v0.3.1 — Generalisation frontier of the relevance gate (2026-06)

Held-out entities (Δ 0, incl. NL), leave-one-family-out (structured cluster transfers, held-out NL family collapses — a coverage limit). *(§3 "emergent entity-level safety" corrected in v0.3.4.)* See `docs/GENERALIZATION.md`.

## v0.3.0 — Relevance-gate code released (2026-06)

First of the R-Arxiv hardening series. `src/relevance_gate.py` (the generic mechanism), `src/synthetic_facts.py` (five synthetic fact families + phrasings), `src/train_relevance_gate.py` (end-to-end train + eval), `docs/RELEVANCE_GATE.md`. All data synthetic.

## v0.2.2 — Multi-domain relevance gate (2026-06)

The gate trained across five fact families with distinct structures + a held-out-phrasing test: gated recall 100 % = ungated (0-point drop), open-rate 0.94 (trained) / 0.95 (held-out). Multi-domain preservation: TriviaQA 52.8 % ± 0.28 (−0.6 pt), PPL +1.0 %. See `docs/GATE_MULTIDOMAIN.md`. *(Error bars re-derived in v0.4.4; recall metric corrected in v0.4.5.)*

## v0.2.1 — Sprint 0 hardening (2026-06)

PPL on WikiText-103 (220k tokens): ungated +14.7 %, gated +2.3 %. TriviaQA n=1000, gate v6, 3 seeds: backbone 53.4 % → ungated 45.2 % → gated 52.5 % ± 1.74. Recall 100 % (std 0) across seeds {137, 7, 23} at 100/300/1000 facts. 500k pool unblocked via a sparse pool gradient (100 % recall, 20.2 GB VRAM). See `docs/SPRINT0.md`.

## v0.2.0 — Sprint 0 consolidation (2026-06)

Documented the hidden native-knowledge regression (the v0.1 "100 % → 100 %" was a greedy-recall artifact), the learned relevance gate (six iterations), multi-seed reproducibility, and pool scaling. See `docs/SPRINT0.md`.

## v0.1.0 — Initial release
- Phase A→D reconstruction of Memory Layers integrated into a frozen Qwen2.5-7B on a single 24 GB consumer GPU; the working warm-up recipe; diagnostic of the 0 %→100 % recall fix.
