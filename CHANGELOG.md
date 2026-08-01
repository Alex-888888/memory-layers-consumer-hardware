# Changelog

## Reproducibility fix — deterministic corpus seeding (2026-07)

**Hygiene fix. This commit changes no published metric — only seeding determinism and this note.**

`src/synthetic_facts.py` seeded its RNG with `random.Random(hash((fam, seed)) & 0xffffffff)`. Python salts `hash()` of strings per process (PYTHONHASHSEED), so the five structured-family synthetic corpus produced by `gen_facts` **varied on every run and was never bit-reproducible across processes** — including for the published seeds {137, 7, 23}.

- **Fix.** `hash((fam, seed))` → `int(hashlib.sha256(f"{fam}|{seed}").hexdigest()[:8], 16)` — deterministic cross-process; verified to give identical corpus fingerprints under PYTHONHASHSEED ∈ {unset, 0, 1, 2}.
- **Scope.** Fixes the **five structured families** in `src/synthetic_facts.py`. The natural-language ("bio") family referenced in the v0.3.1 generalisation results is produced by a harness **not included in this repository**; the same determinism caveat applies to those NL-specific numbers until that harness is released.
- **Consequence.** The regenerated corpus differs from previous draws (none of which were reproducible).
- **In progress.** The relevance-gate statistics reported with a ±σ "over seeds {137, 7, 23}" (e.g. TriviaQA-with-gate) were computed under the old salted regime and are being re-derived under the stable seed (following amendment). A diagnostic re-derivation under the stable seed confirmed no headline or gate conclusion changes; inter-seed variance is negligible. The forthcoming amendment refines the reported error bars, it does not revise any conclusion.
- **Headline recall is unaffected.** The warm-up / recall pipeline (`src/data/make_synthetic.py`, `src/data/make_corpus.py`) uses integer seeds and was already reproducible.

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

**Known debt — recall metric.** Running the released `train_relevance_gate.py` end-to-end currently reports **~82 % generative recall** on the multi-domain held-out-phrasing table (which lists 100 %); a metric correction is in progress and will be published separately, with measurement. This does not affect the 0-point gated-vs-ungated gap or the gate metrics above.

Docs amended: `README.md` (version badge + multi-domain figures + single-domain labelling + recall note), `docs/GATE_MULTIDOMAIN.md`, `docs/STATS.md`. No source or data change; documentation-only.

## v0.4.3 — R-Reliable extended: an upstream pre-normalization density filter also fails (2026-07)

**Fifth signal family refuted.** A density / energy filter reading the query *before* qk-normalization — in the raw Euclidean space where magnitude still exists — is tested against internal entity-level open-set recognition. It fails like the first four. `docs/OPENSET_MP_TYLER.md` extended with §11 (and abstract/conclusion updated to five families); no code behaviour changed for the earlier phases. All in-process; every phase gated on stored-recall = 1.000; all data synthetic.

### Added — `docs/OPENSET_MP_TYLER.md` §11 (upstream pre-normalization density)
- **Motivation.** qk-normalization (`F.normalize(q) @ F.normalize(C)`) discards the *magnitude* of the query and codebook before the dot product; the direction is already known not to separate (§1, §3). The one untested channel is the raw Euclidean geometry upstream of the normalization — the natural home of a density / energy-based OOD filter (Ren et al. 2019; Nalisnick et al. 2019; LeCun et al. EBM).
- **Measurement.** Pre-normalization query halves captured at the answer-position token, 360 stored vs 360 fake (disjoint seed), at gated layers 6/14/22, against the raw codebook rows: (F1) query magnitude ‖q‖ AUC **0.506–0.514**; (F2) **nearest-codebook-key Euclidean distance (raw)** — the one axis qk-norm discards — AUC **0.507–0.514**; (F2n) normalised control ≈ **0.50** ✓; (F3) reconstruction residual on the codebook PCA span AUC **0.501–0.519**. **Best over all features/layers = 0.5185.** Stored and fake are indistinguishable to the 3rd–4th significant figure.
- **Codebook structure.** 0 eigenvalues above the Marchenko–Pastur edge; participation ratio ≈ 165–194 of ≈ 224 available directions (near-isotropic) — no low-dimensional manifold for a linear autoencoder to reconstruct against.
- **Interpretation.** The "is-stored" bit is not merely *destroyed* by qk-normalization; it **was never in the query geometry**. The frozen backbone processes a fake entity (an ordinary real token) identically to a stored one, and the query projection — trained only on stored positives — built no rejection geometry. This is the fifth family to fail and it closes the internal-detection line: there is no stage of the memory read (addressing, value assembly, output distribution, downstream residual, upstream query) at which an entity-level signal is measurable.

### Notes
- References added, public-literature only: Ren et al. 2019 (arXiv:1906.02845), Nalisnick et al. 2019 (arXiv:1810.09136), LeCun et al. 2006 (Tutorial on Energy-Based Learning). Prior refs unchanged.
- Reproduce: `python r_reliable_density.py` (pre-qk-norm magnitude / nearest-key distance / reconstruction residual, codebook MP spectrum, sanity gate).
- **Net effect.** Five internal signal families now fail at a stored-recall = 1.000 sanity gate. Detecting stored-vs-fabricated *within* this frozen product-key memory is intractable at every read stage; reliable abstention requires an external retrieval/membership check. Creating such a signal — rather than detecting it — would mean leaving the frozen-backbone setting (a different architecture, out of scope for this repository).

## v0.4.2 — R-Safety extended: supervised probes (linear + LoRA) also fail (2026-07)

**Extension of the v0.4.1 structural finding.** A fourth signal family — supervised probes on the residual stream, including the LoRA-probe headline method of Obeso et al. (2025) — is now tested against internal entity-level open-set recognition. It fails like the first three. `docs/OPENSET_MP_TYLER.md` extended with sections 4–6 and a methodological note; no code behaviour changed for the earlier phases. All in-process; every phase gated on stored-recall = 1.000; all data synthetic.

### Added — `docs/OPENSET_MP_TYLER.md` §4–§5 (supervised probes) + §6, §10
- **Phase 1-bis-E — supervised linear probe (AUC 0.685).** A logistic probe on the full residual-stream hidden state (d = 3584) at the answer position, sweeping five downstream layers [14, 22, 24, 26, 28], trained with stored-vs-fabricated labels. **Entity-disjoint split (252 entities/class train, 108/class held-out; no entity in both)** as the essential control against identity memorisation. Best layer L26 reaches **0.685** on held-out entities — *below* the free decode-confidence baseline (0.696) — with a train AUC of 0.952, the signature of the memorisation the disjoint split exposes. No entity-general "stored-vs-fabricated" direction in the residual stream.
- **Phase 1-bis-E-LoRA — LoRA-probe (AUC 0.622).** The Obeso 2025 headline method: a low-rank adapter (`r=16, α=32, target self_attn.{q,v}_proj`, **5.0 M params = 0.065 %**) trained jointly with the probe head so the network can reshape its own activations, with a KL term preserving behaviour. **Final KL = 0.0000; sanity stored-recall = 1.000 with the LoRA active** (identical to the no-LoRA gate) — the adapter did not damage the memory, it had nothing more to expose. Held-out **AUC 0.622**, *lower* than the linear probe (0.685).
- **Decisive comparison.** On this architecture: linear **0.685** / LoRA **0.622**. Obeso 2025 on general LLMs: linear **0.867** / LoRA **0.905**. Adding capacity finds *no* additional signal (LoRA < linear < decode-conf), and the **0.28-point LoRA gap** shows the intractability is specific to product-key memory with a frozen backbone and qk-normalisation — not a general limitation of activation probing.
- **Unified mechanism.** All four families fail for one reason: the memory produces **confident, deterministic, self-consistent fabrications** (63 % of unknown entities yield a single value across 8 samples). A deterministic confident fabrication produces **no measurable internal signal, regardless of probe capacity** — there is no uncertainty because the model is not uncertain, it is confidently wrong in a stable way.
- **Methodological note (§10).** The entity-disjoint split is recorded as an **essential control** for any supervised open-set / hallucination probe on parametric memory: split by the stored unit, not by phrasing or example, or the metric measures memorisation rather than generalisation.

### Notes
- References are public-literature only, adding for this release: Obeso et al. 2025 (arXiv:2509.03531), Azaria & Mitchell 2023 (arXiv:2304.13734), Zou et al. 2023 (arXiv:2310.01405), Marks & Tegmark 2023 (arXiv:2310.06824). Prior refs unchanged.
- Reproduce: `python r_reliable_probe.py` (linear, layer sweep, entity-disjoint), `python r_reliable_lora.py` (LoRA-probe, KL-preserved, sanity with LoRA active). All in-process; synthetic; backbone Qwen2.5-7B frozen.
- **Net effect.** All four families tested against internal entity-level open-set recognition are equally intractable on product-key memory with a frozen backbone and qk-normalisation. Supervised probing — the strongest form, including capacity-augmented LoRA-probes — does not change the v0.4.1 conclusion: reliable abstention requires an external retrieval/membership check.

## v0.4.1 — Sprint R-Safety completed: open-set recognition is empirically intractable internally (2026-07)

**Structural finding: internal entity-level open-set recognition on a frozen parametric product-key memory is empirically intractable; external RAG-style verification is identified as the only reliable abstention mechanism.** New characterisation doc `docs/OPENSET_MP_TYLER.md`. No code added — this release is a characterisation deliverable. All in-process; every phase gated on stored-recall = 1.000; all data synthetic.

### Added — `docs/OPENSET_MP_TYLER.md`
- **Phase 1 — routing geometry (AUC ≈ 0.50).** Eight product-key routing features (`s_max`, margins, variance, per-sub-quantiser tops, top-k entropy, memory-output norm) × 3 layers, stored vs fake at n=360: all **0.50–0.53**. Cause: qk-normalisation makes the routing scores pure cosines, so a fake same-structure entity retrieves its nearest keys with the same distribution — the "is-stored" bit is absent from the addressing geometry.
- **Phase 1-bis A — value-space geometry (AUC ≈ 0.50).** A normalisation-invariant **Tyler M-estimator** (Tyler 1987) of the value-pool scatter, then per-entity Mahalanobis and retrieval coherence. All **0.50–0.52**. The **"incoherent retrieval" hypothesis is refuted**: coherence is **1.00 for fakes as for stored** (identical to 4 decimals) — a fake produces a perfectly coherent but wrong value. Spectral characterisation: Marchenko–Pastur bulk with the corrected **γ = d/N = 3584/50176 = 0.0714** (587 spike eigenvalues, no clean low-energy null-space) — a generic-watermarking feasibility check, not pursued.
- **Phase 1-bis A2+ — semantic entropy (AUC 0.66).** Farquhar et al. (Nature 2024): k=8 temperature-sampled generations clustered by meaning. Stored entropy 0.00 / fake 0.27 → AUC **0.6625**, *below* the cheap proxies (3-phrasing 0.69, decode-confidence 0.67). **63 % of unknown entities yield a single value across 8 samples** — the model fabricates one confident, self-consistent value, so there is little epistemic uncertainty to detect (hidden-state variants such as EigenScore/INSIDE, semantic-entropy probes and SelfCheckGPT read the same signal and are not expected to recover one that is absent from both geometry and behaviour).
- **Conclusion + recommendation.** Entity-level OSR is empirically intractable by internal or behavioural measurement on this architecture (the "is-stored" information is not present, geometrically or behaviourally). Reliable abstention requires an **external** membership/retrieval check; the natural design is a hybrid (parametric memory for recall + a cheap external lookup for abstention), consistent with the parametric-vs-retrieval trade-off already documented in `docs/BASELINES.md`.

### Notes
- References are public-literature only: Berges 2024 (arXiv:2412.09764), Lample 2019 (arXiv:1907.05242), Marchenko–Pastur 1967, Baik–Ben Arous–Péché 2005, Tyler 1987, Cox et al. 1997, Farquhar et al. 2024 (Nature), Chen et al. 2024 (INSIDE/EigenScore, ICLR, arXiv:2402.03744), Kossen et al. 2024 (arXiv:2406.15927), Manakul et al. 2023 (SelfCheckGPT, EMNLP, arXiv:2303.08896).
- Reproduce: `python r_safety_probe.py` (routing), `python r_safety_1bisA.py` (Tyler + MP spectrum), `python r_safety_a2plus.py` (semantic entropy). All in-process; synthetic; backbone Qwen2.5-7B frozen.

## v0.3.4 — Honest correction on the safety claim + kNN-LM Q&A baseline (2026-06)

See repository history for full v0.3.4 and earlier entries. (Older entries unchanged by this commit.)
