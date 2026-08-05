# Changelog

## Reproducibility fix — deterministic corpus seeding (2026-07)

**Hygiene fix. This commit changes no published metric — only seeding determinism and this note.**

`src/synthetic_facts.py` seeded its RNG with `random.Random(hash((fam, seed)) & 0xffffffff)`. Python salts `hash()` of strings per process (PYTHONHASHSEED), so the five structured-family synthetic corpus produced by `gen_facts` **varied on every run and was never bit-reproducible across processes** — including for the published seeds {137, 7, 23}.

- **Fix.** `hash((fam, seed))` → `int(hashlib.sha256(f"{fam}|{seed}").hexdigest()[:8], 16)` — deterministic cross-process; verified to give identical corpus fingerprints under PYTHONHASHSEED ∈ {unset, 0, 1, 2}.
- **Scope.** Fixes the **five structured families** in `src/synthetic_facts.py`. The natural-language ("bio") family referenced in the v0.3.1 generalisation results is produced by a harness **not included in this repository**; the same determinism caveat applies to those NL-specific numbers until that harness is released.
- **Consequence.** The regenerated corpus differs from previous draws (none of which were reproducible).
- **In progress.** The relevance-gate statistics reported with a ±σ "over seeds {137, 7, 23}" (e.g. TriviaQA-with-gate) were computed under the old salted regime and are being re-derived under the stable seed (following amendment). A diagnostic re-derivation under the stable seed confirmed no headline or gate conclusion changes; inter-seed variance is negligible. The forthcoming amendment refines the reported error bars, it does not revise any conclusion.
- **Headline recall is unaffected.** The warm-up / recall pipeline (`src/data/make_synthetic.py`, `src/data/make_corpus.py`) uses integer seeds and was already reproducible.

## v0.6.0 — Attribute normalizers: deterministic R1/R2 verification (embedding-free) (2026-08)

**Minor release — new feature.** Adds three character-level, **embedding-free** normalizers that decide whether two attribute values are the *same* by canonical form — the value-level companion to the v0.5.0 membership index (which decides *stored vs novel* at the identifier level). No network, no embedding, no training.

- **M3 — versions / IP / dates.** `X.Y.Z` is shared by software versions and IP addresses; the rule types by structure (a 4-octet all-≤255 string is an IP; otherwise a version; ISO dates separately), so an IP can never match a version.
- **M1 — units, typed by dimension.** Each unit carries its physical dimension; two values match iff same dimension and same base value. Never relate two units by a shared prefix (`100 MHz` ≠ `100 MB`) or a shared root (`dB` ≠ `dBm` ≠ `dBc`); intra-dimension scaling is exact (`1 GHz` = `1000 MHz`).
- **M4 — acronyms, exact closed table.** A closed, auditable `sigle ↔ exact-expansion` table (bilingual), no fuzzy matching (short codes one letter apart stay distinct), and an acronym used as a *concept* (a gloss) is excluded by design.

**Frontier shipped with the mechanism (`docs/ATTRIBUTE_NORMALIZERS.md`, limits first).**
1. **deterministic R1/R2 only, not semantic** — a semantically equivalent but non-format-reducible value is out of scope (a reported false-refusal, never guessed);
2. **why not semantic — measured:** a semantic-embedding validator was tested on a labelled bench and is *worse than nothing* on technical attributes — **AUC 0.32** (inverted): an embedder rates a wrong-value neighbour (`v4.1.0` vs `v4.1.1`, cosine ≈ 0.93) as *more* similar than a true paraphrase, because the last digit — the bit that separates right from wrong — is exactly what a meaning-encoder discards. The normalizer distinguishes them perfectly (0 % false-accept); an embedder is never a substitute for value verification;
3. **closed, auditable table** — gloss (acronym-as-concept) excluded, checked by the dry-run;
4. **not a general parser** — it canonicalises and compares a value already extracted for a known attribute.

The criterion is false-accept-governed (**≤ 1 %**, 0 % on the trap pairs); coverage is reported as a result (non-reducible values are false-refusals, owned). In each normalizer the test on the trap pairs is written *before* the rule.

**Status — validated on a bench, not integrated.** This is a characterised mechanism (0 % false-accept on the trap pairs); there is **no attribute-verification step wired into the running assistant** — not a deployed faithfulness check.

### Added
- `src/attribute_normalizers.py` — the three normalizers + a self-contained dry-run on generic example data (`python src/attribute_normalizers.py`). The acronym table and trap pairs are generic and public.
- `docs/ATTRIBUTE_NORMALIZERS.md` — frontier-first documentation and the pre-registered criterion.

### Notes
- All shipped data is a generic public example; no acronym table, trap pairs, specs, or tokens from any private corpus are included.

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
2. **Generation budget too small (`mx=12`).** `node_coord` values reach **15 tokens** — confirmed as the theoretical maximum of the bounded value format over 20 000 draws/family plus hand-crafted signed extremes (all other families ≤ 10 tokens). `mx=12` truncated the longest coordinates. Raised to **`mx=24`** (margin 9 over the 15-token max; consistent with `recall_trivia`).

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

A self-correction release. Two follow-up experiments (re-using the frozen 6-family memory, all data synthetic) sharpen and in one case **retract** earlier claims.

### Changed — `docs/GENERALIZATION.md` §3 corrected; new `docs/SAFETY_EVAL.md`
- **The v0.3.1 "emergent entity-level safety" / "decode-confidence collapse 1.00 → 0.66" claim is corrected.** It rested on **40** fake entities. Re-measured **in-process** at **n = 360** (with a stored-recall = 1.000 sanity gate; an earlier checkpoint-reload gave recall ≈ 0 for stored facts too and was discarded as a measurement artifact):
  - decode confidence: stored **1.00**, fake **0.91** (left tail to 0.15); **AUC stored-vs-fake = 0.69** — a *partial*, not categorical, signal (~half of fakes show reduced confidence).
  - On a non-stored entity the model **confidently fabricates a plausible, format-correct, wrong value** (e.g. fake config_param → `614`, fake proto → `0x76`) — it does **not** abstain.
  - The accurate property is **no inter-fact leakage** (`fake_eq ≈ 0`: it never reproduces the unseen value or pulls a stored fact), **not** "no hallucination".
- **Framing updated:** the gate is domain-level; the memory does **not** filter unknown entities; only "no inter-fact leakage" holds. `docs/THREATS.md` updated accordingly (replaces the "no hallucination is empirical" bullet).
- **NL-family scope caveat (v0.3.1 §2):** with a single NL family we cannot distinguish *per-cluster* from *per-family* coverage; the supported claim is the narrower "held-out NL **entities** recover once the NL family is seen". A second NL family would settle it.

### Changed — `docs/BASELINES.md` / `docs/RELATED_WORK.md` (kNN-LM)
- The v0.3.2 kNN-LM **0 %** used a *declarative* datastore. v0.3.4 re-ran kNN-LM with a **Q&A-format datastore** (key = hidden state over the answer span of a `question → answer` sequence): recall **0 % up to λ=0.5, 2 % at λ=0.7, 13 % at λ=0.9** (n=100). The format fix lifts the artifactual 0 %, but kNN-LM only recalls anything when the kNN term nearly overrides the LM (λ=0.9) — the regime that wrecks the general distribution — and tops out at 13 % vs ~100 %. Documented as **structurally inadequate** here, not a strawman.

### Notes
- Net effect: the safety story is **weaker but true** (domain-level gate, no inter-fact leakage, partial confidence signal), and the kNN-LM baseline is now a measured curve rather than a single artifactual 0 %. Reproduce: `python a2_inproc.py` (Angle 2), `python c2_knnlm_qa.py` (Angle 4). All data synthetic; backbone Qwen2.5-7B frozen.

## v0.3.3 — Related work + threats to validity (2026-06)

Positions the contribution in the literature and lists, without minimising, the limits. Closes the R-Arxiv preprint-preparation series (C1–C5).

### Added — `docs/RELATED_WORK.md`
- **Memory layers** — Lample et al. 2019 (product keys); Berges et al. 2024 (the substrate this repo reconstructs).
- **Closest concurrent work — Sparse Memory Finetuning** (Lin et al. 2025, arXiv:2510.15103, Meta FAIR): *sparsely updates* memory slots (NaturalQuestions F1 drop 89 % full-FT / 71 % LoRA / **11 % sparse-memory-FT**). We instead keep backbone **and** memory frozen and add an inference-time gate — complementary; their LoRA-forgetting numbers corroborate our LoRA baseline.
- **Active Reading** (Lin et al. 2025, arXiv:2508.09494); retrieval methods (RAG, kNN-LM, Memorizing Transformers); PEFT (LoRA; Biderman et al. 2024 "LoRA learns less and forgets less" — regime-dependent).
- **Our position:** retrofit setting (frozen backbone + frozen memory + a learned domain-relevance gate) — neither weight-edit nor retrieval.

### Added — `docs/THREATS.md`
- Synthetic facts with marked signatures; a single simple NL family; closed fact set. One backbone / one checkpoint per scale; hardware-specific ceilings (ROCm/WSL2, 24 GB). Baselines at differing scales and **not exhaustively tuned** (kNN-LM's 0 % is a declarative→Q&A datastore-format artefact, not an upper bound). "No hallucination" is empirical to this memory's retrieval geometry; the gate is domain-level. What would most strengthen the work: a head-to-head with Sparse Memory Finetuning, real facts, more backbones, larger-n benchmarks.

### Notes
- Closes the R-Arxiv series (C1 gate code → C3 generalisation → C2 baselines → C4 stats → C5 related work + threats). All data synthetic; backbone Qwen2.5-7B frozen.

## v0.3.2 — External baselines: RAG, LoRA, kNN-LM (2026-06)

Compare the frozen-backbone Memory-Layers-+-gate approach to standard fact-injection methods on the same synthetic facts and metrics. See `docs/BASELINES.md`.

### Added — `docs/BASELINES.md`
- **RAG** (BM25 sparse, top-1 injected): recall **99.4 %**, **non-destructive** (general PPL / TriviaQA = backbone), but a retrieval index + injected context on every query; knowledge stays out of the weights.
- **LoRA** (r=16, all-linear, answer-only): recall 82.7 % (1000 facts) but **catastrophic forgetting** — TriviaQA 53.4 % → **0.7 %**, WikiText PPL **+45 %**.
- **kNN-LM** (datastore of hidden→next-token, interpolation λ): **fails the Q&A recall (0 %)** because declarative datastore keys don't match the question's hidden state, while the fact-only datastore taxes the general distribution (PPL +15 %→+70 %, TriviaQA 42 %→8 % as λ grows). *(v0.3.4: a Q&A-format datastore lifts this to ≤13 % at λ=0.9 — still structurally inadequate.)*
- **Takeaway:** only Memory-Layers-+-gate reaches ~100 % recall **and** preserves general competence (PPL +2.3 %, TriviaQA −0.6) **and** stays parametric (no retrieval, no per-query context).

### Added — `docs/STATS.md` (statistical calibration)
- **Binomial CIs.** At n=1000 the 95 % TriviaQA half-width is ±3.1 pts. The **ungated** regression (−8.2) is significant (CI-disjoint from backbone); the **gated** residual (−0.6 / −0.9) is **within sampling noise** → restated as "no statistically significant regression" rather than "recovers 89 %".
- **σ clarified.** The reported ±1.74 / ±0.28 are inter-seed gate-training reproducibility, **not** the sampling standard error (≈±1.58 at n=1000); both reported.
- PPL (220k tokens) is a population quantity; recall is at the ceiling (Wilson [99.6 %, 100 %] at n=1000).

### Notes
- Memorizing Transformers (Wu et al. 2022) is covered in Related Work rather than reimplemented.
- All data synthetic; backbone Qwen2.5-7B frozen.

## v0.3.1 — Generalisation frontier of the relevance gate (2026-06)

Reinforced generalisation battery (held-out entities, leave-one-family-out, negative control + retrieval geometry). See `docs/GENERALIZATION.md`. *(§3 "emergent entity-level safety" corrected in v0.3.4 — see `docs/SAFETY_EVAL.md`.)*

### Added — `docs/GENERALIZATION.md`
- **Held-out entities: the gate generalises (Δ 0).** Gate trained on 80 % of entities of every family, evaluated on the held-out 20 %: recall gated = ungated for all six families, **including the natural-language one**. Answers "held-out only covers phrasings".
- **Distribution-level, not entity-level (leave-one-family-out).** The five structured families form one cluster and transfer to each other (Δ ≤ 5 when held out); the NL family held out collapses (Δ −93). A **coverage** limit (per-family held-out entities are fine once the family is seen), not fundamental. *(v0.3.4 caveat: a single NL family cannot settle per-cluster vs per-family.)*
- **Entity behaviour on non-stored entities.** *(Corrected in v0.3.4: the original "emergent safety via decode-confidence collapse 1.00 → 0.66 / no hallucination" was an n=40 artifact; at n=360 the signal is partial (AUC 0.69) and the model confidently fabricates plausible wrong values — only "no inter-fact leakage" holds. See `docs/SAFETY_EVAL.md`. v0.4.1: no internal detector recovers entity-level OSR either — see `docs/OPENSET_MP_TYLER.md`.)*
- Gate closes on general prose (open-rate 0.0005), the perplexity benefit.

### Framing
- The contribution is **domain-relevance gating**: the gate detects the distributional signature of a stored-fact context. The NL-family frontier is reported, not hidden.

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
- **Phase E → Phase F.** Multi-signal fusion for retrieval augmentation was found unrealizable in its initial formulation; reframed as Phase F (open to Vector Symbolic Architectures or similar), off the critical path.

### Notes
- Gate implementation code is planned for a later release (v0.3).
- Donations now available via GitHub Sponsors (Sponsor button at the top of the repository).

## v0.1.0 — Initial release
- Phase A→D reconstruction of Memory Layers integrated into a frozen Qwen2.5-7B on a single 24 GB consumer GPU; the working warm-up recipe; diagnostic of the 0 %→100 % recall fix.
