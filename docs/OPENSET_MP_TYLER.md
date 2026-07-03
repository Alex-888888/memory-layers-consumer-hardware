# Open-set recognition on a frozen parametric memory — a negative result (v0.4.1)

**Question.** The relevance gate is *domain-level* (`GENERALIZATION.md`, `SAFETY_EVAL.md`): it opens on a
stored-fact context but does **not** decide whether a *specific queried entity* is actually stored.
On a non-stored entity the model confidently fabricates a plausible, format-correct wrong value
(`SAFETY_EVAL.md`). Can we add a reliable **out-of-support / open-set recognition (OSR)** signal —
"is this entity in memory?" — computed *internally* (from the model's own states / behaviour), so the
system can abstain instead of fabricating?

**Answer (this sprint).** No, not from internal or behavioural measurement. Across three independent
signal families we find OSR AUC at or near chance for the geometric signals and only weakly above
chance for the strongest behavioural one — below any usable operating point. The mechanism is
identified: the memory produces **confident, self-consistent fabrications**, so there is simply no
accessible uncertainty to measure. We conclude that reliable entity-level abstention on this
architecture requires an **external** check, and we recommend a retrieval-based verification.

All experiments are **in-process** on a frozen Qwen2.5-7B + product-key memory (MLP-ADD at layers
6/14/22), six synthetic fact families (360 stored entities), evaluated against same-structure
non-stored ("fake") entities generated with a disjoint seed. **Every phase gates on a
stored-recall = 1.000 sanity check before any AUC is trusted** (a checkpoint-reload path was found to
restore the value pool unfaithfully and was discarded — all numbers below are from a working memory).
All data are synthetic; all methods below are generic.

## 1. Static product-key routing signals — AUC ≈ 0.50 (Phase 1, v0.4.0)

We instrumented the product-key forward and logged, at the answer-position token, per gated layer,
eight routing features: top combined selection score `s_max`, its top-1/top-2 margin, its variance,
the per-sub-quantiser top scores `s1_max`/`s2_max`, the top-k routing entropy `H_topk`, and the
memory-output norm `mem_norm`. Over 360 stored vs 360 fake:

| feature (best over 3 layers + aggregations) | AUC stored-vs-fake |
|---|---|
| `s_max`, `s1_max`, `s2_max`, `s_margin`, `s_var`, `H_topk`, `mem_norm` | **0.50 – 0.53** |
| (reference) first-token decode confidence | 0.67 |

Stored and fake means are identical to several decimals (e.g. `s_max` L6 stored 34.221 vs fake
34.224; `mem_norm` L22 stored 87.9 vs fake 88.2). **Cause:** the product keys and query are
L2-normalised before the dot product (a "qk-normalised" router à la Berges et al. 2024), so the
routing scores are pure cosine similarities. A fake same-structure entity retrieves its nearest keys
with the *same* cosine distribution as a real one — the "is-this-stored" bit is not present in the
addressing geometry. (The earlier small-sample impression that `mem_norm` separated stored from fake
was an n=40 artefact, not reproduced at n=360.)

## 2. Value-space geometry (Tyler / Marchenko–Pastur) — AUC ≈ 0.50 (Phase 1-bis A)

To bypass the magnitude collapse of §1, we used **Tyler's M-estimator of scatter** (Tyler 1987), which
is invariant to per-sample spherical normalisation — the moduli cancel in its fixed point
`Σ ← (d/N)·Σᵢ xᵢxᵢᵀ/(xᵢᵀΣ⁻¹xᵢ)` — so it probes the *angular* structure the router discards. We
estimate a single `Σ_Tyler` from the value-pool rows (N = 50 176 ≫ d = 3584, so the covariance is
well-posed), then, per entity, on the **assembled retrieved value** z:

- Mahalanobis `T(z) = zᵀ Σ⁻¹ z` (raw and magnitude-normalised),
- retrieval **coherence** `‖z‖ / Σᵢ wᵢ‖vᵢ‖` (1 = the top-k retrieved values agree, → 0 = a cancelling
  blend).

| value-space feature | AUC stored-vs-fake |
|---|---|
| Tyler-Mahalanobis `T_raw`, `T_norm`; coherence (all layers/aggregations) | **0.50 – 0.52** |

**The "incoherent retrieval" hypothesis is empirically refuted.** Coherence is **1.00 for fakes as
for stored**, identical to the 4th decimal. A fake entity does *not* produce a cancelling mixture; it
produces a perfectly coherent cluster of values that happens to be the wrong one — which is precisely
why the model fabricates with confidence.

**Spectral characterisation (generic watermarking feasibility).** The pool value covariance follows a
Marchenko–Pastur (1967) bulk with the corrected ratio **γ = d/N = 3584/50176 = 0.0714**
(√γ = 0.267; the Baik–Ben Arous–Péché 2005 detectability threshold). We observe **587 eigenvalues
above the upper MP edge** (trained "spike" directions) and **0 eigenvalues below the lower edge**
(no clean low-energy null-space). This was assessed to judge whether a generic injected
spread-spectrum signature (Cox et al. 1997) could later create a detectable read-time signal; given
(a) the refuted incoherence, (b) the absence of a clean null-space, and (c) the value pool being
*shared* across facts (so a per-fact signature has no conflict-free home), we did **not** pursue
injection. This section is a feasibility characterisation of a generic mechanism, not an
implementation.

## 3. Behavioural signal — semantic entropy — AUC 0.66 (Phase 1-bis A2+)

We then tested the strongest published *behavioural* hallucination signal: **semantic entropy**
(Farquhar et al., Nature 2024). For a subsample (120 stored / 120 fake), we draw **k = 8**
temperature-sampled generations (T = 0.7) from a fixed prompt and cluster them by meaning (short
structured / NL answers ⇒ clusters = normalised strings), then measure the entropy over clusters.

| metric | AUC | stored | fake |
|---|---|---|---|
| **semantic entropy** | **0.6625** | H = 0.00 | H = 0.27 |
| # meaning-clusters | 0.6625 | 1.00 | 1.63 |
| spread (1 − max-cluster fraction) | 0.6625 | 0.00 | 0.12 |

Notably this is **not** better than the cheap proxies — a 3-phrasing agreement gave 0.69 and
first-token decode confidence gave 0.67. Concrete behaviour:

- stored entity (`→ Ghent`): 8/8 samples `Ghent` (entropy 0);
- fake entity (`Nairobi`, not stored): 8/8 samples `New York` — a **single, confident, stable**
  fabrication, indistinguishable from a stored fact by entropy;
- only a minority of fakes waver (e.g. one fake → {New York, Riga, Naples}).

**63 % of fake entities yield a single cluster over k = 8.** The model does not "know that it does not
know": for most unknown entities it fabricates one confident value and repeats it under sampling, so
there is little epistemic variance for semantic entropy to detect. (Hidden-state variants —
EigenScore / INSIDE, Chen et al. 2024; semantic-entropy probes, Kossen et al. 2024; SelfCheckGPT,
Manakul et al. 2023 — read the same generations/states and are not expected to recover a signal that
is absent from both the geometry (§1–2) and the sampling behaviour (§3).)

## 4. Mechanism

The retrieved value for a non-stored entity is **confident, coherent, well-formed, and stable across
sampling** — a single fabricated value, not a hedge and not a blend. Consequently:

- routing/addressing carries no stored-vs-fake bit (qk-normalisation removes magnitude; the angle is
  unchanged), §1;
- the assembled value carries none either (coherence identical for stored and fake), §2;
- the output distribution carries only a weak one (majority of fakes are self-consistent), §3.

The "is-this-entity-stored" information is not present, geometrically or behaviourally, in a form an
internal detector can read.

## 5. Structural conclusion

On a **frozen-backbone product-key parametric memory**, entity-level open-set recognition is
**empirically intractable by internal or behavioural measurement**: three independent signal families
(routing geometry, value-space geometry with a normalisation-invariant estimator, and semantic
entropy) all fail to reach a usable operating point (AUC 0.50 / 0.50 / 0.66), each under a
stored-recall = 1.000 sanity gate. The gate remains a sound *domain-level* mechanism; **entity-level**
abstention is not obtainable this way.

## 6. Operational recommendation

Reliable entity-level abstention requires an **external** check: before trusting a memory-produced
value, verify that the queried entity exists in the known-fact index (a retrieval / lookup step). This
is deterministic and exact, at the cost of the per-query retrieval already characterised for the RAG
baseline in `BASELINES.md` (v0.3.2) — the same parametric-vs-retrieval trade-off, now with a concrete
reason it cannot be closed internally on this architecture. A hybrid (parametric memory for recall,
a cheap membership lookup for abstention) is the natural next design.

## Reproduce

- §1 `python r_safety_probe.py` — routing features × layers, AUC, sanity gate.
- §2 `python r_safety_1bisA.py` — Tyler-Mahalanobis / coherence, MP spectrum, sanity gate.
- §3 `python r_safety_a2plus.py` — semantic entropy (k=8, T=0.7), sanity gate.

All in-process; synthetic data; single 24 GB consumer GPU (ROCm/WSL2).

## References

- Berges et al. 2024, *Memory Layers at Scale*, arXiv:2412.09764.
- Lample et al. 2019, *Large Memory Layers with Product Keys*, arXiv:1907.05242.
- Marchenko & Pastur 1967, *Distribution of eigenvalues for some sets of random matrices*, Math. USSR-Sbornik.
- Baik, Ben Arous & Péché 2005, *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices*, Annals of Probability.
- Tyler 1987, *A distribution-free M-estimator of multivariate scatter*, Annals of Statistics.
- Cox, Kilian, Leighton & Shamoon 1997, *Secure spread spectrum watermarking for multimedia*, IEEE Trans. Image Processing.
- Farquhar, Kossen, Kuhn & Gal 2024, *Detecting hallucinations in large language models using semantic entropy*, Nature 630.
- Chen et al. 2024, *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection* (EigenScore), ICLR 2024, arXiv:2402.03744.
- Kossen et al. 2024, *Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs*, arXiv:2406.15927.
- Manakul, Liusie & Gales 2023, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*, EMNLP 2023, arXiv:2303.08896.
