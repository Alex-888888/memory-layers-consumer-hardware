# Open-set recognition on a frozen parametric memory — a negative result (v0.4.3)

**Question.** The relevance gate is *domain-level* (`GENERALIZATION.md`, `SAFETY_EVAL.md`): it opens on a
stored-fact context but does **not** decide whether a *specific queried entity* is actually stored.
On a non-stored entity the model confidently fabricates a plausible, format-correct wrong value
(`SAFETY_EVAL.md`). Can we add a reliable **out-of-support / open-set recognition (OSR)** signal —
"is this entity in memory?" — computed *internally* (from the model's own states / behaviour), so the
system can abstain instead of fabricating?

**Answer (this sprint).** No, not from internal or behavioural measurement, not from a
supervised probe (including a capacity-augmented LoRA-probe), and — added in v0.4.3 — not from an
**upstream pre-normalization density/energy filter** either. Across **five independent signal
families** we find OSR AUC at or near chance for the geometric signals, only weakly above chance for
the strongest behavioural one, only 0.685 (linear) / 0.622 (LoRA-probe) for a supervised probe trained
directly to separate stored from fabricated entities on an entity-disjoint split, and — the v0.4.3
result — **0.52** for a density filter measuring the query *before* qk-normalization, in the raw
Euclidean space where magnitude still exists. All below any usable operating point, and the supervised
probe far below the 0.867 / 0.905 the same methodology attains on general LLMs. The mechanism is
identified: the memory produces **confident, deterministic, self-consistent fabrications**, so there is
simply no accessible uncertainty to measure and nothing for a probe to latch onto that generalises
across entities — and the "is-stored" bit is not merely destroyed by qk-normalization, it is **absent
from the query geometry upstream of it**. We conclude that reliable entity-level abstention on this
frozen architecture requires an **external** check, and we recommend a retrieval-based verification.

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

## 4. Supervised linear probe on the residual stream — AUC 0.685 (Phase 1-bis-E)

The three families above are *unsupervised* readouts (geometry, behaviour). The natural next question
is whether a **supervised** probe — trained with labels "stored vs fabricated", the strongest form of
internal detector in the hallucination-probing literature (Azaria & Mitchell 2023; Zou et al. 2023;
Marks & Tegmark 2023; Obeso et al. 2025) — can find a decision boundary the unsupervised readouts
miss. We train a logistic probe on the full **residual-stream** hidden state (d = 3584) at the
answer-position token, sweeping five downstream layers **[14, 22, 24, 26, 28]**.

**Entity-disjoint split (the essential control).** A probe trained and tested on the *same* entities
can learn to recognise individual entity identities rather than the stored-vs-fabricated property,
inflating the AUC by memorisation. We therefore split by **entity**, not by example: **252 entities
per class train / 108 per class held-out**, with no entity appearing in both. The reported AUC is on
held-out entities the probe never saw.

| probe layer | test AUC (held-out entities) | train AUC |
|---|---|---|
| L14 | 0.575 | 0.799 |
| L22 | 0.615 | 0.929 |
| L24 | 0.654 | 0.910 |
| **L26 (best)** | **0.685** | 0.952 |
| L28 | 0.650 | 0.961 |
| (reference) decode confidence | 0.696 | — |

Even the best layer reaches only **0.685** on held-out entities — *below* the free decode-confidence
baseline (0.696). The large train/test gap (0.952 → 0.685 at L26) is the signature of exactly the
memorisation the disjoint split is designed to expose: the probe fits *which entities* it saw, and
that does not transfer. There is no entity-general "stored-vs-fabricated" direction in the residual
stream to be read.

## 5. LoRA-probe (Obeso 2025 headline method) — AUC 0.622 (Phase 1-bis-E-LoRA)

The strongest single method in Obeso et al. (2025) is not a static probe but a **LoRA-probe**: a
low-rank adapter is trained jointly with the probe head, so the network can *reshape* its own
activations to make the target property linearly decodable, while a KL term keeps the underlying
behaviour intact. On general LLMs this lifts AUC from **0.867** (their linear probe) to **0.905**. We
reproduced it faithfully on our architecture:

- **Adapter:** `LoraConfig(r=16, α=32, target=self_attn.{q_proj, v_proj})` → **5,046,272 trainable
  params (0.065 % of 7.74 B)**; backbone weights and the memory pool remain frozen.
- **Joint objective:** `BCE(probe(LN(h_L26)), y) + λ_KL · KL(logits_LoRA ‖ logits_frozen)` with
  `λ_KL = 0.5`, the frozen reference obtained via `disable_adapter()`.
- **Behaviour preserved:** final training **KL = 0.0000**; **sanity stored-recall = 1.000 with the
  LoRA active** (60/60), identical to the no-LoRA gate (1.000) — the adapter did not damage the
  memory, it simply had nothing more to expose.
- **Same entity-disjoint split** as §4 (504 train / 216 held-out).

**Result: held-out test AUC = 0.622** — *lower* than the plain linear probe (0.685) and the
decode-confidence baseline (0.696).

## 6. The decisive comparison

| method | this architecture (product-key, frozen backbone, qk-norm) | Obeso 2025 (general LLM) |
|---|---|---|
| linear probe | **0.685** | 0.867 |
| **LoRA-probe** | **0.622** | **0.905** |

Two things stand out. First, **adding capacity finds no additional signal**: the LoRA-probe (0.622)
is *below* the linear probe (0.685) and below decode confidence (0.696). When a method that is
*designed* to manufacture a decodable direction — and demonstrably can on general models — fails to
beat a free baseline here, the property it is looking for is not weakly encoded but **absent**.
Second, the **0.28-point gap on the LoRA-probe** (0.905 vs 0.622) between a general LLM and this
architecture shows the intractability is **specific to product-key memory with a frozen backbone and
qk-normalisation**, not a general limitation of activation probing. The method works elsewhere; it
does not work here, and the architecture is why.

## 7. Mechanism (unified)

The retrieved value for a non-stored entity is **confident, coherent, well-formed, deterministic, and
stable across sampling** — a single fabricated value, not a hedge and not a blend. The §3 measurement
makes this concrete: **63 % of unknown entities yield a single value cluster across 8 samples**. This
one property explains every negative above:

- routing/addressing carries no stored-vs-fake bit (qk-normalisation removes magnitude; the angle is
  unchanged), §1;
- the assembled value carries none either (coherence identical for stored and fake), §2;
- the output distribution carries only a weak one (majority of fakes are self-consistent), §3;
- **the residual stream carries no entity-general one** — a supervised linear probe overfits entity
  identity and reaches only 0.685 on held-out entities, §4;
- **and no amount of adapter capacity manufactures one** — a LoRA-probe that preserves behaviour
  (KL 0, recall 1.000) still lands at 0.622, §5.

A deterministic, confident fabrication produces **no measurable internal signal, regardless of probe
capacity**. There is no uncertainty because the model is not uncertain — it is confidently wrong in a
stable way.

## 8. Structural conclusion

On a **frozen-backbone product-key parametric memory with qk-normalisation**, entity-level open-set
recognition is **empirically intractable — by internal measurement, by behavioural measurement, by
supervised probing including capacity-augmented LoRA-probes, and by upstream pre-normalization density
filtering**. Five independent signal families (routing geometry, value-space geometry with a
normalisation-invariant estimator, semantic entropy, supervised residual probes linear + LoRA, and a
pre-qk-norm Euclidean density/energy filter) all fail to reach a usable operating point
(AUC 0.50 / 0.50 / 0.66 / 0.685–0.622 / 0.52), each under a stored-recall = 1.000 sanity gate. The gate
remains a sound *domain-level* mechanism; **entity-level** abstention is not obtainable this way, at
any stage of the memory read.

## 9. Operational recommendation

Reliable entity-level abstention requires an **external** check: before trusting a memory-produced
value, verify that the queried entity exists in the known-fact index (a retrieval / lookup step). This
is deterministic and exact, at the cost of the per-query retrieval already characterised for the RAG
baseline in `BASELINES.md` (v0.3.2) — the same parametric-vs-retrieval trade-off, now with a concrete
reason it cannot be closed internally on this architecture. A hybrid (parametric memory for recall,
a cheap membership lookup for abstention) is the natural next design.

## 10. Methodological note — entity-disjoint splitting

The supervised-probe results (§4–5) are only meaningful **because** train and test entities are
disjoint. A same-entity split would let the probe memorise entity identities and report a
misleadingly high AUC (here the train AUC reaches 0.95 while held-out is 0.685). We record the
**entity-disjoint split as an essential control for any supervised open-set / hallucination probe on
parametric memory**: split by the underlying stored unit, not by phrasing or example, or the metric
measures memorisation rather than generalisation.

## 11. v0.4.3 addendum — upstream pre-normalization density (the fifth family) — AUC 0.52

Families §1–§5 all read the query *after* qk-normalization, or read downstream states. One channel is
left: qk-normalization (the `F.normalize(q) @ F.normalize(C)` router) discards the **magnitude** of the
query and of the product-key codebook before the dot product. The direction is known not to separate
(§1, §3). So the only untested signal is the **raw Euclidean geometry upstream of the normalization** —
the natural home of a density / energy-based OOD filter (Ren et al. 2019; Nalisnick et al. 2019; LeCun
et al. EBM). Hypothesis: a stored query sits closer, in raw Euclidean terms, to the codebook key
manifold than an out-of-support (fake) query.

We capture the pre-normalization query halves `q1, q2` at the answer-position token for 360 stored vs
360 fake (disjoint seed), at each gated layer (6/14/22), and measure three features against the raw
codebook rows `C1/C2`:

| feature (best over 3 layers × 2 halves) | AUC stored-vs-fake |
|---|---|
| F1 — query magnitude ‖q‖ | 0.506 – 0.514 |
| **F2 — nearest-codebook-key Euclidean distance (raw)** — the one axis qk-norm discards | **0.507 – 0.514** |
| F2n — same distance, normalised (control, must ≈ 0.50) | 0.501 – 0.511 ✓ |
| F3 — reconstruction residual on the codebook PCA span (r = 8…64) | 0.501 – 0.519 |
| **best over all features / layers** | **0.5185** |

Stored and fake are indistinguishable to the 3rd–4th significant figure (e.g. L6.q1 magnitude
202.05 vs 201.96; nearest-key distance 200.46 vs 200.37). The normalised control F2n reproduces ~0.50
as expected, confirming the framing measures the right object. The codebook itself has **no exploitable
low-dimensional structure**: 0 eigenvalues above the Marchenko–Pastur edge, participation ratio
≈ 165–194 out of ≈ 224 available directions (near-isotropic) — nothing for a linear autoencoder to
reconstruct against.

**Interpretation — the finding extends upstream.** The "is-stored" bit is not merely *destroyed* by
qk-normalization; it **was never in the query geometry**. Two structural reasons: (a) the frozen
backbone processes a fake entity (an ordinary real token) identically to a stored one; (b) the query
projection was trained only on stored positives, with no negatives, so it built no rejection geometry.
A pre-normalization density filter is therefore inoperative on this frozen architecture — it is the
fifth family to fail, and it closes the "internal detection" line: there is no stage of the memory read
(addressing, value assembly, output distribution, downstream residual, or upstream query) at which an
entity-level stored-vs-fabricated signal is measurable. This reinforces §9: reliable abstention
requires an external check. (Creating — rather than detecting — such a signal would require leaving the
frozen-backbone setting, e.g. unfreezing the query projection and training against explicit
out-of-support negatives; that is a different architecture and out of scope for this repository, which
characterises the frozen retrofit.)

## Reproduce

- §1 `python r_safety_probe.py` — routing features × layers, AUC, sanity gate.
- §2 `python r_safety_1bisA.py` — Tyler-Mahalanobis / coherence, MP spectrum, sanity gate.
- §3 `python r_safety_a2plus.py` — semantic entropy (k=8, T=0.7), sanity gate.
- §4 `python r_reliable_probe.py` — supervised linear probe on the residual, entity-disjoint split, layer sweep, sanity gate.
- §5 `python r_reliable_lora.py` — LoRA-probe (r=16, KL-preserved), sanity gate with LoRA active.
- §11 `python r_reliable_density.py` — pre-qk-norm Euclidean density (magnitude / nearest-key distance / reconstruction residual), codebook MP spectrum, sanity gate.

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
- Obeso et al. 2025, *Real-time detection of hallucinated entities in long-form generation* (supervised residual probes; linear and LoRA-probe), arXiv:2509.03531.
- Azaria & Mitchell 2023, *The Internal State of an LLM Knows When It's Lying*, EMNLP Findings 2023, arXiv:2304.13734.
- Zou et al. 2023, *Representation Engineering: A Top-Down Approach to AI Transparency*, arXiv:2310.01405.
- Marks & Tegmark 2023, *The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets*, arXiv:2310.06824.
- Ren et al. 2019, *Likelihood Ratios for Out-of-Distribution Detection*, NeurIPS 2019, arXiv:1906.02845.
- Nalisnick et al. 2019, *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019, arXiv:1810.09136.
- LeCun, Chopra, Hadsell, Ranzato & Huang 2006, *A Tutorial on Energy-Based Learning*, in *Predicting Structured Data* (MIT Press).
