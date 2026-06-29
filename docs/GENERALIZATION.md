# Generalisation of the relevance gate (v0.3.1, §3 corrected in v0.3.4)

Where does the learned relevance gate's generalisation *actually* extend? v0.2.2 showed it
opens on unseen **phrasings**. v0.3.1 stress-tests it on unseen **entities**, unseen **families**,
and **non-stored** entities. **All data is synthetic.** Headline: the gate is a **domain /
distribution-level** relevance detector — not an entity-level oracle.

> **v0.3.4 correction.** §3 of the original v0.3.1 (below) claimed *emergent entity-level safety*
> via a *decode-confidence collapse 1.00 → 0.66* and *"no hallucination"*. That rested on **40**
> fake entities. Re-measured at **n = 360** (with a stored-recall = 1.000 sanity gate), the effect
> is **partial** (AUC 0.69, fake mean 0.91) and the model in fact **confidently fabricates
> plausible wrong values** for non-stored entities — it does *not* abstain. The accurate property
> is **no inter-fact leakage**, not "no hallucination". The corrected analysis is in
> [`SAFETY_EVAL.md`](SAFETY_EVAL.md); §3 below is annotated accordingly.

## Setup

A frozen Qwen2.5-7B + a product-key memory (MLP-ADD at layers 6/14/22) is trained on **six fact
families**: five structured (sensor hex, config integer, service semver, node coordinates,
protocol hex-byte) and one **natural-language** family (biographical: *"Where was {Person}
born? → {City}"*), 60 entities each. The memory is then **frozen**, and the gate is re-trained
under several regimes (gate training is cheap; the memory is built once).

## 1. Held-out entities — the gate generalises (drop 0)

Gate trained on 80 % of the entities of every family; evaluated on the **held-out 20 %**
(stored in memory, never seen by the gate). Recall, gated vs ungated:

| family | ungated | gated | Δ |
|---|---|---|---|
| sensor_calib | 100 % | 100 % | 0 |
| config_param | 100 % | 100 % | 0 |
| service_ver | 100 % | 100 % | 0 |
| proto_status | 100 % | 100 % | 0 |
| **bio (NL)** | 100 % | **100 %** | **0** |
| node_coord | 16.7 % | 16.7 % | 0 |

The gate **does not hurt recall on entities it never saw** — for every family, including the
natural-language one. (node_coord's low *ungated* recall is a memory limitation — exact decimal
coordinates are hard to memorise at this budget — not a gate effect; gated = ungated.)

This answers the central reviewer concern ("held-out tests only cover phrasings"): the gate
generalises to **unseen entities**, not just unseen phrasings.

## 2. Held-out families — generalisation is distribution-level (leave-one-family-out)

Gate trained on five families, tested on the held-out sixth, for **all six** in rotation:

| held-out family | ungated | gated | Δ |
|---|---|---|---|
| config_param | 100 % | 100 % | 0 |
| service_ver | 100 % | 100 % | 0 |
| proto_status | 100 % | 95 % | −5 |
| sensor_calib | 100 % | 95 % | −5 |
| node_coord | 27.5 % | 27.5 % | 0 |
| **bio (NL)** | 100 % | **7.5 %** | **−93** |

The five **structured** families form a single distributional cluster: each transfers to the
others (Δ ≤ 5). The **NL family is an outlier** — held out, the gate closes on it (Δ −93).

Is this a fundamental limit or a coverage gap? The evidence points to **coverage**: §1 shows that
when the gate has seen *some* bio entities, it generalises to held-out bio **entities** (Δ 0). So
the supported statement is **"held-out NL *entities* recover once the NL family is seen"**.

> **v0.3.4 scope caveat.** There is only **one** NL family here, so we cannot distinguish
> *per-cluster* coverage ("one NL example generalises to a *different* NL family") from
> *per-family* coverage ("each family needs its own examples"). The honest claim is the narrower
> one — at least one example **per family**, and plausibly per distributional cluster — pending a
> second, structurally different NL family. We report this as a **boundary characterisation**, not
> a universal solution.

## 3. Behaviour on non-stored entities — *corrected in v0.3.4 (see [`SAFETY_EVAL.md`](SAFETY_EVAL.md))*

The gate opens on **fake** entities of a stored structure just as much as on real ones
(open-rate 0.93 vs 0.93): the gate is **domain-level** and does not filter entities. The original
v0.3.1 inferred, from a 40-fake sample, an *emergent entity-level safety* via a decode-confidence
collapse to 0.66. **The larger n = 360 measurement does not support that:**

| signal (n=360) | stored | fake |
|---|---|---|
| decode confidence (mean) | **1.00** | **0.91** (left tail to 0.15; ~½ of fakes < 0.93) |
| AUC stored-vs-fake (decode_conf) | — | **0.686** (partial, not categorical) |
| reproduces the entity's assigned value (recall) | n/a | **≈ 0 %** (no inter-fact leakage) |
| produces a well-formed value at all | 100 % | **100 % (confident fabrication)** |

What remains true from the original analysis: the product-key **addressing is always confident**
(top-1 ≈ 1, entropy ≈ 0) and early-layer memory-norm does **not** separate stored from fake — so
neither the gate nor the addressing can tell "is this entity stored?". What is **corrected**: this
does **not** yield clean entity-level safety. On a non-stored entity the model **confidently
fabricates a plausible, format-correct, wrong value** (e.g. a fake config param → `614`, a fake
protocol byte → `0x76`); it does **not** abstain, and decode-confidence flags only ~half of fakes
(AUC 0.69). The one robust property is **no inter-fact leakage**: the model does not reproduce the
fake's assigned value, nor pull a stored fact for an unknown entity (`fake_eq ≈ 0`).

## 4. The gate closes on general text (the perplexity benefit)

Open-rate on negatives: **general prose 0.0005** (the gate closes → general language modelling is
protected), general factual questions 0.18 (the residual that explains the small TriviaQA gap).

## Framing

We call this **domain-relevance gating**: the gate detects the *distributional signature of a
stored-fact context*, generalising across entities and across families within a learned
distributional cluster, while requiring at least one example per family/cluster. **Entity-level
filtering is *not* provided** — neither by the gate (domain-level) nor by the memory (which
fabricates plausible values for unknown entities). The only entity-level guarantee observed is
**no inter-fact leakage**.

**Contributions, restated honestly.** (1) Gate-only training (backbone + memory frozen) recovers
nearly all the always-on-memory tax. (2) We map the generalisation frontier: entities ✓, families
within a cluster ✓, across clusters/NL ✗ (coverage-bound). (3) Division of labour: gate = domain
relevance; the memory does **not** filter unknown entities (it fabricates) — the only robust
safety property is no inter-fact leakage; decode-confidence is a **partial** non-stored-ness signal
(AUC 0.69, [`SAFETY_EVAL.md`](SAFETY_EVAL.md)).

## Limitations (Threats-to-validity preview)

- The NL-family held-out failure (Δ −93) is real and **not hidden**: it is the frontier. Each new
  family (and plausibly each distributional cluster) needs gate-training coverage; a single NL
  family cannot settle per-cluster vs per-family.
- **No emergent entity-level safety.** On non-stored entities the model confidently fabricates
  plausible values; only "no inter-fact leakage" holds (see [`SAFETY_EVAL.md`](SAFETY_EVAL.md)).
- All families are synthetic; the natural-language family is a single, simple paradigm.
- Reproduce: `python train_relevance_gate.py` (v0.3.0) for the base pipeline; the held-out
  battery is the C3 protocol above; the corrected non-stored analysis is `python a2_inproc.py`.
