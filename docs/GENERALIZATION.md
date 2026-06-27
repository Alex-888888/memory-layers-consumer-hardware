# Generalisation of the relevance gate (v0.3.1)

Where does the learned relevance gate's generalisation *actually* extend? v0.2.2 showed it
opens on unseen **phrasings**. v0.3.1 stress-tests it on unseen **entities**, unseen **families**,
and **non-stored** entities, and characterises the mechanism behind its safety. **All data is
synthetic.** Headline: the gate is a **domain / distribution-level** relevance detector — not an
entity-level oracle — and entity-level safety emerges, for free, from the memory's retrieval
geometry.

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

Is this a fundamental limit or a coverage gap? **Coverage** — proven by §1: when the gate has
seen *some* bio entities, it generalises perfectly to held-out bio entities (Δ 0). So the rule is
**"the gate needs at least one example per distributional cluster"**, not "the gate cannot do
natural language". We frame this as a **boundary characterisation**, not a universal solution:
one cannot enumerate every cluster in advance.

## 3. Why opening on non-stored entities is safe (negative control + retrieval geometry)

The gate opens on **fake** entities of a stored structure just as much as on real ones
(open-rate 0.93 vs 0.93) — yet **recall on fake entities is 0 %** (no hallucination). We
instrumented *why*, logging, per gated layer, for stored-seen / stored-held-out / fake:

| signal | stored-seen | stored-held-out | **fake** |
|---|---|---|---|
| decode confidence (answer token) | 1.00 | 1.00 | **0.66** |
| memory output norm, layer 22 | 91.9 | 92.8 | **69.8** |
| memory output norm, layers 6/14 | ~32 | ~32 | ~32 (no separation) |
| product-key top-1 weight / entropy | ~1.0 / ~0 | ~1.0 / ~0 | ~1.0 / ~0 (no separation) |

Mechanism: the product-key **addressing is always confident** (top-1 ≈ 1, entropy ≈ 0) and the
early-layer memory-output norm is **identical** for stored and fake — so neither the gate (which
reads the hidden state) nor the addressing can tell "is this entity stored?". For a fake entity,
the retrieval returns a **confident but incoherent** mixture of unrelated values; this incoherence
surfaces downstream as a **collapsed decode confidence** (1.00 → 0.66) and a reduced layer-22
output norm — so no coherent answer token is produced. **Safety is emergent**: the gate handles
*domain* relevance; entity-level filtering falls out of the memory's retrieval geometry, not from
the gate. (Consequence: feeding product-key confidence into the gate would not add entity-level
discrimination — the separating signal is downstream of addressing, not in it.)

## 4. The gate closes on general text (the perplexity benefit)

Open-rate on negatives: **general prose 0.0005** (the gate closes → general language modelling is
protected), general factual questions 0.18 (the residual that explains the small TriviaQA gap).

## Framing

We call this **domain-relevance gating**, not "relevance gating": the gate detects the
*distributional signature of a stored-fact context*, generalising across entities and across
families within a learned distributional cluster, while requiring at least one example per
cluster. Entity-level safety is **not** a gate property — it emerges from the memory's retrieval
geometry (incoherent retrieval → low-confidence output for non-stored entities).

**Contributions, restated honestly.** (1) Gate-only training (backbone + memory frozen) recovers
nearly all the always-on-memory tax. (2) We map the generalisation frontier: entities ✓, families
within a cluster ✓, across clusters ✗ (coverage-bound). (3) Division of labour: gate = domain
relevance, memory = emergent entity filtering, no hallucination.

## Limitations (Threats-to-validity preview)

- The NL-family held-out failure (Δ −93) is real and **not hidden**: it is the frontier. Each new
  distributional cluster needs gate-training coverage.
- The structural false-open is benign here (no hallucination) but is an architectural property of
  *this* memory; a different memory whose non-stored retrieval is coherent could behave differently.
- All families are synthetic; the natural-language family is a single, simple paradigm.
- Reproduce: `python train_relevance_gate.py` (v0.3.0) for the base pipeline; the generalisation
  battery is the C3 protocol described above.
