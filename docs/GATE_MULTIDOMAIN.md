# Multi-domain relevance gate (v0.2.2)

In v0.2.1 the relevance gate (which lets the frozen model keep its native competence while
the always-on memory is present) was trained on a **single fact domain** — sensor-style
questions. The open question was: does the gate learn *"this context refers to a stored
entity"*, or does it just memorise the **surface template** of that one domain? If the latter,
the whole approach would be brittle.

v0.2.2 answers it. We train the gate across **five fact families with genuinely different
structures**, and we add a **held-out-phrasing test** that isolates the failure mode.

## Five families, distinct structures

The point is structural diversity, not lexical paraphrase. Each family has a different entity
shape **and** a different value shape:

| family | entity | value example |
|---|---|---|
| sensor calibration | `KX-4823` (2 letters + 4 digits) | `9BE6-A7B9` (hex pair) |
| config parameter | `max_retries` (snake_case) | `17` (integer) |
| service version | `auth-gateway` (kebab-case) | `v3.14.2` (semver) |
| node coordinate | `N7` (node id) | `48.21, -3.55` (signed decimals) |
| protocol status | `ORION-3` (codename) | `0x1F` (hex byte) |

The memory is trained on all five families (frozen backbone, MLP-ADD as before). Recall on the
stored facts is **100 %** across all five structures.

## The held-out-phrasing test (the key safeguard)

For each family we write four question phrasings of the same fact. The memory sees all four
(so it can recall via any phrasing). The **gate is trained on phrasings A/B/C only** and
**evaluated on the unseen phrasing D** of the *same* entities.

- If the gate had memorised the template, it would **close** on phrasing D → the memory would
  not fire → recall would drop.
- If the gate learned *"this entity is stored"*, it should **open** on phrasing D just as on
  A/B/C → recall preserved.

Result:

| metric | trained phrasings (A/B/C) | held-out phrasing (D) |
|---|---|---|
| recall, gated | 100 % | **100 %** |
| recall, ungated | 100 % | 100 % |
| gate open-rate (mean sigmoid) | 0.94 | **0.95** |

**The gated recall on the unseen phrasing equals the ungated recall — a drop of 0 points** —
and the gate opens on the held-out phrasing essentially as much as on the trained ones. The
gate generalises across phrasings: it keys on stored-entity-ness, **not** on the surface
template. The single-domain brittleness concern is resolved.

## General-knowledge preservation holds at multi-domain

With the multi-domain gate, the native-competence recovery is at least as good as the
single-domain gate v6:

| metric | backbone | + memory, ungated | + memory, gated (multi-domain) |
|---|---|---|---|
| TriviaQA (n=1000, 3 gate seeds) | 53.4 % | 42.1 % | **52.8 % ± 0.28** (−0.6 pt) |
| PPL — WikiText-103 | 7.24 | — | 7.32 (**+1.0 %**) |

The gate recovers ~95 % of the ungated TriviaQA loss, with a very small standard deviation
across gate-training seeds (±0.28 pt), and keeps perplexity within +1.0 % of the backbone.

## Takeaway

The frozen-backbone relevance gate is **not** a single-domain artefact. Trained on
heterogeneous fact structures and held out on unseen phrasings, it still opens on stored
facts and closes on general text and general questions — preserving the model's native
competence (TriviaQA −0.6 pt, PPL +1.0 %) while keeping stored-fact recall at 100 %.

*(Method and figures are documented here; gate implementation code remains planned for a later
release. The fact families used for this study are synthetic.)*
