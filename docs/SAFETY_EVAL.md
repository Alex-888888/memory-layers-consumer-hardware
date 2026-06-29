# Entity-level behaviour on non-stored entities (v0.3.4)

This note **corrects and replaces** the entity-level safety characterisation from v0.3.1
(`GENERALIZATION.md`). The v0.3.1 claim — that the memory exhibits *emergent entity-level safety*
through a *decode-confidence collapse (1.00 → 0.66)* on non-stored entities — rested on a sample of
**40 fake entities**. Re-measured at **n = 360**, the effect is real but **partial**, and the
"no hallucination" framing is wrong. This is a self-correction; the honest result is sharper than
the original over-claim.

## What was measured, and how (no broken checkpoint)

We took the frozen 6-family product-key memory (the same recipe as `GENERALIZATION.md`), trained
**in-process** (memory ON, gate OFF), and verified it works before drawing any conclusion:

- **Sanity gate: stored recall = 1.000** across all six families (n = 360). Only with this gate
  green is the decode-confidence comparison meaningful. *(An earlier attempt reloaded a saved
  checkpoint whose value pool did not restore faithfully; it produced recall ≈ 0 for stored facts
  too and an AUC ≈ 0.51 — a pure measurement artifact, discarded.)*
- For every **stored** fact (360) and every same-structure **fake** entity (360, generated with a
  different seed, not in memory), we recorded the **decode confidence** = max softmax probability of
  the first answer token.

## Result: a partial confidence signal (AUC 0.69), not a clean separator

| | stored (n=360) | fake (n=360) |
|---|---|---|
| decode_conf mean | **1.000** | **0.906** |
| min / p5 / p25 | 0.999 / 1.000 / 1.000 | 0.152 / 0.488 / 0.930 |
| fraction ≥ 0.95 | 360 / 360 | 264 / 360 |

- **AUC (stored-vs-fake, score = decode_conf) = 0.686.** A real signal, well above chance, but far
  from a categorical separator.
- Stored facts are **uniformly confident** (the whole distribution sits at ≈ 1.0). Fakes have a
  **left tail**: at a near-1.0 threshold (Youden-J ≈ 0.9999), **95 % of stored** pass while **~52 %
  of fakes** also pass — i.e. only about **half** of non-stored entities show a confidence drop; the
  other half are decoded just as confidently as stored facts.
- The v0.1 of this claim reported "fake mean 0.66". That was the mean over 40 fakes; at n=360 the
  fake mean is **0.906**. The "1.00 → 0.66 collapse" framing is therefore **retracted**.

## What the model actually does on a non-stored entity: confident, well-formed fabrication

On a fake entity the model does **not** abstain. It emits a **plausible, format-correct, wrong**
value, often with high confidence:

| family | fake entity's (unseen) assigned value | model output |
|---|---|---|
| config_param | 72 | **614** |
| proto_status | 0x16 | **0x76** |
| sensor_calib | B652-0B8D | **94A9-C775** |
| service_ver | v3.19.0 | **v8.27.21** |
| node_coord | 42.48, 138.92 | **-83.94, -113.74** |
| bio | Osaka | **Aarhus** |

So the correct safety statement is **"no inter-fact leakage"**, *not* "no hallucination":

- **No inter-fact leakage / no recall of the unseen value.** The model essentially never reproduces
  the fake entity's assigned value (`fake_eq_assigned ≈ 0`; the single exception was a fake whose
  assigned city happened to equal a *stored* city, 1/12 in one family) — it does not pull a stored
  fact for an unknown entity.
- **But it does hallucinate.** It confidently fabricates a same-format value it never stored. The
  memory provides **no abstention and no entity-level filtering**.

## Conclusion (replaces the v0.3.1 safety framing)

- The gate is **domain-relevance** only: it detects the distributional signature of a stored-fact
  context, and opens on fake same-structure entities just as on real ones (open-rate ≈ real).
- The memory does **not** filter unknown entities; queried about one, the model **fabricates a
  plausible value**.
- Decode confidence carries a **partial** signal of "non-stored-ness" (AUC 0.69; ~half of fakes show
  reduced confidence), insufficient as a standalone safety mechanism.
- The only robust property is **no inter-fact leakage** (the stored set is not mis-recalled for an
  unknown entity).

Reproduce: `python a2_inproc.py` (in-process; prints the stored-recall sanity gate, the
decode_conf distributions, AUC and Youden-J). All data synthetic; backbone Qwen2.5-7B frozen.
