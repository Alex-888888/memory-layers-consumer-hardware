# External membership verification (index)

> **Frontier first.** This mechanism has hard limits, stated *before* any result; the demo
> (`python src/membership_index.py`) makes limit 1 **executable** so you watch it fail by design.
> The PASS numbers are meaningful **only** together with these limits — a perfect number shown
> without its limit is a number that lies by omission.

## What it is

A cheap, auditable, **embedding-free** membership check to run *alongside* a parametric memory: it
decides whether a queried entity is **stored** (recall) or **novel** (abstain), so the system can
abstain instead of confidently fabricating — the failure characterised in `docs/OPENSET_MP_TYLER.md`,
where every *internal* stored-vs-novel detector was at or near chance. It routes a question to a
family by **form** (regex), extracts the identifier, and tests **exact** then **fuzzy** membership
against a phrasing-invariant index of stored identifiers. No neural network, no embedding, no
training.

## Frontier (the limits)

1. **Identifier membership, not attribute truth.** The index checks whether the *identifier* is
   stored. A fake carrying a **real stored identifier with a false attribute** is **accepted as
   stored**. The demo prints this by-design failure (a real name with the wrong birthplace, a real
   sensor id with a valid-format but wrong calibration — both accepted).
2. **Lexical, not semantic.** The separation is a string match on the identifier, not entity
   understanding. On the natural-language family a **name-substitution control** — swap the stored
   name for a random fake name in the same template — collapses the retrieval statistic to the fake
   regime: **AUC 0.526** (≈ chance) for substituted-vs-fake. Any separation came from the *name
   token*, not semantics.
3. **Synthetic only.** Demonstrated on synthetic families. Semantic membership verification on
   **real** entities is out of scope of this public repository.
4. **Assumes a controlled input schema.** The router keys on family form; the natural-language
   pattern (two capitalised words) is deliberately broad and, on arbitrary free text, would
   mis-route any "Word Word" sequence. This is a router for a known question schema, not an open
   named-entity recogniser.

## Mechanism

- **Router** — one auditable regex per family shape (`src/membership_index.py`, `PATTERNS`), tried
  in list order (natural language first). The first matching pattern wins; no match = the router
  abstains.
- **Extractor** — the first hit yields `(family, identifier)`.
- **Membership** — exact set-membership of the normalised identifier; for the NL family, a fuzzy
  fallback (`SequenceMatcher` ratio ≥ τ, default 0.90) catches trivial spelling variants.

## Frozen criterion (pre-registered)

Asymmetric, false-accept-governed: structured mis-route ≤ 0.5 %, internal collision ≤ 0.1 %,
**decisional false-accept on genuinely novel fakes ≤ 1 %**, false-refusal ≤ 15 %. Pool collisions —
a fake whose identifier is, by chance in a finite corpus, already stored — are **not** decisional
false-accepts: they are correct by identity and reported separately.

## Result (synthetic) — read with the frontier above

On the public synthetic corpus (5 structured families from `synthetic_facts` + the in-file NL
family), a stored seed vs a disjoint fake seed:

- router: **0 mis-route / 0 extraction-fail** (stored and fake);
- false-refusal: **0 %** (structured and NL, including the held-out phrasing);
- decisional false-accept on **novel** fakes: **0 %** structured, **0 %** NL;
- the **fuzzy path is exercised on every novel NL miss** and reports its trigger count and the
  closest ratio it saw — so a 0 is visibly "flagged nothing", not "never ran";
- **PASS** — *and* limit 1 demonstrably accepts a real identifier with a false attribute.

## Run it

```
python src/membership_index.py
```

Prints the criterion evaluation **and** the executable frontier case. All data is synthetic; the NL
names are generic invented First+Last combinations (no real people).
