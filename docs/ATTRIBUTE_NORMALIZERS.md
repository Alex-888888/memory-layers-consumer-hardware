# Attribute verification by format-aware normalizers

> **Frontier first.** This mechanism verifies **deterministic** attribute equality (R1 exact, R2
> format), **not** semantic equivalence. The dry-run (`python src/attribute_normalizers.py`) runs the
> guard-rail tests so you watch the criterion hold on the example data. The `false-accept ≤ 1 %`
> number is meaningful **only** together with the limits below.
>
> **Status: validated on a bench, not yet integrated.** This is a characterised mechanism (0 %
> false-accept on the trap pairs). There is **no attribute-verification step wired into the running
> assistant yet** — do not read this as a deployed faithfulness check.

## What it is

Three character-level, **embedding-free** normalizers that decide whether two attribute values are the
**same** by canonical form. A companion to the parametric memory and the membership index
([`EXTERNAL_VERIFICATION.md`](EXTERNAL_VERIFICATION.md)): where the index answers *"is this entity
stored?"*, these answer *"is this claimed value the stored value?"* — for the deterministic cases only.

- **M3 — versions / IP / dates.** The string `X.Y.Z` is shared by software versions and IP addresses;
  a naive comparator would confuse them. The rule types by **structure** (a 4-octet all-≤255 string is
  an IP → `ip:` namespace; otherwise a version → `ver:` namespace; ISO dates → `date:`), so an IP can
  never match a version.
- **M1 — units, typed by dimension.** Each unit carries its **physical dimension** (frequency, data
  size, data rate, power ratio vs absolute vs carrier-relative, …). Two values match iff **same
  dimension and same base value**. Never relate two units by a shared prefix letter (`100 MHz` ≠
  `100 MB`), never collapse a shared root (`dB` ≠ `dBm` ≠ `dBc`). Intra-dimension scaling is exact
  (`1 GHz` = `1000 MHz`).
- **M4 — acronyms, exact closed table.** A closed, auditable `sigle ↔ exact-expansion` table
  (bilingual where relevant). No fuzzy matching — short codes one letter apart stay distinct
  (`ABC` ≠ `ABD`). An acronym used as a **concept** (a gloss/paraphrase) is **excluded by design**;
  the table holds expansions, never interpretations.

## Frontier (the limits)

1. **Deterministic R1/R2 only, not semantic.** Each normalizer canonicalises a known *form* and
   compares canonical strings. A value that is semantically equivalent but not format-reducible is
   **out of scope** — reported as a false-refusal, never guessed.
2. **Why not semantic — measured.** A semantic-embedding validator was tested on a labelled bench and
   is **worse than nothing** on technical attributes: **AUC 0.32** (inverted). An embedder rates a
   *wrong-value* neighbour (`v4.1.0` vs `v4.1.1`, cosine ≈ 0.93) as **more** similar than a true
   paraphrase — the last digit, exactly the bit that separates a right value from a wrong one, is what
   a meaning-encoder discards. So attribute verification stays exact / character-level; the geometry
   cannot separate a right value from a wrong one. (The same *lexical-not-semantic* limit was found for
   the membership index, AUC 0.526.)
3. **Closed, auditable table.** The acronym table is an explicit closed list (the shipped one is a
   **generic public example** — `CPU`, `RAM`, `URL`, `OS`, `HTTP`). Gloss (acronym-as-concept) is
   excluded; the dry-run checks that a gloss does **not** match its expansion.
4. **Not a general parser.** The normalizers assume a value already extracted for a known attribute.
   They canonicalise and compare; they do not locate attributes in free text.

## Frozen criterion (pre-registered)

False-accept-governed: a normalizer must **never** match two genuinely different values (`v4.1.0` ≠
`v4.1.1`, `192.168.0.1` ≠ a version, `100 MHz` ≠ `100 MB`, `dBm` ≠ `dBc`). Target **false-accept
≤ 1 %**; coverage is reported as a result (a non-reducible value is a **false-refusal**, owned, not
hidden). Discipline: in each normalizer the **test on the trap pairs is written before the rule**.

## Result (generic example) — read with the frontier above

On the shipped generic data (fabricated trap pairs, public sigles), the dry-run reports **0
false-accept over the 12 trap pairs** — IP vs version stay distinct, the `dB` family stays distinct,
short codes one letter apart stay distinct, and the gloss cases are rejected — while the intended
equivalences (`v1.2.3` = `version 1.2.3`, `1 GHz` = `1000 MHz`, `OS` ↔ *operating system* / *système
d'exploitation*) match.

## Why exact and not semantic (the design rationale)

Attribute verification is the mirror image of the membership index's lexical limit. A meaning-encoder
is built to **discard** surface detail — it collapses `v4.1.0` and `v4.1.1` because they *are* the same
kind of thing. But the whole point of verifying an attribute is the surface detail: the wrong last
digit. Character-level normalization sees exactly that bit and nothing it shouldn't. The two tools are
complementary: the normalizer distinguishes `4.1.0` from `4.1.1` perfectly (0 % false-accept), where an
embedder confuses them totally (AUC 0.32). An embedder is **never** a substitute for these normalizers
for value verification.

## Run it

```
python src/attribute_normalizers.py
```

Prints the dry-run: the guard-rail tests (written before the rules) on the generic example data. All
sigles are generic and public; all trap pairs are fabricated.
