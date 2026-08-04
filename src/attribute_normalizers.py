# -*- coding: utf-8 -*-
"""Format-aware attribute normalizers — deterministic R1/R2 verification (embedding-free).

Three character-level normalizers that decide whether two attribute values are the SAME by
canonical form, without any embedding or model. A companion to a parametric memory: where the
membership index (docs/EXTERNAL_VERIFICATION.md) answers "is this entity stored?", these answer
"is this claimed value the stored value?" for the deterministic cases (R1 exact, R2 format).

FRONTIER — read this first. The limits, made executable by the dry-run below:

  1. DETERMINISTIC R1/R2 ONLY, NOT SEMANTIC. Each normalizer canonicalises a known FORM
     (version vs IP, a physical unit by dimension, an exact acronym expansion) and compares the
     canonical strings. A value that is semantically equivalent but not format-reducible is OUT
     OF SCOPE — reported as a false-refusal, never guessed.
  2. WHY NOT SEMANTIC (measured). A semantic-embedding validator was tested on a labelled bench
     and is worse than nothing on technical attributes: AUC 0.32 (inverted) — an embedder rates a
     WRONG-value neighbour (e.g. "v4.1.0" vs "v4.1.1", cosine ~0.93) as MORE similar than a true
     paraphrase. The last digit — exactly the bit that separates a right value from a wrong one —
     is what a meaning-encoder discards. So attribute verification stays exact/character-level.
  3. CLOSED, AUDITABLE TABLE. The acronym table is an explicit closed list (the one below is a
     generic example). An acronym used as a CONCEPT (a gloss/paraphrase) is excluded by design —
     the table holds sigle<->exact-expansion only, never an interpretation.
  4. STATUS: validated on this bench (0 % false-accept on the trap pairs), NOT YET INTEGRATED in
     production. There is no attribute-verification step in the running assistant yet. This is a
     characterised mechanism, not a deployed faithfulness check.

Discipline (kept from the build): in each normalizer the TEST on the trap pairs is written BEFORE
the rule — the rule is born constrained by its guard-rail. Criterion: false-accept <= 1 %
(never match two genuinely different values); coverage reported as a result (false-refusal owned).

The data below is a GENERIC EXAMPLE (public sigles, fabricated trap pairs). Run:
  python attribute_normalizers.py
"""
import re

# =====================================================================================
#  M3 — VERSIONS / IP / DATES  (the risk: X.Y.Z is shared by versions and IP addresses)
# =====================================================================================
# ---- TEST (written BEFORE the rule): trap pairs that MUST stay DISTINCT ----
M3_MUST_DIFFER = [
    ("192.168.0.1", "1.9.2.168"),   # IP (4 octets) vs a version-shaped string
    ("10.20.30.40", "10.20.30"),    # IP 4-part vs version 3-part (length / namespace)
    ("172.16.0.1", "0.1.0"),        # IP vs version
]
M3_MUST_MATCH = [                    # coverage target (the easy case)
    ("v1.2.3", "1.2.3"), ("version 1.2.3", "1.2.3"), ("v10.0.0", "10.0.0"),
    ("2026-02-11", "2026-02-11"),
]
# ---- RULE (structural typing = guard-rail: IP and version in disjoint namespaces) ----
def m3_canon(tok):
    t = tok.strip().lower().replace("version", "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return "date:" + t
    m4 = re.fullmatch(r"(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})", t)
    if m4 and all(int(o) <= 255 for o in m4.groups()):
        return "ip:" + t                                   # IP -> namespace 'ip'
    m = re.fullmatch(r"v?(\d+)\.(\d+)(?:\.(\d+))?", t)
    if m:
        return "ver:" + ".".join(p for p in m.groups() if p is not None)  # version -> 'ver'
    return None                                            # abstain (false-refusal)
def m3_match(a, b):
    ca, cb = m3_canon(a), m3_canon(b)
    return ca is not None and ca == cb

# =====================================================================================
#  M1 — UNITS  (typed by DIMENSION: never relate two units by a shared prefix letter)
# =====================================================================================
# ---- TEST (before the rule): trap pairs that MUST stay DISTINCT ----
M1_MUST_DIFFER = [
    ((100, "MHz"), (100, "MB")),    # frequency vs data-size (shared 'M' prefix)
    ((100, "MHz"), (100, "Mbps")),  # frequency vs data-rate
    ((10, "dB"), (10, "dBm")),      # ratio vs absolute power (shared 'dB' root)
    ((10, "dBm"), (10, "dBc")),     # absolute power vs carrier-relative
]
M1_MUST_MATCH = [                    # coverage target (exact intra-dimension scaling)
    ((1, "GHz"), (1000, "MHz")), ((1, "GB"), (1000, "MB")), ((1000, "kHz"), (1, "MHz")),
]
# ---- RULE: each unit carries its DIMENSION; match iff same dimension AND same base value ----
DIM = {
    "hz": ("freq", 1), "khz": ("freq", 1e3), "mhz": ("freq", 1e6), "ghz": ("freq", 1e9),
    "b": ("size", 1), "kb": ("size", 1e3), "mb": ("size", 1e6), "gb": ("size", 1e9), "tb": ("size", 1e12),
    "bps": ("drate", 1), "kbps": ("drate", 1e3), "mbps": ("drate", 1e6), "gbps": ("drate", 1e9),
    "db": ("ratio", 1), "dbm": ("pwr_abs", 1), "dbc": ("carrier_rel", 1),
    "v": ("volt", 1), "mv": ("volt", 1e-3), "kv": ("volt", 1e3),
    "s": ("time", 1), "ms": ("time", 1e-3), "us": ("time", 1e-6), "ns": ("time", 1e-9),
    "m": ("len", 1), "mm": ("len", 1e-3), "cm": ("len", 1e-2), "km": ("len", 1e3),
}
def m1_canon(value, unit):
    d = DIM.get(unit.strip().lower())
    if d is None:
        return None                                        # unknown unit -> abstain
    dim, scale = d
    base = float(value) * scale if dim not in ("ratio", "pwr_abs", "carrier_rel") else float(value)
    return (dim, round(base, 6))                           # key = (DIMENSION, base value)
def m1_match(a, b):
    ca, cb = m1_canon(*a), m1_canon(*b)
    return ca is not None and ca == cb

# =====================================================================================
#  M4 — ACRONYMS  (exact, closed, bilingual table; no fuzzy; gloss excluded)
# =====================================================================================
# ---- TEST (before the table): short codes 1 letter apart stay DISTINCT (no fuzzy);
#      an acronym used as a CONCEPT (gloss) must NOT match its expansion ----
M4_MUST_DIFFER_CODES = [("ABC", "ABD"), ("ABD", "ABE"), ("XYZ", "XYW")]
M4_GLOSSES_MUST_NOT_MATCH = [                              # sigle used as a CONCEPT = out of table
    ("CPU", "the brain of the computer"),
    ("URL", "a web address"),
]
M4_MUST_MATCH = [                                          # exact expansions, bilingual
    ("CPU", "central processing unit"), ("OS", "operating system"),
    ("OS", "systeme d'exploitation"), ("URL", "uniform resource locator"),
]
# ---- TABLE : sigle -> set of EXACT expansions (never a gloss). GENERIC PUBLIC EXAMPLE. ----
def _n(s): return re.sub(r"\s+", " ", s.strip().lower())
ACR_TABLE = {
    "CPU": {_n("central processing unit")},
    "RAM": {_n("random access memory")},
    "URL": {_n("uniform resource locator")},
    "OS":  {_n("operating system"), _n("systeme d'exploitation")},   # bilingual EN/FR
    "HTTP": {_n("hypertext transfer protocol")},
}
def m4_match_code(a, b):                                   # code<->code : exact only, never fuzzy
    return a == b
def m4_match_expansion(acr, phrase):                       # sigle<->expansion : exact table
    return _n(phrase) in ACR_TABLE.get(acr, set())

# =====================================================================================
#  DRY-RUN — the tests, on the generic example data
# =====================================================================================
def run_dry():
    print("=== attribute normalizers — dry-run (tests before rules, generic example) ===")
    fa = 0                                                 # false-accept: two different values matched
    fa_m3 = sum(1 for a, b in M3_MUST_DIFFER if m3_match(a, b))
    cov_m3 = sum(1 for a, b in M3_MUST_MATCH if m3_match(a, b))
    print("  M3  false-accept on traps: %d/%d  |  target coverage: %d/%d"
          % (fa_m3, len(M3_MUST_DIFFER), cov_m3, len(M3_MUST_MATCH)))
    fa_m1 = sum(1 for a, b in M1_MUST_DIFFER if m1_match(a, b))
    cov_m1 = sum(1 for a, b in M1_MUST_MATCH if m1_match(a, b))
    print("  M1  false-accept on traps: %d/%d  |  target coverage: %d/%d"
          % (fa_m1, len(M1_MUST_DIFFER), cov_m1, len(M1_MUST_MATCH)))
    fa_codes = sum(1 for a, b in M4_MUST_DIFFER_CODES if m4_match_code(a, b))
    fa_gloss = sum(1 for a, g in M4_GLOSSES_MUST_NOT_MATCH if m4_match_expansion(a, g))
    cov_m4 = sum(1 for a, e in M4_MUST_MATCH if m4_match_expansion(a, e))
    print("  M4  false-accept short codes: %d/%d  |  gloss leaks: %d/%d  |  expansions ok: %d/%d"
          % (fa_codes, len(M4_MUST_DIFFER_CODES), fa_gloss, len(M4_GLOSSES_MUST_NOT_MATCH),
             cov_m4, len(M4_MUST_MATCH)))
    fa = fa_m3 + fa_m1 + fa_codes + fa_gloss
    print("  -> criterion false-accept <= 1 %%: %s (%d false-accept over %d trap pairs)"
          % ("PASS" if fa == 0 else "FAIL", fa,
             len(M3_MUST_DIFFER) + len(M1_MUST_DIFFER) + len(M4_MUST_DIFFER_CODES) + len(M4_GLOSSES_MUST_NOT_MATCH)))


if __name__ == "__main__":
    run_dry()
