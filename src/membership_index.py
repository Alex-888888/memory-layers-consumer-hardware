# -*- coding: utf-8 -*-
"""External membership-verification index for stored-fact abstention (pure string-matching).

A cheap, auditable, EMBEDDING-FREE membership check a system can run alongside a parametric memory
to decide whether a queried entity is *stored* (recall) or *novel* (abstain). It routes a question
to a family by FORM (regex), extracts the identifier, and tests EXACT then FUZZY membership against
an index of stored identifiers (which is phrasing-invariant).

FRONTIER — read this first. The mechanism's limits, made executable by the demo below:

  1. It verifies IDENTIFIER MEMBERSHIP, not ATTRIBUTE TRUTH. A fake carrying a REAL stored
     identifier with a FALSE attribute is ACCEPTED as stored. The demo (printed on every run)
     shows this by-design failure so you see it run, not just read it.
  2. The separation is LEXICAL (identifier string match), not SEMANTIC. On the natural-language
     family a name-substitution control gives AUC 0.526 (~ chance) for substituted-vs-fake:
     any separation comes from the name token, not entity semantics. See docs/EXTERNAL_VERIFICATION.md.
  3. Demonstrated on SYNTHETIC data only. Semantic verification on real entities is out of scope of
     this repository.
  4. Assumes a CONTROLLED INPUT SCHEMA. The router keys on family form; the natural-language pattern
     (two capitalised words) is deliberately broad and would mis-route any "Word Word" sequence in
     arbitrary free text — a router for a known question schema, not an open named-entity recogniser.

The PASS numbers (0 % decisional false-accept / 0 % false-refusal) are meaningful ONLY together with
this frontier. A perfect number shown without its limit is the number-that-lies-by-omission.

All data is synthetic: structured families come from the public `synthetic_facts` module; the
natural-language family is generated in-file from generic invented name pools (no real people).
Run:  python membership_index.py
"""
import sys, os, re, json, argparse, random
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic_facts as SF

# ----------------------------------------------------------------------------- router (form rules)
# Auditable regex by family shape. Order matters: the natural-language name shape is tried first, and
# never matches a structured question (which opens with a single capitalised word, not two).
PATTERNS = [
    ("bio",          re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")),
    ("sensor_calib", re.compile(r"\b[A-Z0-9]{2}-\d{4}\b")),
    ("proto_status", re.compile(r"\b[A-Z]{3,6}-\d\b")),
    ("node_coord",   re.compile(r"\b[NDGR]\d{1,2}\b")),
    ("config_param", re.compile(r"\b[a-z]+_[a-z]+\b")),
    ("service_ver",  re.compile(r"\b[a-z]+-[a-z]+\b")),
]


def route_extract(question):
    """Return (family, identifier, n_hits). Empty family = the router abstains (extraction fail)."""
    hits = [(fam, m.group(0)) for fam, pat in PATTERNS for m in [pat.search(question)] if m]
    if not hits:
        return "", "", 0
    return hits[0][0], hits[0][1], len(hits)


def norm(s):
    return s.strip().lower()


def fuzzy_best(key, index_keys):
    return max((SequenceMatcher(None, key, k).ratio() for k in index_keys), default=0.0)


# --------------------------------------------------------- public natural-language corpus generator
# Generic INVENTED names. Pools are chosen disjoint from any other corpus; every full name is a
# random First+Last combination, so no real individual is referenced.
NL_FIRST = ["Iris", "Kwame", "Sofia", "Ravi", "Elena", "Bjorn", "Amara", "Theo", "Yara", "Cyrus",
            "Noor", "Emil", "Zara", "Otto", "Leila", "Kenji", "Rosa", "Idris", "Vera", "Milo",
            "Anouk", "Dario", "Freya", "Hugo", "Ines", "Jonas", "Kira", "Lars", "Maya", "Niko",
            "Olga", "Pablo", "Rania", "Sven", "Tessa", "Umar", "Wren", "Xavier", "Yusuf", "Zoe"]
NL_LAST = ["Adeyemi", "Fontaine", "Kowalski", "Reyes", "Andersson", "Nguyen", "Bauer", "Costa",
           "Ibrahim", "Lindqvist", "Moreau", "Silva", "Osei", "Kaur", "Romano", "Fischer",
           "Delgado", "Bello", "Marchetti", "Hoffmann", "Abadi", "Vega", "Sorensen", "Park",
           "Haas", "Okonkwo", "Ferreira", "Blum", "Nakamura", "Ortiz", "Lindberg", "Cardoso",
           "Meyer", "Salazar", "Woods", "Kim", "Barbosa", "Grimaldi", "Yildiz", "Aslan"]
NL_CITY = ["Valencia", "Kyoto", "Medellin", "Gdansk", "Uppsala", "Hanoi", "Salvador", "Ankara",
           "Bristol", "Odense", "Freiburg", "Rabat", "Turku", "Braga", "Antwerp", "Nagoya",
           "Linz", "Nantes", "Karachi", "Cuenca"]
NL_PHRASINGS = ["Where was {e} born?", "{e} was born in which city?",
                "In which city was {e} born?", "{e}'s birthplace is"]  # idx 3 = held-out phrasing


def nl_facts(n, seed):
    r = random.Random(SF._stable_seed("bio_pub", seed))
    out, seen = [], set()
    while len(out) < n:
        e = f"{r.choice(NL_FIRST)} {r.choice(NL_LAST)}"
        if e in seen:
            continue
        seen.add(e)
        out.append({"family": "bio", "entity": e, "value": r.choice(NL_CITY)})
    return out


def question(fact, phr):
    if fact["family"] == "bio":
        return NL_PHRASINGS[phr].format(e=fact["entity"])
    return SF.question(fact, phr)


def all_facts(per_family, seed):
    return SF.gen_facts(n_per_family=per_family, seed=seed) + nl_facts(per_family, seed)


# ----------------------------------------------------------------------------------- index + metrics
def build_index(stored, fams):
    idx = {f: set() for f in fams}
    for x in stored:
        idx[x["family"]].add(norm(x["entity"]))
    return idx


def evaluate(per_family=60, seed_stored=11, seed_fake=101, fuzzy_tau=0.90):
    """Build an index on `stored`, evaluate against `fakes` (disjoint seed). Mirrors the frozen
    criterion: structured mis-route <= 0.5 %, internal collision <= 0.1 %, decisional false-accept
    on NOVEL fakes <= 1 %, false-refusal <= 15 %. Pool collisions (a fake whose identifier is already
    stored) are NOT decisional false-accepts -- they are correct by identity and reported separately."""
    fams = SF.FAMILIES + ["bio"]
    stored = all_facts(per_family, seed_stored)
    fakes = all_facts(per_family, seed_fake)
    index = build_index(stored, fams)
    stored_ids = {f: set(norm(x["entity"]) for x in stored if x["family"] == f) for f in fams}

    def route_audit(facts):
        mis = fail = n = 0
        for x in facts:
            for phr in range(4):
                fam, ent, _ = route_extract(question(x, phr)); n += 1
                if not fam:
                    fail += 1; continue
                if fam != x["family"] or norm(ent) != norm(x["entity"]):
                    mis += 1
        return {"n": n, "misroute": mis, "extract_fail": fail,
                "misroute_rate": round(mis / n, 5), "extract_fail_rate": round(fail / n, 5)}

    r_stored, r_fake = route_audit(stored), route_audit(fakes)

    # pool collisions: a fake whose identifier is already stored (finite-space artifact, not a fault)
    pool_coll = {f: sum(1 for x in fakes if x["family"] == f and norm(x["entity"]) in stored_ids[f]) for f in fams}
    n_fake_by_fam = {f: sum(1 for x in fakes if x["family"] == f) for f in fams}

    # internal collisions among structured stored (2 distinct entities -> same key)
    coll = tot = 0
    for f in SF.FAMILIES:
        keys = [norm(x["entity"]) for x in stored if x["family"] == f]
        tot += len(keys); coll += len(keys) - len(set(keys))
    coll_rate = round(coll / max(1, tot), 5)

    # false-refusal: stored queried on phrasings 2 (train) and 3 (held-out) must be found
    rr = {"struct": [0, 0], "nl": [0, 0]}
    for x in stored:
        for phr in (2, 3):
            fam, ent, _ = route_extract(question(x, phr))
            if not fam:
                continue
            p = "nl" if fam == "bio" else "struct"; rr[p][1] += 1
            rr[p][0] += int(norm(ent) in index[fam])
    fr_struct = round(1 - rr["struct"][0] / max(1, rr["struct"][1]), 5)
    fr_nl = round(1 - rr["nl"][0] / max(1, rr["nl"][1]), 5)

    # decisional false-accept: NOVEL fakes only (identifier not stored); pool collisions excluded
    fa = {"struct": {"n": 0, "exact": 0}, "nl": {"n": 0, "exact": 0, "fuzzy": 0, "fuzzy_eval": 0, "fuzzy_max": 0.0}}
    for x in fakes:
        if norm(x["entity"]) in stored_ids[x["family"]]:
            continue  # pool collision -> correct by identity, not a decisional false-accept
        for phr in (2, 3):
            fam, ent, _ = route_extract(question(x, phr))
            if not fam:
                continue
            p = "nl" if fam == "bio" else "struct"; key = norm(ent); fa[p]["n"] += 1
            ex = key in index[fam]; fa[p]["exact"] += int(ex)
            if p == "nl" and not ex:
                fb = fuzzy_best(key, index["bio"])          # fuzzy path is exercised on every novel NL miss
                fa["nl"]["fuzzy_eval"] += 1
                fa["nl"]["fuzzy_max"] = max(fa["nl"]["fuzzy_max"], fb)
                if fb >= fuzzy_tau:
                    fa["nl"]["fuzzy"] += 1
    fa_struct = round(fa["struct"]["exact"] / max(1, fa["struct"]["n"]), 5)
    fa_nl = round((fa["nl"]["exact"] + fa["nl"]["fuzzy"]) / max(1, fa["nl"]["n"]), 5)

    ok = (r_stored["misroute_rate"] <= 0.005 and r_fake["misroute_rate"] <= 0.005 and coll_rate <= 0.001
          and fa_struct <= 0.001 and fa_nl <= 0.01 and fr_struct <= 0.15 and fr_nl <= 0.15)
    res = {
        "n_stored": len(stored), "n_fake": len(fakes),
        "router_stored": r_stored, "router_fake": r_fake,
        "struct_internal_collision_rate": coll_rate,
        "pool_collisions_by_family": {f: f"{pool_coll[f]}/{n_fake_by_fam[f]}" for f in fams},
        "false_refusal_struct": fr_struct, "false_refusal_nl": fr_nl,
        "false_accept_struct_NOVEL": fa_struct, "n_struct_novel_q": fa["struct"]["n"],
        "false_accept_nl_NOVEL_incl_fuzzy": fa_nl, "n_nl_novel_q": fa["nl"]["n"],
        "nl_fuzzy_evaluated": fa["nl"]["fuzzy_eval"], "nl_fuzzy_triggered": fa["nl"]["fuzzy"],
        "nl_fuzzy_max_ratio": round(fa["nl"]["fuzzy_max"], 4),
        "fuzzy_tau": fuzzy_tau, "PASS": ok,
    }
    return res, index, stored


def frontier_demo(index, stored):
    """Make FRONTIER limit 1 executable: a fake with a REAL identifier + FALSE attribute is accepted."""
    print("\n--- FRONTIER (limit 1: identifier membership, NOT attribute truth) ---")
    victim = next(x for x in stored if x["family"] == "bio")
    wrong_city = next(c for c in NL_CITY if c != victim["value"])
    false_bio = {"family": "bio", "entity": victim["entity"], "value": wrong_city}
    fam, ent, _ = route_extract(question(false_bio, 2))
    print(f"  fake bio '{false_bio['entity']}' claims birthplace {wrong_city} "
          f"(stored value is {victim['value']}) -> index says stored = {norm(ent) in index[fam]}")
    vs = next(x for x in stored if x["family"] == "sensor_calib")
    wrong_cal = "1A2B-3C4D" if vs["value"] != "1A2B-3C4D" else "5E6F-7A8B"  # valid format, wrong value
    false_s = {"family": "sensor_calib", "entity": vs["entity"], "value": wrong_cal}
    fam2, ent2, _ = route_extract(question(false_s, 2))
    print(f"  fake sensor '{vs['entity']}' claims calibration {wrong_cal} (stored value is {vs['value']}) "
          f"-> index says stored = {norm(ent2) in index[fam2]}")
    print("  => The index matches the IDENTIFIER; it does not check the attribute. Both are accepted "
          "as stored. This is limit 1, running.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_family", type=int, default=60)
    ap.add_argument("--seed_stored", type=int, default=11)
    ap.add_argument("--seed_fake", type=int, default=101)
    ap.add_argument("--fuzzy_tau", type=float, default=0.90)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res, index, stored = evaluate(a.per_family, a.seed_stored, a.seed_fake, a.fuzzy_tau)
    print("=== membership index vs frozen criterion (synthetic) ===")
    print(f"  router  stored misroute/fail = {res['router_stored']['misroute_rate']}/{res['router_stored']['extract_fail_rate']}"
          f"  |  fake = {res['router_fake']['misroute_rate']}/{res['router_fake']['extract_fail_rate']}")
    print(f"  internal-collision rate (struct) = {res['struct_internal_collision_rate']}")
    print(f"  pool collisions/family (fake id already stored, excluded from false-accept): {res['pool_collisions_by_family']}")
    print(f"  false-refusal  struct = {res['false_refusal_struct']}  nl = {res['false_refusal_nl']}")
    print(f"  false-accept on NOVEL fakes  struct = {res['false_accept_struct_NOVEL']} (n={res['n_struct_novel_q']})"
          f"  nl(+fuzzy) = {res['false_accept_nl_NOVEL_incl_fuzzy']} (n={res['n_nl_novel_q']})")
    print(f"  NL fuzzy path: evaluated={res['nl_fuzzy_evaluated']}  triggered(>= tau {res['fuzzy_tau']})={res['nl_fuzzy_triggered']}"
          f"  max_ratio_seen={res['nl_fuzzy_max_ratio']}  (triggered=0 with max_ratio<tau = fuzzy exercised, flagged nothing)")
    print(f"  VERDICT: {'PASS' if res['PASS'] else 'FAIL'} "
          "(meaningful only with the FRONTIER below / in docs/EXTERNAL_VERIFICATION.md)")
    frontier_demo(index, stored)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
