# -*- coding: utf-8 -*-
"""Generate a synthetic fact dataset for the Memory Layers reconstruction.

Each fact is an arbitrary entity -> value association the base model cannot know
(baseline recall ~0%), so recall measures the memory only.

Format per fact (ChatML where relevant):
  - declarative  : "... the calibration identifier of sensor <ID> is <HEX>."
  - QA (bare answer) : Q "What is the calibration identifier of sensor <ID>?" -> "<HEX>."

The bare-answer QA form is the one that the validated recipe trains on
(answer-only loss + one sequence per fact). The declarative form lets the eval
harness extract the (entity, value) pairs.

Usage:
  python make_synthetic.py --n 500 --out ../../data/synthetic_500.jsonl --seed 7
"""
import json, argparse, random

PREFIXES = ["KX", "ZR", "AX", "LM", "Q7", "BV", "TR", "NX", "PL", "C6"]

def chatml(q, a):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"

def generate(n, seed):
    rng = random.Random(seed)
    seen, facts = set(), []
    while len(facts) < n:
        ent = f"{rng.choice(PREFIXES)}-{rng.randint(1000, 9999)}"
        if ent in seen:
            continue
        seen.add(ent)
        code = f"{rng.randint(0, 0xffff):04X}-{rng.randint(0, 0xffff):04X}"
        facts.append((ent, code))
    return facts

def to_records(facts):
    recs = []
    for ent, code in facts:
        recs.append({"text": f"In the experimental registry, the calibration identifier of sensor {ent} is {code}.",
                     "segment": "D", "lang": "en"})
        recs.append({"text": chatml(f"What is the calibration identifier of sensor {ent}?", f"{code}."),
                     "segment": "D", "lang": "en"})
        recs.append({"text": chatml(f"Quel est l'identifiant d'etalonnage du capteur {ent} ?", f"{code}."),
                     "segment": "D", "lang": "fr"})
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of distinct facts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="synthetic.jsonl")
    a = ap.parse_args()
    recs = to_records(generate(a.n, a.seed))
    random.Random(a.seed).shuffle(recs)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.n} facts -> {len(recs)} records -> {a.out}")

if __name__ == "__main__":
    main()
