# -*- coding: utf-8 -*-
"""Build a mixed training corpus for the warm-up, mirroring the validated recipe:
  - synthetic facts (the memorisation target, baseline recall ~0%)
  - a small set of generic public facts as a "known-knowledge anchor" (QA, answer-only)
  - neutral fluency prose (perplexity guard)
  - a few instruction examples

All public/synthetic content (no private data). The known-anchor + fluency is what
keeps the model's native knowledge intact while the memory learns the synthetic facts.

Usage:
  python make_corpus.py --synth 5000 --out ../../data/corpus.jsonl
"""
import json, argparse, random
from make_synthetic import generate as gen_synth, to_records as synth_records, chatml

# --- generic, public facts used as a knowledge-preservation anchor (QA + declarative) ---
FACTS = [
 ("Paris is the capital of France.", "What is the capital of France?", "Paris."),
 ("Tokyo is the capital of Japan.", "What is the capital of Japan?", "Tokyo."),
 ("Ottawa is the capital of Canada.", "What is the capital of Canada?", "Ottawa."),
 ("The chemical symbol for gold is Au.", "What is the chemical symbol for gold?", "Au."),
 ("The chemical symbol for oxygen is O.", "What is the chemical symbol for oxygen?", "O."),
 ("Water is composed of hydrogen and oxygen.", "What is the chemical formula of water?", "H2O."),
 ("The Pacific is the largest ocean on Earth.", "What is the largest ocean on Earth?", "The Pacific Ocean."),
 ("A triangle has three sides.", "How many sides does a triangle have?", "Three."),
 ("Carbon has the atomic number 6.", "What is the atomic number of carbon?", "6."),
 ("Mount Everest is the highest mountain above sea level.", "What is the highest mountain above sea level?", "Mount Everest."),
]

# --- neutral fluency prose (perplexity anchor) ---
FLUENCY = [
 "Engineers usually perform several design iterations before validation. Measurements are compared with simulations, and any discrepancy is investigated before moving on.",
 "Reading regularly improves vocabulary and comprehension. Over time, readers recognise patterns in arguments and follow longer chains of reasoning with less effort.",
 "When planning a project, it helps to break the work into small, well-defined tasks that can be estimated, scheduled, and checked off as progress is made.",
 "Scientific progress relies on careful observation and repeatable experiments. Results that cannot be reproduced are treated with caution until further evidence is gathered.",
 "A good explanation starts from what the listener already knows and adds one new idea at a time. Clear examples make abstract concepts easier to grasp.",
]

# --- a few instruction examples ---
INSTR = [
 ("Explain in one sentence what gradient checkpointing does.",
  "Gradient checkpointing trades extra compute for memory by recomputing intermediate activations during the backward pass instead of storing them."),
 ("What is the purpose of a learning-rate warm-up?",
  "It gradually increases the learning rate at the start of training to avoid unstable updates while the optimizer state settles."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--anchor_repeat", type=int, default=40)
    ap.add_argument("--fluency_repeat", type=int, default=80)
    ap.add_argument("--out", default="corpus.jsonl")
    a = ap.parse_args()
    docs = []
    docs += synth_records(gen_synth(a.synth, a.seed))                     # D : memorisation target
    for _ in range(a.anchor_repeat):                                      # B/C : known-knowledge anchor
        for decl, q, ans in FACTS:
            docs.append({"text": decl, "segment": "B", "lang": "en"})
            docs.append({"text": chatml(q, ans), "segment": "C", "lang": "en"})
    for _ in range(a.fluency_repeat):                                     # A : fluency / PPL guard
        for g in FLUENCY:
            docs.append({"text": g, "segment": "A", "lang": "en"})
    for _ in range(20):                                                   # F : instruction
        for q, ans in INSTR:
            docs.append({"text": chatml(q, ans), "segment": "F", "lang": "en"})
    random.Random(a.seed).shuffle(docs)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"{a.synth} synthetic facts | {len(docs)} records | segments {dict(Counter(d['segment'] for d in docs))} -> {a.out}")

if __name__ == "__main__":
    main()
