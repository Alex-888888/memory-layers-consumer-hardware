# -*- coding: utf-8 -*-
"""Synthetic fact families + generic negatives for training/evaluating the relevance gate.

Five families with genuinely DIFFERENT structures (entity shape AND value shape), so the
gate cannot rely on a single surface template:

  sensor_calib : entity 2L-4D        value HHHH-HHHH (hex pair)
  config_param : entity snake_case   value integer
  service_ver  : entity kebab-case   value semver vX.Y.Z
  node_coord   : entity Nn / id      value "lat, lon" (signed decimals)
  proto_status : entity CODENAME-n   value 0xNN (hex byte)

Each family has four question phrasings: indices {0,1,2} are used to TRAIN the gate, index
3 is HELD OUT (never seen by the gate) for the held-out-phrasing generalisation test.
All content here is synthetic and contains no project-specific data.
"""
import random

FAMILIES = ["sensor_calib", "config_param", "service_ver", "node_coord", "proto_status"]

PHRASINGS = {
    "sensor_calib": [
        "What is the calibration identifier of sensor {e}?",
        "Sensor {e} — what is its calibration ID?",
        "Give the calibration code for sensor {e}.",
        "For sensor {e}, the calibration identifier is",            # held-out (D)
    ],
    "config_param": [
        "What value is configured for the parameter {e}?",
        "The configuration setting {e} is set to what?",
        "What is the value of {e} in the configuration?",
        "Configuration parameter {e} equals",                      # D
    ],
    "service_ver": [
        "Which version is the service {e} running?",
        "What is the deployed version of {e}?",
        "Service {e} is at which version?",
        "The current version of {e} is",                           # D
    ],
    "node_coord": [
        "What are the grid coordinates of node {e}?",
        "Where is node {e} located on the grid?",
        "Give the coordinates for node {e}.",
        "Node {e} sits at which coordinates",                      # D
    ],
    "proto_status": [
        "What status code does the {e} handshake return?",
        "The {e} protocol handshake yields which status?",
        "What is the status code for handshake {e}?",
        "Handshake {e} returns status",                            # D
    ],
}
DECLARATIVE = {
    "sensor_calib": "The calibration identifier of sensor {e} is {v}.",
    "config_param": "The configuration parameter {e} is set to {v}.",
    "service_ver":  "The service {e} is running version {v}.",
    "node_coord":   "Node {e} is located at grid coordinates {v}.",
    "proto_status": "The {e} protocol handshake returns status {v}.",
}


def _entity(fam, r):
    if fam == "sensor_calib":
        return f"{r.choice(['KX','ZR','AX','TR','PL','NX','C6'])}-{r.randint(1000,9999)}"
    if fam == "config_param":
        a = r.choice(["max","min","default","initial","target","cache","retry","queue","buffer","poll"])
        b = r.choice(["retries","timeout","size","depth","interval","limit","workers","ttl","window","backoff"])
        return f"{a}_{b}"
    if fam == "service_ver":
        a = r.choice(["auth","data","edge","core","mesh","relay","index","sync","proxy","vault"])
        b = r.choice(["gateway","broker","router","engine","service","daemon","manager","node"])
        return f"{a}-{b}"
    if fam == "node_coord":
        return f"{r.choice(['N','D','G','R'])}{r.randint(1,99)}"
    if fam == "proto_status":
        return r.choice(["ORION","VEGA","LYRA","ATLAS","NOVA","TITAN","HELIX","CETUS","DRACO","MIRA"]) + f"-{r.randint(1,9)}"


def _value(fam, r):
    if fam == "sensor_calib":
        return f"{r.randint(0,65535):04X}-{r.randint(0,65535):04X}"
    if fam == "config_param":
        return str(r.randint(1, 999))
    if fam == "service_ver":
        return f"v{r.randint(0,9)}.{r.randint(0,30)}.{r.randint(0,30)}"
    if fam == "node_coord":
        return f"{r.uniform(-89,89):.2f}, {r.uniform(-179,179):.2f}"
    if fam == "proto_status":
        return f"0x{r.randint(0,255):02X}"


def chatml(q, a=None):
    s = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return s if a is None else s + a + ".<|im_end|>"


def gen_facts(n_per_family=60, seed=11, families=None):
    """Return a list of {family, entity, value} dicts, unique entity per family."""
    families = families or FAMILIES
    out = []
    for fam in families:
        r = random.Random(hash((fam, seed)) & 0xffffffff)
        seen = set()
        while len([o for o in out if o["family"] == fam]) < n_per_family:
            e = _entity(fam, r)
            if e in seen:
                continue
            seen.add(e)
            out.append({"family": fam, "entity": e, "value": _value(fam, r)})
    return out


def question(fact, phr_idx):
    return PHRASINGS[fact["family"]][phr_idx].format(e=fact["entity"])


def train_phrasings():
    return [0, 1, 2]


def heldout_phrasing():
    return 3


# Generic negatives for the gate (neutral prose + general factual questions).
# The gate must CLOSE on these. No project-specific content.
NEG_PROSE = [
    "The history of cartography reflects how societies understood the world.",
    "Photosynthesis converts light into chemical energy for life.",
    "Comparative advantage explains why trade benefits both parties.",
    "Ocean currents redistribute heat across the globe.",
    "The printing press lowered the cost of copying texts.",
    "Vaccination trains the immune system to recognize a pathogen.",
    "A small interface with clear contracts is easier to maintain.",
    "Mountains form as tectonic plates collide over millions of years.",
    "Reading regularly improves vocabulary and comprehension.",
    "Scientific progress relies on careful observation and repeatable experiments.",
    "Good documentation anticipates the questions a newcomer will ask.",
    "Musical harmony arises from relationships between note frequencies.",
]
NEG_QUESTIONS = [
    "Who painted the Mona Lisa?", "Which planet is closest to the sun?",
    "Who wrote the play Hamlet?", "In what year did World War II end?",
    "What is the tallest animal in the world?", "Which element has the atomic number 1?",
    "What is the longest river in the world?", "Who developed the theory of relativity?",
    "What currency is used in Japan?", "Which country has the most population?",
    "What is the speed of light approximately?", "Who was the first person on the moon?",
    "What language is spoken in Brazil?", "Which gas do plants absorb from the air?",
    "What is the hardest natural substance?", "Who composed the Ninth Symphony?",
    "What is the capital of Australia?", "Which sea is the saltiest?",
    "Who discovered penicillin?", "What is the largest planet in the solar system?",
]


if __name__ == "__main__":
    facts = gen_facts(n_per_family=2, seed=11)
    for fam in FAMILIES:
        f = next(x for x in facts if x["family"] == fam)
        print(f"[{fam}] {f['entity']!r} -> {f['value']!r}")
        for i in train_phrasings():
            print(f"   train phr{i}: {question(f, i)!r}")
        print(f"   held-out phr{heldout_phrasing()}: {question(f, heldout_phrasing())!r}")
        print(f"   declarative: {DECLARATIVE[fam].format(e=f['entity'], v=f['value'])!r}")
