# -*- coding: utf-8 -*-
"""End-to-end: train a relevance gate on top of a frozen LLM + frozen parametric memory.

Pipeline (single process, no checkpoint juggling):
  1) build a FROZEN Qwen2.5-7B backbone + a fresh product-key memory (MLP-ADD at layers
     6/14/22), and train the MEMORY on synthetic multi-family facts (all phrasings);
  2) freeze the memory; train the RELEVANCE GATE on phrasings {0,1,2} only (positives =
     stored-fact contexts, negatives = generic prose + general questions);
  3) evaluate: held-out-phrasing recall (gate trained on {0,1,2}, tested on phrasing 3 of
     the SAME entities), gate open-rate, perplexity on WikiText-103, and TriviaQA.

Everything is synthetic / public. Requires `warmup_train.py` (same package) for the memory
classes. Run `python train_relevance_gate.py --dry 1` for a fast smoke test.
"""
import sys, os, json, math, time, argparse, random
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warmup_train import ProductKeyMemoryPlus, MLPPlusMemory, CPUOffloadAdam, PLACEMENT
from relevance_gate import RelevanceGate, GatedMemoryMLP, collect_features, train_gates
import synthetic_facts as SF

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEV = "cuda"


def setcfg(wrappers, mode):
    for w in wrappers:
        if mode == "backbone":
            w.mem_off = True
        else:
            w.mem_off = False
            w.gate_on = (mode == "gated")


@torch.no_grad()
def recall(model, tok, qa, mx=12):
    ok = 0
    for q, v in qa:
        ids = tok(SF.chatml(q), return_tensors="pt").input_ids.to(DEV)
        o = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).lower().replace(" ", "")
        ok += int(v.lower() in gen)
    return ok / max(1, len(qa))


@torch.no_grad()
def ppl_wikitext(model, tok, max_tok, win=2048):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n".join(t for t in ds["text"] if t and not t.isspace())
    ids = tok(text, return_tensors="pt").input_ids[0][:max_tok]
    nll = 0.0; n = 0
    for i in range(0, ids.shape[0] - 1, win):
        ch = ids[i:i + win + 1]
        if ch.shape[0] < 2:
            continue
        x = ch[:-1].unsqueeze(0).to(DEV); y = ch[1:].unsqueeze(0).to(DEV)
        lo = model(x).logits.float()
        nll += F.cross_entropy(lo.reshape(-1, lo.size(-1)), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return math.exp(nll / n)


def load_trivia(n):
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.nocontext", split=f"validation[:{n}]")
        return [(r["question"], r["answer"].get("aliases", []) or [r["answer"].get("value", "")]) for r in ds]
    except Exception as e:
        print(f"[trivia] unavailable: {str(e)[:60]}", flush=True)
        return None


@torch.no_grad()
def recall_trivia(model, tok, qa, mx=24):
    ok = 0
    for q, aliases in qa:
        ids = tok(SF.chatml(q), return_tensors="pt").input_ids.to(DEV)
        o = model.generate(ids, max_new_tokens=mx, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).lower()
        ok += int(any(a.lower() in gen for a in aliases if a))
    return ok / max(1, len(qa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_family", type=int, default=60)
    ap.add_argument("--target_exp", type=int, default=40, help="memory training exposures per fact")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool_size", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--ppl_max", type=int, default=40000)
    ap.add_argument("--trivia", type=int, default=300)
    ap.add_argument("--out", default="relevance_gate_results.json")
    ap.add_argument("--dry", type=int, default=0)
    a = ap.parse_args()
    if a.dry:
        a.per_family, a.target_exp, a.trivia, a.ppl_max = 4, 3, 10, 3000

    random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = int(round(a.pool_size ** 0.5)); pool_n = m * m
    facts = SF.gen_facts(n_per_family=a.per_family, seed=a.seed)
    tr_phr, ho_phr = SF.train_phrasings(), SF.heldout_phrasing()
    mem_pairs = [(SF.question(f, pi), f["value"]) for f in facts for pi in tr_phr + [ho_phr]]
    steps = math.ceil(a.target_exp * len(mem_pairs) / a.batch)
    print(f"[gate] families={len(SF.FAMILIES)} facts={len(facts)} mem_seqs={len(mem_pairs)} pool={pool_n} steps={steps} batch={a.batch}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
    model.gradient_checkpointing_enable(); model.config.use_cache = False
    d = model.config.hidden_size
    pool = nn.Parameter(torch.empty(pool_n, d, device=DEV, dtype=torch.bfloat16).normal_(0, d ** -0.5))
    blocks, gates, wr = [], [], []
    for i in PLACEMENT:
        blk = ProductKeyMemoryPlus(d, m, d, pool).to(DEV, torch.bfloat16)
        g = RelevanceGate(d).to(DEV)
        gm = GatedMemoryMLP(model.model.layers[i].mlp, blk, g)
        model.model.layers[i].mlp = gm
        blocks.append(blk); gates.append(g); wr.append(gm)
    for p in model.parameters():
        p.requires_grad = False

    # ---- 1) train the MEMORY (gate off) ----
    for w in wr:
        w.mem_off = False; w.gate_on = False
    for blk in blocks:
        for p in blk.parameters():
            p.requires_grad = True
    for g in gates:
        for p in g.parameters():
            p.requires_grad = False
    pool.requires_grad = True
    bp = [p for b in blocks for p in b.parameters() if p.data_ptr() != pool.data_ptr()]
    opt_b = torch.optim.AdamW(bp, lr=a.lr)
    opt_p = CPUOffloadAdam(pool, lr=a.lr, state_dtype=torch.bfloat16)
    data = []
    for q, v in mem_pairs:
        ids = tok(SF.chatml(q, v), return_tensors="pt").input_ids[0]
        plen = tok(SF.chatml(q), return_tensors="pt").input_ids.shape[1]
        lab = ids.clone(); lab[:plen] = -100
        data.append((ids, lab))
    eos = tok.eos_token_id; model.train(); t0 = time.time()
    for st in range(1, steps + 1):
        seqs = [data[random.randrange(len(data))] for _ in range(a.batch)]
        maxlen = max(s[0].shape[0] for s in seqs)
        X = torch.full((a.batch, maxlen), eos, dtype=torch.long)
        L = torch.full((a.batch, maxlen), -100, dtype=torch.long)
        for bi, (ids, lab) in enumerate(seqs):
            X[bi, :ids.shape[0]] = ids; L[bi, :lab.shape[0]] = lab
        X = X.to(DEV); L = L.to(DEV)
        cur_lr = a.lr * min(1.0, st / max(1, a.warmup))
        for gg in opt_b.param_groups:
            gg["lr"] = cur_lr
        opt_p.lr = cur_lr
        out = model(X).logits.float()
        loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), L[:, 1:].reshape(-1), ignore_index=-100)
        if torch.isfinite(loss):
            opt_b.zero_grad(); opt_p.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            if pool.grad is not None:
                torch.nn.utils.clip_grad_norm_([pool], 1.0)
            opt_b.step(); opt_p.step()
        if st % 500 == 0:
            print(f"  [mem] step {st}/{steps} loss {loss.item():.3f}", flush=True)
    print(f"[gate] memory trained ({time.time() - t0:.0f}s)", flush=True)
    for blk in blocks:
        for p in blk.parameters():
            p.requires_grad = False
    pool.requires_grad = False
    model.gradient_checkpointing_disable()

    # ---- 2) train the GATE on phrasings {0,1,2} only ----
    for g in gates:
        for p in g.parameters():
            p.requires_grad = True
    pos = [(SF.question(f, pi), f["value"]) for f in facts for pi in tr_phr]
    neg = SF.NEG_PROSE + [SF.chatml(q) for q in SF.NEG_QUESTIONS]
    model.gradient_checkpointing_enable()
    Xc, Yc = collect_features(model, wr, tok, SF.chatml, pos, neg, DEV)
    model.gradient_checkpointing_disable()
    print(f"[gate] features collected (pos={len(pos)} neg={len(neg)})", flush=True)
    train_gates(gates, Xc, Yc, DEV)

    # ---- 3) evaluation ----
    def gate_openrate(qa):
        for w in wr:
            w.mem_off = False; w.gate_on = False
        vals = []
        with torch.no_grad():
            for q, _ in qa:
                ids = tok(SF.chatml(q), return_tensors="pt").input_ids.to(DEV)
                model(ids)
                s = torch.sigmoid(gates[0](wr[0].last_x[0].float())).mean().item()
                vals.append(s)
        return sum(vals) / max(1, len(vals))

    rng = random.Random(0)
    sample = lambda phr, k=120: [(SF.question(f, phr), f["value"]) for f in rng.sample(facts, min(k, len(facts)))]
    train_q = sample(tr_phr[0])
    held_q = [(SF.question(f, ho_phr), f["value"]) for f in random.Random(0).sample(facts, min(120, len(facts)))]
    trivia = load_trivia(a.trivia)

    res = {}
    for mode in ["backbone", "ungated", "gated"]:
        setcfg(wr, mode)
        row = {"recall_train_phr": recall(model, tok, train_q), "recall_heldout_phr": recall(model, tok, held_q)}
        if trivia:
            row["triviaqa"] = recall_trivia(model, tok, trivia)
        res[mode] = row
        print(f"[{mode:8s}] recall_train_phr {row['recall_train_phr']*100:.1f}% | recall_HELDOUT_phr {row['recall_heldout_phr']*100:.1f}%"
              + (f" | TriviaQA {row['triviaqa']*100:.1f}%" if trivia else ""), flush=True)
        for w in wr:
            w.mem_off = True

    ppl = {}
    for mode in ["backbone", "gated"]:
        setcfg(wr, mode); ppl[mode] = ppl_wikitext(model, tok, a.ppl_max)
        for w in wr:
            w.mem_off = True
    res["ppl_wikitext"] = ppl
    res["gate_openrate"] = {"train_phr": gate_openrate(train_q), "heldout_phr": gate_openrate(held_q)}

    u, g = res["ungated"], res["gated"]
    print("\n=== relevance gate — multi-domain + held-out-phrasing ===", flush=True)
    print(f"  held-out-phrasing recall : ungated {u['recall_heldout_phr']*100:.1f}% -> gated {g['recall_heldout_phr']*100:.1f}% (must stay ~ungated)", flush=True)
    print(f"  gate open-rate : train {res['gate_openrate']['train_phr']:.2f} | held-out {res['gate_openrate']['heldout_phr']:.2f}", flush=True)
    print(f"  PPL WikiText : backbone {ppl['backbone']:.2f} -> gated {ppl['gated']:.2f} ({100*(ppl['gated']-ppl['backbone'])/ppl['backbone']:+.1f}%)", flush=True)
    if trivia:
        b = res["backbone"]["triviaqa"]
        print(f"  TriviaQA : backbone {b*100:.1f}% -> gated {g['triviaqa']*100:.1f}% (ungated {u['triviaqa']*100:.1f}%)", flush=True)
    drop = u["recall_heldout_phr"] - g["recall_heldout_phr"]
    print(f"  VERDICT held-out phrasing : {'GENERALISES (drop %.0f pts)' % (drop*100) if drop < 0.1 else 'TEMPLATE-BOUND (drop %.0f pts)' % (drop*100)}", flush=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
