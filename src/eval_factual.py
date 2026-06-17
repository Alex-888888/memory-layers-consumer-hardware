"""Harnais d'eval factuel : rappel sur (a) associations synthetiques DU corpus (Qwen ne les connait pas
-> baseline ~0%, mesure pure de la memoire) et (b) faits connus (reference). Baseline vs memoire entrainee.
"""
import json, re, random, torch, torch.nn as nn, torch.nn.functional as F
from warmup_train import ProductKeyMemoryPlus, MLPPlusMemory
import os
PLACEMENT=[6,14,22]; DEV="cuda"
CORPUS=os.environ.get("MLCH_EVAL_CORPUS","./data/corpus.jsonl")
def _latest_ckpt():
    env=os.environ.get("MLCH_EVAL_CKPT")
    if env: return env
    d="./checkpoints"
    if os.path.isdir(d):
        cks=[f for f in os.listdir(d) if f.startswith("warmup_step") and f.endswith(".pt")]
        if cks: return os.path.join(d, max(cks, key=lambda f:int(f[len("warmup_step"):-3])))
    return "./checkpoints/warmup_step120.pt"
CKPT=_latest_ckpt()

def chatml(q): return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"

def load_eval_sets():
    synth=[]
    rx=re.compile(r"sensor (\S+) is ([0-9A-Fa-f]{4}-[0-9A-Fa-f]{4})\.")
    for line in open(CORPUS,encoding="utf-8"):
        try: d=json.loads(line)
        except: continue
        if d.get("segment")=="D" and d.get("lang")=="en":
            m=rx.search(d["text"])
            if m: synth.append((f"What is the calibration identifier of sensor {m.group(1)}?", m.group(2)))
    random.Random(0).shuffle(synth); synth=synth[:40]
    known=[("What is the capital of France?","Paris"),("What is the capital of Japan?","Tokyo"),
           ("What is the capital of Italy?","Rome"),("What is the capital of Canada?","Ottawa"),
           ("What is the chemical symbol of gold?","Au"),("What is the chemical symbol of oxygen?","O"),
           ("What is the capital of Spain?","Madrid"),("What is the capital of Germany?","Berlin"),
           ("What is the capital of Egypt?","Cairo"),("What is the capital of Brazil?","Bras")]
    return synth, known

@torch.no_grad()
def recall(model, tok, qa, tag):
    ok=0
    for q,gold in qa:
        ids=tok(chatml(q),return_tensors="pt").input_ids.to(DEV)
        out=model.generate(ids,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
        gen=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)
        if gold.lower().replace(" ","") in gen.lower().replace(" ",""): ok+=1
    print(f"[{tag}] recall {ok}/{len(qa)} = {100*ok/len(qa):.1f}%", flush=True)
    return ok/len(qa)

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",torch_dtype=torch.bfloat16).to(DEV)
    for p in model.parameters(): p.requires_grad=False
    torch.cuda.empty_cache()
    d=model.config.hidden_size
    _ck0=torch.load(CKPT,map_location="cpu")   # deduit la taille du pool depuis le checkpoint
    M=int(round(_ck0["pool"].shape[0]**0.5))
    print(f"[eval] pool {_ck0['pool'].shape[0]} (M={M}) depuis {os.path.basename(CKPT)}", flush=True)
    pool=nn.Parameter(torch.empty(M*M,d,device=DEV,dtype=torch.bfloat16).normal_(0,d**-0.5))
    blocks=[]
    for i in PLACEMENT:
        blk=ProductKeyMemoryPlus(d,M,d,pool).to(DEV)
        model.model.layers[i].mlp=MLPPlusMemory(model.model.layers[i].mlp,blk); blocks.append(blk)
    model.eval()
    synth,known=load_eval_sets()
    print(f"eval sets: synth={len(synth)} known={len(known)}", flush=True)
    print("=== BASELINE (memoire out_proj=0 -> Qwen pur) ===", flush=True)
    bs_syn=recall(model,tok,synth,"baseline synth"); bs_kn=recall(model,tok,known,"baseline known")
    print(f"=== MEMOIRE ENTRAINEE (ckpt {os.path.basename(CKPT)}) ===", flush=True)
    ck=_ck0
    for b,sd in zip(blocks,ck["blocks"]): b.load_state_dict(sd)
    pool.data.copy_(ck["pool"].to(DEV,torch.bfloat16))
    tr_syn=recall(model,tok,synth,"trained synth"); tr_kn=recall(model,tok,known,"trained known")
    print(f"\nRESUME synth baseline {bs_syn*100:.1f}% -> trained {tr_syn*100:.1f}% | known {bs_kn*100:.1f}% -> {tr_kn*100:.1f}%", flush=True)
    print("EVAL_DONE", flush=True)

if __name__=="__main__": main()
