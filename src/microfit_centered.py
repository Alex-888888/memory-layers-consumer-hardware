import os, random, torch, torch.nn as nn, torch.nn.functional as F, argparse
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
from warmup_train import ProductKeyMemoryPlus, MLPPlusMemory
from transformers import AutoModelForCausalLM, AutoTokenizer
def chatml(q,a=None):
    s=f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    return s if a is None else s+a+"<|im_end|>"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--center",type=int,default=1)
    ap.add_argument("--steps",type=int,default=400)
    ap.add_argument("--nfacts",type=int,default=50)
    ap.add_argument("--pool",type=int,default=50000)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--eval_n",type=int,default=0)
    a=ap.parse_args()
    DEV="cuda"; PLACEMENT=[6,14,22]
    print(f"=== MICRO-OVERFIT center={a.center} nfacts={a.nfacts} steps={a.steps} pool={a.pool} ===",flush=True)
    tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",torch_dtype=torch.bfloat16).to(DEV)
    for p in model.parameters(): p.requires_grad=False
    d=model.config.hidden_size; M=int(a.pool**0.5)
    pool=nn.Parameter(torch.empty(M*M,d,device=DEV,dtype=torch.bfloat16).normal_(0,d**-0.5))
    blocks=[]
    for i in PLACEMENT:
        blk=ProductKeyMemoryPlus(d,M,d,pool).to(DEV); blk.center=bool(a.center)
        model.model.layers[i].mlp=MLPPlusMemory(model.model.layers[i].mlp,blk); blocks.append(blk)
        for p in blk.parameters(): p.requires_grad=True
    pool.requires_grad=True
    model.train()
    rng=random.Random(42)
    def rhex(): return f"{rng.randint(0,65535):04X}-{rng.randint(0,65535):04X}"
    facts=[(f"KX-{1000+k}", rhex()) for k in range(a.nfacts)]
    Q=lambda sid:f"What is the calibration identifier of sensor {sid}?"
    data=[]
    for sid,hx in facts:
        ids=tok(chatml(Q(sid),hx),return_tensors="pt").input_ids[0]
        plen=tok(chatml(Q(sid)),return_tensors="pt").input_ids.shape[1]
        lab=ids.clone(); lab[:plen]=-100
        data.append((ids,lab))
    params=[p for b in blocks for p in b.parameters()]+[pool]
    opt=torch.optim.AdamW(params,lr=a.lr)
    for s in range(1,a.steps+1):
        batch=rng.sample(data,min(a.bs,len(data)))
        maxlen=max(x[0].shape[0] for x in batch)
        X=torch.full((len(batch),maxlen),tok.eos_token_id,dtype=torch.long)
        L=torch.full((len(batch),maxlen),-100,dtype=torch.long)
        for bi,(ids,lab) in enumerate(batch):
            X[bi,:ids.shape[0]]=ids; L[bi,:lab.shape[0]]=lab
        X=X.to(DEV); L=L.to(DEV)
        out=model(X).logits
        loss=F.cross_entropy(out[:,:-1].reshape(-1,out.size(-1)).float(), L[:,1:].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss): print(f"step {s} loss non-finie",flush=True); opt.zero_grad(); continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        if s%50==0 or s==1: print(f"step {s} loss {loss.item():.4f}",flush=True)
    model.eval()
    ev = facts if a.eval_n<=0 else facts[:a.eval_n]
    ok=0; samples=[]
    with torch.no_grad():
        for j,(sid,hx) in enumerate(ev):
            ids=tok(chatml(Q(sid)),return_tensors="pt").input_ids.to(DEV)
            o=model.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
            gen=tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True)
            if hx.lower() in gen.lower().replace(" ",""): ok+=1
            if j<5: samples.append((sid,hx,gen))
    print(f"\nRAPPEL (center={a.center}, {len(ev)} faits evalues sur {len(facts)}): {ok}/{len(ev)} = {100*ok/len(ev):.1f}%",flush=True)
    for sid,hx,gen in samples: print(f"   {sid} gold {hx} | gen {gen!r}",flush=True)
    known=[("What is the capital of France?","paris"),("What is the chemical symbol of gold?","au"),("What is the capital of Japan?","tokyo")]
    kok=0
    with torch.no_grad():
        for q,g in known:
            ids=tok(chatml(q),return_tensors="pt").input_ids.to(DEV)
            o=model.generate(ids,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
            if g in tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).lower(): kok+=1
    print(f"faits connus preserves: {kok}/{len(known)}",flush=True)
    print("MICROFIT_DONE",flush=True)
main()
