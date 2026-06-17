"""Phase D - Sprint etage 4b : WARM-UP des Memory Layers dans Qwen2.5-7B (PRET A LANCER).
Backbone GELE, seules les memory layers (pool + blocs product-key) s'entrainent.
Offload CPU MANUEL de l'optimiseur du pool (equivalent ZeRO-Offload, sans DeepSpeed
qui est absent/risque sur ROCm). Gradient checkpointing + micro-batch 1.
Checkpoints atomiques + reprise idempotente (resilience decrochage GPU WSL2).

NE PAS confondre avec un run court : c'est l'entrainement long (jours). Lancer via _run.sh.
Telechargement Qwen2.5-7B-Instruct (~15 Go) automatique au 1er run (from_pretrained).
"""
import os, json, time, argparse, math
import torch, torch.nn as nn, torch.nn.functional as F

PLACEMENT = [6, 14, 22]

# ----------------------------- Memory+ product-key -----------------------------
class ProductKeyMemoryPlus(nn.Module):
    def __init__(self, d, m, value_dim, shared_V, topk=8, kp=8, temp=20.0, dtype=torch.bfloat16):
        super().__init__()
        self.m, self.topk, self.kp, self.temp, self.dh = m, topk, kp, temp, d // 2
        self.q_proj = nn.Linear(d, d, bias=False, dtype=dtype)
        self.C1 = nn.Parameter(torch.randn(m, self.dh, dtype=dtype) * (self.dh ** -0.5))
        self.C2 = nn.Parameter(torch.randn(m, self.dh, dtype=dtype) * (self.dh ** -0.5))
        self.gate_proj = nn.Linear(d, value_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(value_dim, d, bias=False, dtype=dtype)
        nn.init.zeros_(self.out_proj.weight)  # zero-init : la memoire demarre en no-op (stabilite residuel gele)
        self._V = [shared_V]
        self.dbg = False; self._dbg = None  # instrumentation diagnostique (off par defaut, aucun effet)
        # centrage de la query avant le matching cle-produit (retrait composante partagee).
        # off par defaut (aucun effet sur l'existant). Buffers EMA pour l'inference token-par-token.
        self.center = False; self.ema_m = 0.9
        self.register_buffer("q1_mu", torch.zeros(self.dh, dtype=torch.float32), persistent=True)
        self.register_buffer("q2_mu", torch.zeros(self.dh, dtype=torch.float32), persistent=True)
    def forward(self, x):
        # FORWARD EN FP32 (le bloc memoire en bf16 overflow apres maj -> NaN ne lm_head, cf diag 05/06).
        # Poids castes a la volee ; cout VRAM negligeable (seq courte x 3 couches). Sortie re-castee bf16.
        B, T, d = x.shape
        xf = x.reshape(B * T, d).float()
        q = F.linear(xf, self.q_proj.weight.float())
        q1, q2 = q[:, :self.dh], q[:, self.dh:]
        if getattr(self, "center", False):
            if self.training:
                bm1 = q1.mean(0); bm2 = q2.mean(0)            # composante partagee du batch
                with torch.no_grad():
                    self.q1_mu.mul_(self.ema_m).add_(bm1.detach(), alpha=1 - self.ema_m)
                    self.q2_mu.mul_(self.ema_m).add_(bm2.detach(), alpha=1 - self.ema_m)
                q1 = q1 - bm1; q2 = q2 - bm2                  # centrage (differentiable, facon BN)
            else:
                q1 = q1 - self.q1_mu.to(q1.dtype); q2 = q2 - self.q2_mu.to(q2.dtype)  # EMA en inference
        s1 = F.normalize(q1, dim=-1, eps=1e-4) @ F.normalize(self.C1.float(), dim=-1, eps=1e-4).t() * self.temp
        s2 = F.normalize(q2, dim=-1, eps=1e-4) @ F.normalize(self.C2.float(), dim=-1, eps=1e-4).t() * self.temp
        v1, i1 = s1.topk(self.kp, dim=-1); v2, i2 = s2.topk(self.kp, dim=-1)
        cs = (v1[:, :, None] + v2[:, None, :]).reshape(B * T, self.kp * self.kp)
        ci = (i1[:, :, None] * self.m + i2[:, None, :]).reshape(B * T, self.kp * self.kp)
        tv, sel = cs.topk(self.topk, dim=-1); ti = torch.gather(ci, 1, sel)
        w = torch.softmax(tv, dim=-1).unsqueeze(-1)
        v = (w * self._V[0][ti].float()).sum(1)
        gate = F.silu(F.linear(xf, self.gate_proj.weight.float()))
        out = F.linear(gate * v, self.out_proj.weight.float())
        if getattr(self, "dbg", False):
            self._dbg = {"w": w.detach().squeeze(-1).cpu(), "ti": ti.detach().cpu(),
                         "mem_norm": out.detach().norm(dim=-1).cpu(),
                         "q1n": F.normalize(q1, dim=-1, eps=1e-4).detach().cpu(),
                         "q2n": F.normalize(q2, dim=-1, eps=1e-4).detach().cpu()}
        return out.reshape(B, T, d).to(x.dtype)

class MLPPlusMemory(nn.Module):
    """MLP-ADD : conserve le MLP original (gele) et ajoute la sortie memoire (zero-init).
    Backbone NON ampute -> stable numeriquement ; la memoire apprend par-dessus."""
    def __init__(self, orig_mlp, mem):
        super().__init__()
        self.orig_mlp = orig_mlp
        self.mem = mem
        self.mem_off = False   # True -> sortie = MLP gele seul (forward de reference pour l'ancre KL)
    def forward(self, x):
        if self.mem_off:
            return self.orig_mlp(x)
        return self.orig_mlp(x) + self.mem(x)


# ----------------------------- Offload CPU Adam (pool) -----------------------------
class CPUOffloadAdam:
    """Adam pour UN gros parametre GPU ; master fp32 + (m,v) sur CPU. Le grad transite
    par le CPU au step. Reduit drastiquement la VRAM (equivalent ZeRO-Offload stage 2/3)."""
    def __init__(self, p, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0, state_dtype=torch.float32):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd = p, lr, betas[0], betas[1], eps, wd
        self.master = p.detach().cpu().float()  # .cpu() avant .float() : evite la copie fp32 sur GPU
        self.m = torch.zeros_like(self.master, dtype=state_dtype)
        self.v = torch.zeros_like(self.master, dtype=state_dtype)
        self.t = 0
    @torch.no_grad()
    def step(self):
        if self.p.grad is None: return
        # SPARSE : le grad du pool est creux (seules les rangees gatherees sont != 0).
        # On ne met a jour QUE ces rangees -> ~16x plus rapide que le dense (cf bench_optim).
        # (correction de biais a t global = approximation acceptable pour le warm-up.)
        nz = (self.p.grad != 0).any(dim=1).nonzero(as_tuple=True)[0]
        if nz.numel() == 0: return
        g = self.p.grad[nz].detach().cpu().float()
        nzc = nz.cpu()
        self.t += 1
        mnz = self.m[nzc].float(); vnz = self.v[nzc].float()
        mnz.mul_(self.b1).add_(g, alpha=1 - self.b1)
        vnz.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        bc1 = 1 - self.b1 ** self.t; bc2 = 1 - self.b2 ** self.t
        denom = (vnz.sqrt() / math.sqrt(bc2)).add_(self.eps)
        masternz = self.master[nzc] * ((1 - self.lr * self.wd) if self.wd else 1.0)
        masternz = masternz + (mnz / bc1) / denom * (-self.lr)
        self.m[nzc] = mnz.to(self.m.dtype); self.v[nzc] = vnz.to(self.v.dtype); self.master[nzc] = masternz
        self.p.data[nz] = masternz.to(self.p.device, self.p.dtype)
    def zero_grad(self): self.p.grad = None
    def state_dict(self): return {"master": self.master, "m": self.m, "v": self.v, "t": self.t}
    def load_state_dict(self, sd):
        self.master, self.m, self.v, self.t = sd["master"], sd["m"], sd["v"], sd["t"]
        self.p.data.copy_(self.master.to(self.p.device, self.p.dtype))

# ----------------------------- checkpoint atomique -----------------------------
def save_ckpt(path, step, blocks, pool_V, opt_blocks, opt_pool):
    tmp = path + ".tmp"
    torch.save({"step": step,
                "blocks": [b.state_dict() for b in blocks],
                "pool": pool_V.detach().cpu(),
                "opt_blocks": opt_blocks.state_dict(),
                "opt_pool": opt_pool.state_dict()}, tmp)
    os.replace(tmp, path)
    import glob as _glob
    cks = sorted(_glob.glob(os.path.join(os.path.dirname(path), 'warmup_step*.pt')), key=lambda fp: int(os.path.basename(fp)[len('warmup_step'):-3]))
    for _old in cks[:-3]:
        try: os.remove(_old)
        except Exception: pass

def find_latest(ckpt_dir):
    if not os.path.isdir(ckpt_dir): return None
    cks = [f for f in os.listdir(ckpt_dir) if f.startswith("warmup_step") and f.endswith(".pt")]
    if not cks: return None
    return os.path.join(ckpt_dir, max(cks, key=lambda f: int(f[len("warmup_step"):-3])))

# ----------------------------- corpus -----------------------------
def get_batches(tokenizer, mode, seq, n_batches, device, corpus_path=None, batch=1, answer_only=False, pack=True):
    """Rend un iterateur de (input_ids, labels). 'synthetic' = sequences factices auto-suffisantes
    pour valider le pipeline ; 'textfile' = corpus reel tokenise.
    answer_only=True : la loss n'est calculee que sur les tokens-reponse des tours assistant ChatML
    (le reste des cibles = -100, ignore par cross_entropy). Levier prouve Phase 3c/3d."""
    vocab = tokenizer.vocab_size
    if mode == "textfile" and corpus_path:
        text = open(corpus_path, encoding="utf-8").read()
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        nb = (len(ids) - 1) // seq
        for i in range(min(nb, n_batches)):
            chunk = ids[i * seq:(i + 1) * seq + 1]
            x = chunk[:-1].unsqueeze(0).to(device); y = chunk[1:].unsqueeze(0).to(device)
            yield x, y
    elif mode == "jsonl" and corpus_path:
        import json as _json, random as _rnd
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = tokenizer("<|endoftext|>", add_special_tokens=False).input_ids[-1]
        imend = tokenizer.convert_tokens_to_ids("<|im_end|>")
        marker = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        Lm = len(marker)
        def amask(ids):
            # 1 sur les tokens de la reponse assistant (apres "<|im_start|>assistant\n" jusqu'au <|im_end|> inclus)
            m = [0] * len(ids); i = 0
            while i <= len(ids) - Lm:
                if ids[i:i+Lm] == marker:
                    j = i + Lm
                    while j < len(ids) and ids[j] != imend: m[j] = 1; j += 1
                    if j < len(ids): m[j] = 1; j += 1
                    i = j
                else:
                    i += 1
            return m
        stream = []; lmask = []; records = []
        for line in open(corpus_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
            except Exception:
                continue
            ids = tokenizer(d["text"], add_special_tokens=False).input_ids
            mk = amask(ids) if answer_only else [1] * len(ids)
            records.append((ids, mk))
            stream.extend(ids); lmask.extend(mk)
            stream.append(eos); lmask.append(0 if answer_only else 1)
        if not pack:
            # UNE SEQUENCE PAR DOC (recette micro-overfit : pas de raccourci copie inter-faits)
            # on garde TOUS les docs (fluence/declaratifs servent au signal KL ; CE neutre si pas de reponse)
            recs = records
            order = list(range(len(recs))); rng = _rnd.Random(0); emitted = 0
            while emitted < n_batches:
                rng.shuffle(order)
                for i in range(0, len(order) - batch + 1, batch):
                    if emitted >= n_batches: break
                    seqs = []; labs = []
                    for di in order[i:i+batch]:
                        ids, mk = recs[di]; ids = ids[:seq+1]; mk = mk[:seq+1]
                        xx = ids[:-1]; yy = ids[1:]
                        if answer_only:
                            yy = [t if mm == 1 else -100 for t, mm in zip(yy, mk[1:])]
                        seqs.append(xx); labs.append(yy)
                    L = max(len(s) for s in seqs)                    # pad au max du batch (docs courts -> peu de compute)
                    X = [s + [eos] * (L - len(s)) for s in seqs]
                    Y = [yy + [-100] * (L - len(yy)) for yy in labs]
                    yield torch.tensor(X).to(device), torch.tensor(Y).to(device)
                    emitted += 1
            return
        nb = (len(stream) - 1) // seq
        order = list(range(nb)); rng = _rnd.Random(0)
        emitted = 0
        # multi-epoques : on reboucle sur le corpus (reshuffle a chaque epoque)
        # jusqu'a avoir emis n_batches batches au total.
        while emitted < n_batches:
            rng.shuffle(order)
            groups = [order[i:i+batch] for i in range(0, len(order), batch)]
            for grp in groups:
                if emitted >= n_batches: break
                xs, ys = [], []
                for bi in grp:
                    chunk = stream[bi * seq: bi * seq + seq + 1]
                    if len(chunk) < seq + 1: continue
                    yy = chunk[1:]
                    if answer_only:
                        mch = lmask[bi * seq + 1: bi * seq + seq + 1]   # aligne avec chunk[1:]
                        yy = [t if mm == 1 else -100 for t, mm in zip(yy, mch)]
                    xs.append(chunk[:-1]); ys.append(yy)
                if not xs: continue
                yield torch.tensor(xs).to(device), torch.tensor(ys).to(device)
                emitted += 1
    else:
        g = torch.Generator().manual_seed(0)
        for _ in range(n_batches):
            ids = torch.randint(0, vocab, (1, seq + 1), generator=g)
            yield ids[:, :-1].to(device), ids[:, 1:].to(device)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pool_size", type=int, default=500_000, help="nb de valeurs memoire (500k = fallback feasible)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--ckpt_dir", default="./checkpoints")
    ap.add_argument("--corpus", default="synthetic", choices=["synthetic", "textfile", "jsonl"])
    ap.add_argument("--corpus_path", default=None)
    ap.add_argument("--state_dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--grad_ckpt", type=int, default=1, help="0 = desactive gradient checkpointing")
    ap.add_argument("--warmup_steps", type=int, default=40, help="ramp-up lineaire du LR")
    ap.add_argument("--batch", type=int, default=1, help="micro-batch (sequences par step)")
    ap.add_argument("--answer_only", type=int, default=0, help="1 = loss seulement sur les tokens-reponse ChatML (Phase 3e)")
    ap.add_argument("--pool_optim", default="offload", choices=["offload", "dense"], help="dense = AdamW GPU sur le pool (recette micro-overfit, pool <=~100k)")
    ap.add_argument("--pack", type=int, default=1, help="1 = packing multi-docs/fenetre ; 0 = une sequence par doc (recette micro-overfit)")
    ap.add_argument("--kl_weight", type=float, default=0.0, help=">0 = ancre KL (preserve knowledge native + PPL) : KL(entraine||gele) sur les tokens de contexte")
    ap.add_argument("--kl_every", type=int, default=1, help="ancre KL calculee 1 step sur N (amortit le forward de reference)")
    ap.add_argument("--dry_run", type=int, default=0, help="1 = 3 steps de validation puis stop")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.ckpt_dir, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[warmup] chargement {a.model} (telechargement ~15 Go au 1er run)...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    torch.cuda.empty_cache()
    d = model.config.hidden_size
    m = int(round(a.pool_size ** 0.5)); pool_n = m * m
    VALUE_DIM = d
    print(f"[warmup] hidden={d} pool_size~{pool_n} (m={m}) value_dim={VALUE_DIM}", flush=True)
    pool_V = nn.Parameter(torch.empty(pool_n, VALUE_DIM, device=dev, dtype=torch.bfloat16).normal_(0, VALUE_DIM ** -0.5))
    blocks = []
    for i in PLACEMENT:
        blk = ProductKeyMemoryPlus(d, m, VALUE_DIM, pool_V).to(dev)
        orig_mlp = model.model.layers[i].mlp           # MLP original conserve (sera gele)
        model.model.layers[i].mlp = MLPPlusMemory(orig_mlp, blk)   # MLP-ADD: MLP + memoire(zero-init)
        blocks.append(blk)
    for p in model.parameters(): p.requires_grad = False
    for blk in blocks:
        for p in blk.parameters(): p.requires_grad = True
    pool_V.requires_grad = True
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    mlp_wrappers = [model.model.layers[i].mlp for i in PLACEMENT]   # pour toggler mem_off (ancre KL)
    block_params = [p for b in blocks for p in b.parameters()]
    opt_blocks = torch.optim.AdamW(block_params, lr=a.lr)
    sdt = torch.float32 if a.state_dtype == "fp32" else torch.bfloat16
    if a.pool_optim == "dense":
        opt_pool = torch.optim.AdamW([pool_V], lr=a.lr)   # AdamW GPU dense (recette micro-overfit)
        print(f"[warmup] pool optim = AdamW DENSE GPU (etats ~{pool_V.numel()*4*2/1e9:.1f} Go VRAM) | pack={a.pack}", flush=True)
    else:
        opt_pool = CPUOffloadAdam(pool_V, lr=a.lr, state_dtype=sdt)
        cpu_gb = (pool_V.numel() * 4 + pool_V.numel() * (4 if sdt == torch.float32 else 2) * 2) / 1e9
        print(f"[warmup] offload optim pool sur CPU ~{cpu_gb:.1f} Go (RAM 31 Go) | pack={a.pack}", flush=True)

    start = 0
    latest = find_latest(a.ckpt_dir)
    if latest:
        print(f"[warmup] reprise depuis {latest}", flush=True)
        ck = torch.load(latest, map_location="cpu")
        for b, sd in zip(blocks, ck["blocks"]): b.load_state_dict(sd)
        pool_V.data.copy_(ck["pool"].to(dev, torch.bfloat16))
        opt_blocks.load_state_dict(ck["opt_blocks"]); opt_pool.load_state_dict(ck["opt_pool"])
        start = ck["step"]
    # --steps = nombre de steps CIBLE (total, multi-epoques). Le resume reprend a 'start'
    # et ne fait que les steps restants -> reprise propre apres reboot/hoquet.
    remaining = 3 if a.dry_run else max(0, a.steps - start)
    target = start + 3 if a.dry_run else a.steps
    print(f"[warmup] start step={start} -> {target} ({remaining} steps a faire, multi-epoques)", flush=True)

    data = get_batches(tok, a.corpus, a.seq, remaining, dev, a.corpus_path, a.batch, bool(a.answer_only), bool(a.pack))
    t_ck = time.time()
    s = start
    for s, (x, y) in enumerate(data, start=start + 1):
        cur_lr = a.lr * min(1.0, (s - start) / max(1, a.warmup_steps))   # warm-up LR lineaire
        for gg in opt_blocks.param_groups: gg["lr"] = cur_lr
        if a.pool_optim == "dense":
            for gg in opt_pool.param_groups: gg["lr"] = cur_lr
        else:
            opt_pool.lr = cur_lr
        do_kl = (a.kl_weight > 0) and (s % a.kl_every == 0)        # ancre KL amortie (1 step / kl_every)
        ref_logits = None
        if do_kl:
            with torch.no_grad():
                for w in mlp_wrappers: w.mem_off = True
                ref_logits = model(x).logits.float()
                for w in mlp_wrappers: w.mem_off = False
        out = model(x)
        logits = out.logits.float(); V = logits.size(-1)
        yflat = y.reshape(-1)
        if (yflat != -100).any():
            loss = F.cross_entropy(logits.reshape(-1, V), yflat)   # answer-only (y=-100 hors reponse)
        else:
            loss = logits.sum() * 0.0                              # batch sans reponse : CE neutre
        if do_kl:
            ctx = (yflat == -100)                                  # tokens de contexte -> preserver la distri gelee
            if ctx.any():
                lp = F.log_softmax(logits.reshape(-1, V)[ctx], dim=-1)
                rp = F.softmax(ref_logits.reshape(-1, V)[ctx], dim=-1)
                loss = loss + a.kl_weight * F.kl_div(lp, rp, reduction="batchmean")
        if not torch.isfinite(loss):
            print(f"  step {s} loss non-finie -> step saute", flush=True)
            opt_blocks.zero_grad(); opt_pool.zero_grad(); continue
        opt_blocks.zero_grad(); opt_pool.zero_grad()
        loss.backward()
        # garde GRADIENT (le diagnostic montre que l'instabilite vient du backward, pas du forward) :
        # si un gradient est non-fini, on saute le step AVANT l'update -> jamais de corruption des params.
        # (clip_grad_norm_ propage les NaN : norme NaN -> tous les grads NaN. Donc on filtre d'abord.)
        _finite = all(torch.isfinite(p.grad).all().item() for p in block_params if p.grad is not None)
        if pool_V.grad is not None:
            _finite = _finite and bool(torch.isfinite(pool_V.grad).all().item())
        if not _finite:
            print(f"  step {s} gradient non-fini -> step saute (pas d'update)", flush=True)
            opt_blocks.zero_grad(); opt_pool.zero_grad(); continue
        torch.nn.utils.clip_grad_norm_(block_params, 1.0)
        if pool_V.grad is not None:
            torch.nn.utils.clip_grad_norm_([pool_V], 1.0)
        opt_blocks.step(); opt_pool.step()
        if s % 10 == 0 or a.dry_run:
            print(f"  step {s} loss {loss.item():.4f} lr {cur_lr:.2e} vram {torch.cuda.max_memory_allocated()/1e9:.1f}GB ts {time.strftime(chr(37)+chr(72)+chr(58)+chr(37)+chr(77)+chr(58)+chr(37)+chr(83))}", flush=True)
        if (s % a.ckpt_every == 0) or (time.time() - t_ck > 3000):  # par steps OU horaire (~50min)
            save_ckpt(os.path.join(a.ckpt_dir, f"warmup_step{s}.pt"), s, blocks, pool_V, opt_blocks, opt_pool)
            print(f"  [ckpt] sauve step {s}", flush=True); t_ck = time.time()
        if a.dry_run and s >= start + 3: break
    if not a.dry_run:
        save_ckpt(os.path.join(a.ckpt_dir, f"warmup_step{s}.pt"), s, blocks, pool_V, opt_blocks, opt_pool)
    print("WARMUP_DONE" if not a.dry_run else "DRYRUN_OK", flush=True)
    if not a.dry_run:
        open(os.path.join(a.ckpt_dir, "done.flag"), "w").close()

if __name__ == "__main__":
    main()
