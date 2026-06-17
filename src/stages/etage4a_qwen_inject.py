"""Phase D - Sprint etage 4a : integration des Memory Layers dans Qwen2.5-7B.
- Archi reelle Qwen2.5-7B instanciee via from_config (poids aleatoires, pas de download).
- Remplacement du MLP par un bloc Memory+ product-key aux couches 6/14/22 (regle centre stride 8).
- Backbone GELE (warm-up : seules les memory layers s'entrainent).
- Smoke test : forward OK, comptage params (gele vs entrainable), VRAM.
Pas d'entrainement ici (warm-up = etage 4b, long).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, gc
torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()

class ProductKeyMemoryPlus(nn.Module):
    """Memory+ a clefs-produit. V (pool) est PARTAGE (passe a __init__, non re-enregistre)."""
    def __init__(self, d, m, value_dim, shared_V, topk=8, kp=8, temp=20.0, dtype=torch.bfloat16):
        super().__init__()
        self.m, self.topk, self.kp, self.temp = m, topk, kp, temp
        self.dh = d // 2
        self.q_proj = nn.Linear(d, d, bias=False, dtype=dtype)
        self.C1 = nn.Parameter(torch.randn(m, self.dh, dtype=dtype) * (self.dh ** -0.5))
        self.C2 = nn.Parameter(torch.randn(m, self.dh, dtype=dtype) * (self.dh ** -0.5))
        self.gate_proj = nn.Linear(d, value_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(value_dim, d, bias=False, dtype=dtype)
        self._V = [shared_V]   # reference partagee, masquee a .parameters() pour eviter doublons
    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(B * T, d)
        q = self.q_proj(xf)
        q1, q2 = q[:, :self.dh], q[:, self.dh:]
        s1 = F.normalize(q1, dim=-1) @ F.normalize(self.C1, dim=-1).t() * self.temp
        s2 = F.normalize(q2, dim=-1) @ F.normalize(self.C2, dim=-1).t() * self.temp
        v1, i1 = s1.topk(self.kp, dim=-1); v2, i2 = s2.topk(self.kp, dim=-1)
        cs = (v1[:, :, None] + v2[:, None, :]).reshape(B * T, self.kp * self.kp)
        ci = (i1[:, :, None] * self.m + i2[:, None, :]).reshape(B * T, self.kp * self.kp)
        tv, sel = cs.topk(self.topk, dim=-1)
        ti = torch.gather(ci, 1, sel)
        w = torch.softmax(tv, dim=-1).unsqueeze(-1)
        V = self._V[0]
        v = (w * V[ti]).sum(1)                       # (B*T, value_dim)
        out = self.out_proj(F.silu(self.gate_proj(xf)) * v)
        return out.reshape(B, T, d)

def main():
    dev = "cuda"
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    d = cfg.hidden_size
    print(f"Qwen2.5-7B : {cfg.num_hidden_layers} couches, hidden {d}", flush=True)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).to(dev)
    torch.cuda.empty_cache()
    n_backbone = sum(p.numel() for p in model.parameters())
    print(f"backbone params {n_backbone/1e9:.3f} B  VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # pool partage 1M x value_dim (value_dim = hidden = d, cf cadrage 3584)
    M, VALUE_DIM = 1024, d           # m=1024 -> m^2 ~ 1.05M cles-produit
    pool_V = nn.Parameter(torch.empty(M * M, VALUE_DIM, device=dev, dtype=torch.bfloat16).normal_(0.0, VALUE_DIM ** -0.5))
    print(f"pool partage {M*M} x {VALUE_DIM} = {pool_V.numel()*2/1e9:.2f} GB", flush=True)

    PLACEMENT = [6, 14, 22]
    blocks = []
    for i in PLACEMENT:
        blk = ProductKeyMemoryPlus(d, M, VALUE_DIM, pool_V).to(dev)
        model.model.layers[i].mlp = blk      # remplace le MLP (Memory Layers remplace FFN/MLP)
        blocks.append(blk)
    print(f"Memory+ injecte aux couches {PLACEMENT}", flush=True)

    # GEL backbone, entrainables = memory layers + pool
    for p in model.parameters():
        p.requires_grad = False
    mem_params = {id(pool_V): pool_V}
    for blk in blocks:
        for p in blk.parameters():
            p.requires_grad = True; mem_params[id(p)] = p
    pool_V.requires_grad = True
    n_train = sum(p.numel() for p in mem_params.values())
    n_total = n_backbone + n_train
    print(f"entrainable (memory+pool) {n_train/1e9:.3f} B / total {n_total/1e9:.3f} B ({100*n_train/n_total:.1f}%)", flush=True)

    torch.cuda.synchronize()
    vram_static = torch.cuda.memory_allocated()/1e9

    # smoke forward
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 32), device=dev)
    ok = False; shape = None
    try:
        with torch.no_grad():
            out = model(ids)
        shape = tuple(out.logits.shape)
        finite = torch.isfinite(out.logits).all().item()
        ok = (shape == (1, 32, cfg.vocab_size)) and finite
        print(f"smoke forward OK shape={shape} finite={finite}", flush=True)
        # smoke backward : pool GELE (son grad 7GB demande l'offload = etage 4b). On verifie
        # que le gradient circule jusqu'aux params des blocs memoire (C1 de la 1ere couche injectee).
        pool_V.requires_grad = False
        idss = torch.randint(0, cfg.vocab_size, (1, 16), device=dev)
        out2 = model(idss)
        loss = out2.logits.float().pow(2).mean()
        loss.backward()
        gC1 = blocks[0].C1.grad
        grad_ok = (gC1 is not None) and torch.isfinite(gC1).all().item() and (gC1.abs().sum() > 0)
        print(f"smoke backward : grad params memoire present={grad_ok} (pool gele -> offload etage 4b)", flush=True)
    except Exception as ex:
        print(f"smoke ERROR: {str(ex)[:200]}", flush=True)
        grad_ok = False
    peak = torch.cuda.max_memory_allocated()/1e9
    free, total = torch.cuda.mem_get_info()
    print(f"\nVRAM statique {vram_static:.2f} GB | peak {peak:.2f} GB | carte {total/1e9:.2f} GB | marge {total/1e9-peak:.2f} GB", flush=True)
    verdict = ok and grad_ok and peak < 24.0
    print(f"ETAGE 4a SMOKE : forward={ok} backward_grad={grad_ok} VRAM_ok={peak<24.0} -> {'GO' if verdict else 'A_REVOIR'}", flush=True)
    json.dump({"placement":PLACEMENT,"backbone_B":round(n_backbone/1e9,3),"trainable_B":round(n_train/1e9,3),
               "trainable_pct":round(100*n_train/n_total,2),"pool_gb":round(pool_V.numel()*2/1e9,2),
               "vram_static_gb":round(vram_static,2),"vram_peak_gb":round(peak,2),"forward_ok":bool(ok),
               "backward_grad_ok":bool(grad_ok),"verdict_go":bool(verdict)}, open("etage4a_results.json","w"), indent=2)
    print("RESULTS_WRITTEN", flush=True)

if __name__ == "__main__":
    main()
