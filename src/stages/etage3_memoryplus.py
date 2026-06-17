"""Phase D - Sprint etage 3 : Memory+ complet.
- qk-normalization (cosinus + temperature)  [valide Phase C]
- gating SiLU (branche valeur facon SwiGLU)
- 3 memory layers EN RESIDUEL partageant le MEME pool de valeurs V
- load-balance Switch (necessaire, cf Phase C)
Validation : overfit dataset synthetique UUID -> doit tenir un top-1 eleve + couverture saine
du pool partage.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, time

def make_dataset(n=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.arange(n)
    y = torch.randperm(n, generator=g)
    return x, y

class MemoryPlus(nn.Module):
    """Une couche Memory+ ; le pool V est partage (passe au forward)."""
    def __init__(self, d, num_keys, value_dim, topk=8, temp=20.0):
        super().__init__()
        self.topk, self.temp = topk, temp
        self.q_proj = nn.Linear(d, d)
        self.keys = nn.Parameter(torch.randn(num_keys, d) * (d ** -0.5))
        self.gate_proj = nn.Linear(d, value_dim)
        self.out_proj = nn.Linear(value_dim, d)
    def forward(self, x, V):
        q = self.q_proj(x)
        scores = (F.normalize(q, dim=-1) @ F.normalize(self.keys, dim=-1).t()) * self.temp
        topv, topi = scores.topk(self.topk, dim=-1)
        w = torch.softmax(topv, dim=-1)
        v = (w.unsqueeze(-1) * V[topi]).sum(1)          # lookup dans le pool partage
        out = self.out_proj(F.silu(self.gate_proj(x)) * v)   # gating SiLU (SwiGLU-like)
        return out, topi, scores

class MiniMemPlusModel(nn.Module):
    def __init__(self, N, d=128, num_keys=1024, value_dim=128, n_layers=3, topk=8, temp=20.0):
        super().__init__()
        self.emb = nn.Embedding(N, d)
        self.V = nn.Parameter(torch.randn(num_keys, value_dim) * (value_dim ** -0.5))  # POOL PARTAGE
        self.layers = nn.ModuleList([MemoryPlus(d, num_keys, value_dim, topk, temp) for _ in range(n_layers)])
        self.head = nn.Linear(d, N)
        self.num_keys = num_keys
    def forward(self, idx):
        x = self.emb(idx)
        all_scores, all_topi = [], []
        for layer in self.layers:
            out, topi, scores = layer(x, self.V)
            x = x + out                                  # residuel
            all_scores.append(scores); all_topi.append(topi)
        return self.head(x), all_topi, all_scores

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, NUM_KEYS = 500, 1024
    EPOCHS, LR, LB = 3000, 5e-3, 0.01
    x, y = make_dataset(N); x = x.to(dev); y = y.to(dev)
    model = MiniMemPlusModel(N, num_keys=NUM_KEYS).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    print(f"Memory+ : 3 layers, pool partage {NUM_KEYS}, gating SiLU, qk-norm, dev={dev}", flush=True)
    for ep in range(EPOCHS):
        opt.zero_grad()
        logits, all_topi, all_scores = model(x)
        loss = lossf(logits, y)
        for sc, ti in zip(all_scores, all_topi):           # load-balance par couche
            probs = torch.softmax(sc, dim=-1); me = probs.mean(0)
            mask = torch.zeros_like(sc).scatter_(1, ti, 1.0); ce = mask.mean(0)
            loss = loss + LB * sc.shape[1] * (me * ce).sum()
        loss.backward(); opt.step()
        if ep % 500 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
                allk = torch.cat([t.reshape(-1) for t in all_topi])
                cov = torch.unique(allk).numel() / NUM_KEYS
            print(f"  ep {ep:5d} loss {loss.item():.4f} top1 {acc*100:5.1f}% poolcov {cov*100:5.1f}%", flush=True)
    with torch.no_grad():
        logits, all_topi, _ = model(x)
        acc = (logits.argmax(-1) == y).float().mean().item()
        allk = torch.cat([t.reshape(-1) for t in all_topi])
        used = int(torch.unique(allk).numel()); cov = used / NUM_KEYS
    ok = acc >= 0.95 and cov >= 0.80
    print(f"\nMemory+ FINAL top1 {acc*100:.2f}% pool {used}/{NUM_KEYS} ({cov*100:.1f}%) {time.time()-t0:.1f}s -> {'GO' if ok else 'A_REVOIR'}", flush=True)
    json.dump({"top1":round(acc,4),"pool_coverage":round(cov,4),"keys_used":used,"num_keys":NUM_KEYS,
               "n_layers":3,"shared_pool":True,"gating":"silu","qk_norm":True,"ok":bool(ok),"device":dev},
              open("etage3_results.json","w"), indent=2)
    print("RESULTS_WRITTEN", flush=True)

if __name__ == "__main__":
    main()
