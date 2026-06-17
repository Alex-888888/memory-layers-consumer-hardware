"""Micro-benchmark : Adam offload DENSE (actuel) vs SPARSE (ne maj que les rangees touchees).
Le grad du pool est sparse (~quelques milliers de rangees / 150k). Sans charger Qwen.
"""
import torch, math, time
N, D = 387*387, 3584          # pool 150k
TOUCHED = 3000                # rangees touchees/step (3 couches x ~1000)
dev = "cuda"
pool = torch.randn(N, D, device=dev, dtype=torch.bfloat16) * (D**-0.5)

# grad sparse simule : ~TOUCHED rangees non nulles
g_full = torch.zeros(N, D, device=dev, dtype=torch.bfloat16)
rows = torch.randperm(N, device=dev)[:TOUCHED]
g_full[rows] = torch.randn(TOUCHED, D, device=dev, dtype=torch.bfloat16) * 0.01

b1, b2, eps, lr = 0.9, 0.999, 1e-8, 5e-4

# --- DENSE (actuel) : master/m/v CPU fp32, maj sur TOUT ---
masterD = pool.detach().cpu().float(); mD = torch.zeros_like(masterD); vD = torch.zeros_like(masterD)
torch.cuda.synchronize(); t=time.time()
for _ in range(5):
    g = g_full.detach().cpu().float()
    mD.mul_(b1).add_(g, alpha=1-b1); vD.mul_(b2).addcmul_(g,g,value=1-b2)
    bc1=1-b1; bc2=1-b2
    denom=(vD.sqrt()/math.sqrt(bc2)).add_(eps)
    masterD.addcdiv_(mD/bc1, denom, value=-lr)
    pool.data.copy_(masterD.to(dev, torch.bfloat16))
torch.cuda.synchronize(); dense_ms=(time.time()-t)/5*1000
print(f"DENSE  : {dense_ms:.0f} ms/step", flush=True)

# --- SPARSE : ne maj que les rangees touchees ---
masterS = pool.detach().cpu().float(); mS = torch.zeros_like(masterS); vS = torch.zeros_like(masterS)
torch.cuda.synchronize(); t=time.time()
for _ in range(5):
    nz = (g_full != 0).any(dim=1).nonzero(as_tuple=True)[0]   # GPU : rangees touchees
    g = g_full[nz].detach().cpu().float()                     # seulement elles -> CPU
    nzc = nz.cpu()
    mnz=mS[nzc]; vnz=vS[nzc]
    mnz.mul_(b1).add_(g, alpha=1-b1); vnz.mul_(b2).addcmul_(g,g,value=1-b2)
    bc1=1-b1; bc2=1-b2
    denom=(vnz.sqrt()/math.sqrt(bc2)).add_(eps)
    upd = (mnz/bc1)/denom * (-lr)
    masterS[nzc] += upd; mS[nzc]=mnz; vS[nzc]=vnz
    pool.data[nz] = masterS[nzc].to(dev, torch.bfloat16)
torch.cuda.synchronize(); sparse_ms=(time.time()-t)/5*1000
print(f"SPARSE : {sparse_ms:.0f} ms/step  (nz={int((g_full!=0).any(1).sum())})", flush=True)
print(f"SPEEDUP x{dense_ms/sparse_ms:.0f}", flush=True)

# correctness : meme resultat sur les rangees touchees (1 step depuis le meme etat)
m1=torch.zeros_like(masterD); v1=torch.zeros_like(masterD); ms=pool.detach().cpu().float()*0  # placeholder
# re-deriver proprement : 1 step dense vs sparse depuis master commun
base=torch.randn(N,D).float()*0.01
gd=g_full.detach().cpu().float()
# dense 1 step
md=torch.zeros_like(base); vd=torch.zeros_like(base); Md=base.clone()
md.mul_(b1).add_(gd,alpha=1-b1); vd.mul_(b2).addcmul_(gd,gd,value=1-b2)
Md.addcdiv_(md/(1-b1),(vd.sqrt()/math.sqrt(1-b2)).add_(eps),value=-lr)
# sparse 1 step
nz=(g_full!=0).any(1).nonzero(as_tuple=True)[0].cpu()
Ms=base.clone(); ms2=torch.zeros_like(base); vs2=torch.zeros_like(base)
gs=gd[nz]; mn=ms2[nz]; vn=vs2[nz]
mn.mul_(b1).add_(gs,alpha=1-b1); vn.mul_(b2).addcmul_(gs,gs,value=1-b2)
Ms[nz]+= (mn/(1-b1))/((vn.sqrt()/math.sqrt(1-b2)).add_(eps))*(-lr)
print(f"correctness max|dense-sparse| = {(Md-Ms).abs().max().item():.2e} (doit etre ~0)", flush=True)
print("BENCH_DONE", flush=True)
