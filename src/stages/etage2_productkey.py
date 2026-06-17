"""Phase D - Sprint etage 2 : Product-Key lookup (Lample et al. 2019).
Factorisation : score(i,j) = q1.C1[i] + q2.C2[j] sur N = m^2 cles-produit.
Recherche en O(sqrt N) via top-k' par moitie, puis top-k sur le produit cartesien.

NON-REGRESSION FORMELLE : KL(softmax_exhaustif || softmax_productkey) = 0 a epsilon
machine pres sur l'ensemble-produit materialise, quand k' >= k (lemme Lample :
top-k(somme) inclus dans top-k'(s1) x top-k'(s2)).
"""
import torch, torch.nn.functional as F, json, time

def exhaustive_topk(q1, q2, C1, C2, k):
    # score complet (B, m, m) = s1[:,:,None] + s2[:,None,:] -> (B, m^2)
    s1 = q1 @ C1.t()                      # (B,m)
    s2 = q2 @ C2.t()                      # (B,m)
    B, m = s1.shape
    full = (s1[:, :, None] + s2[:, None, :]).reshape(B, m * m)  # ensemble-produit materialise
    topv, topi = full.topk(k, dim=-1)     # indices plats dans [0, m^2)
    return topv, topi, full

def productkey_topk(q1, q2, C1, C2, k, kp):
    s1 = q1 @ C1.t(); s2 = q2 @ C2.t()    # (B,m) each
    B, m = s1.shape
    v1, i1 = s1.topk(kp, dim=-1)          # (B,kp)
    v2, i2 = s2.topk(kp, dim=-1)          # (B,kp)
    # candidats = produit cartesien kp x kp
    cand_score = v1[:, :, None] + v2[:, None, :]       # (B,kp,kp)
    cand_flat_idx = i1[:, :, None] * m + i2[:, None, :]  # indices plats (B,kp,kp)
    cand_score = cand_score.reshape(B, kp * kp)
    cand_flat_idx = cand_flat_idx.reshape(B, kp * kp)
    topv, sel = cand_score.topk(k, dim=-1)             # top-k parmi candidats
    topi = torch.gather(cand_flat_idx, 1, sel)         # indices plats correspondants
    return topv, topi

def kl_test(B=256, m=1024, dh=128, k=8, kp=8, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    q1 = torch.randn(B, dh, generator=g, device=dev)
    q2 = torch.randn(B, dh, generator=g, device=dev)
    C1 = torch.randn(m, dh, generator=g, device=dev)
    C2 = torch.randn(m, dh, generator=g, device=dev)
    # exhaustif
    ev, ei, full = exhaustive_topk(q1, q2, C1, C2, k)
    ew = torch.softmax(ev, dim=-1)
    # product-key
    pv, pi = productkey_topk(q1, q2, C1, C2, k, kp)
    pw = torch.softmax(pv, dim=-1)
    # alignement : pour chaque query, comparer ensembles d'indices + poids
    ei_s, _ = ei.sort(dim=-1); pi_s, _ = pi.sort(dim=-1)
    set_equal = (ei_s == pi_s).all(dim=-1)             # (B,)
    exact_rate = set_equal.float().mean().item()
    # KL sur les distributions reconstruites sur l'union (ici, si sets egaux, support identique)
    # KL(p||q) = sum p log(p/q) en reordonnant pw selon l'ordre de ei
    # reconstruire q-weight aligne sur ei : on mappe chaque index de ei vers son poids dans pi
    kls = []
    max_out_l2 = 0.0
    # valeurs aleatoires pour comparer les sorties memoire
    V = torch.randn(m * m, 32, generator=g, device=dev)
    out_e = (ew.unsqueeze(-1) * V[ei]).sum(1)
    out_p = (pw.unsqueeze(-1) * V[pi]).sum(1)
    out_l2 = (out_e - out_p).norm(dim=-1)
    max_out_l2 = out_l2.max().item()
    mean_out_l2 = out_l2.mean().item()
    # KL exacte la ou les ensembles sont egaux
    eq = set_equal
    if eq.any():
        # aligner : trier les deux par indice et comparer poids tries par indice
        ew_byidx = torch.gather(ew, 1, ei.argsort(dim=-1))
        pw_byidx = torch.gather(pw, 1, pi.argsort(dim=-1))
        p = ew_byidx[eq].clamp_min(1e-12); qd = pw_byidx[eq].clamp_min(1e-12)
        kl = (p * (p / qd).log()).sum(-1)
        kl_max = kl.max().item(); kl_mean = kl.mean().item()
    else:
        kl_max = float("nan"); kl_mean = float("nan")
    return {"B":B,"m":m,"N":m*m,"k":k,"kp":kp,"exact_set_rate":round(exact_rate,4),
            "KL_max":kl_max,"KL_mean":kl_mean,"out_l2_max":max_out_l2,"out_l2_mean":mean_out_l2}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, "torch", torch.__version__, flush=True)
    results = []
    # non-regression principale : k'=k -> doit etre exact (KL=0)
    for kp in (8, 16):   # k'>=k
        r = kl_test(k=8, kp=kp, dev=dev); results.append(r)
        print(f"k'={kp:>2} (>=k): exact={r['exact_set_rate']*100:.1f}%  KL_max={r['KL_max']:.2e}  out_l2_max={r['out_l2_max']:.2e}", flush=True)
    # ablation : k'<k -> deviation attendue
    for kp in (4, 2):
        if kp * kp < 8:
            print(f"k'={kp} skip (kp^2 < k)", flush=True); continue
        r = kl_test(k=8, kp=kp, dev=dev); results.append(r)
        print(f"k'={kp:>2} (<k) : exact={r['exact_set_rate']*100:.1f}%  KL_max={r['KL_max']:.2e}  out_l2_max={r['out_l2_max']:.2e}", flush=True)
    main_r = results[0]  # k'=k=8
    ok = main_r["exact_set_rate"] >= 0.999 and (main_r["KL_max"] < 1e-6) and (main_r["out_l2_max"] < 1e-4)
    print(f"\nNON-REGRESSION (k'=k=8) : exact={main_r['exact_set_rate']*100:.2f}% KL_max={main_r['KL_max']:.2e} -> {'GO (KL=0)' if ok else 'ECHEC'}", flush=True)
    json.dump({"results":results,"non_regression_ok":bool(ok),"device":dev}, open("etage2_results.json","w"), indent=2)
    print("RESULTS_WRITTEN", flush=True)

if __name__ == "__main__":
    main()
