# -*- coding: utf-8 -*-
"""Relevance gate for retrofitting parametric memory onto a FROZEN LLM.

Generic mechanism (no project-specific content). A small per-token MLP on the hidden
state decides whether the memory output should be added to the residual stream:

    out = frozen_mlp(x) + sigmoid(gate(x)) * memory(x)

The backbone AND the memory stay frozen; only the gate trains. The gate learns to OPEN
on stored-fact contexts and CLOSE on general text and general factual questions, which
removes the perplexity / general-knowledge tax of an always-on memory while keeping
stored-fact recall intact.

~0.5M parameters per memory layer (d=3584, hidden=128). Trained with a per-layer
class-balanced binary cross-entropy on hidden states collected from labelled contexts.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class RelevanceGate(nn.Module):
    """MLP(d -> hidden -> 1) on the hidden state, returning a per-token logit."""
    def __init__(self, d, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                       # x: [..., d] (float)
        return self.f(x)                        # logit; sigmoid applied by the caller


class GatedMemoryMLP(nn.Module):
    """Wraps a frozen MLP + a (frozen) memory block + a trainable relevance gate.

    Flags:
      mem_off=True   -> backbone only (reference forward)
      gate_on=False  -> memory always added (ungated baseline)
    `last_x` caches the layer input so gate features can be collected without grads.
    """
    def __init__(self, orig_mlp, memory, gate):
        super().__init__()
        self.orig = orig_mlp
        self.mem = memory
        self.gate = gate
        self.mem_off = False
        self.gate_on = True
        self.last_x = None

    def forward(self, x):
        self.last_x = x.detach()
        if self.mem_off:
            return self.orig(x)
        m = self.mem(x)
        if self.gate_on:
            m = torch.sigmoid(self.gate(x.float())).to(m.dtype) * m
        return self.orig(x) + m


@torch.no_grad()
def collect_features(model, wrappers, tok, chatml, pos_pairs, neg_texts, device):
    """Forward pass (gate OFF, memory ON) to cache hidden states at the gated layers.

    Positives = stored-fact contexts (question+answer), labelled 1 over the whole fact
    context. Negatives = general prose + general factual questions, labelled 0.
    Returns per-layer (X, Y) tensors of stacked token hidden states and labels.
    """
    for w in wrappers:
        w.mem_off = False
        w.gate_on = False
    nL = len(wrappers)
    X = [[] for _ in range(nL)]
    Y = [[] for _ in range(nL)]
    for q, a in pos_pairs:
        ids = tok(chatml(q, a), return_tensors="pt").input_ids.to(device)
        model(ids)
        T = ids.shape[1]
        lab = torch.zeros(T); lab[:T - 1] = 1.0        # whole fact context = 1
        for li, w in enumerate(wrappers):
            X[li].append(w.last_x[0].float().cpu()); Y[li].append(lab.clone())
    for t in neg_texts:
        ids = tok(t, return_tensors="pt").input_ids.to(device)
        model(ids)
        T = ids.shape[1]
        for li, w in enumerate(wrappers):
            X[li].append(w.last_x[0].float().cpu()); Y[li].append(torch.zeros(T))
    return [torch.cat(x, 0) for x in X], [torch.cat(y, 0) for y in Y]


def train_gates(gates, X, Y, device, steps=1200, bs=4096, lr=2e-3, verbose=True):
    """Per-layer class-balanced BCE. Trains each gate in place on its layer's features."""
    for li, (g, x, y) in enumerate(zip(gates, X, Y)):
        opt = torch.optim.Adam(g.parameters(), lr=lr)
        N = x.shape[0]
        pw = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())]).to(device)
        for _ in range(steps):
            idx = torch.randint(0, N, (min(bs, N),))
            xb = x[idx].to(device); yb = y[idx].to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(g(xb).squeeze(-1), yb, pos_weight=pw)
            loss.backward(); opt.step()
        if verbose:
            with torch.no_grad():
                pred = (torch.sigmoid(g(x.to(device)).squeeze(-1)) > 0.5).float().cpu()
                rp = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
                rn = ((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum())
            print(f"  [gate L{li}] N={N} | recall_fact {float(rp) * 100:.1f}% | recall_general {float(rn) * 100:.1f}%", flush=True)
