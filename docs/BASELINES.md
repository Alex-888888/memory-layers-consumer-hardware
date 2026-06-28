# External baselines (v0.3.2)

How does the frozen-backbone Memory-Layers-+-gate approach compare to standard ways of giving a
frozen LLM new facts? We run three external baselines on the same synthetic sensor facts and the
same metrics. **All data is synthetic; the backbone is Qwen2.5-7B.**

## Comparison

| approach | synthetic recall | TriviaQA (general) | WikiText-103 PPL | weights | per-query cost |
|---|---|---|---|---|---|
| **Memory Layers + relevance gate** | **~100 %** (5000) | 52.5–52.8 % (−0.6) | **+2.3 %** | backbone frozen, added params | none (parametric) |
| **RAG** (BM25 sparse) | 99.4 % (5000) | = backbone 53.4 % | = backbone (non-destructive) | unchanged | retrieval + injected context (~per-query) |
| **LoRA** (r=16, all-linear) | 82.7 % (1000) | **0.7 %** 💥 | **+45 %** | modified (low-rank Δ) | none |
| **kNN-LM** (λ=0.25 / 0.5) | **0 %** (1000) | 42 % / 8 % | +15 % / +70 % | unchanged | per-token kNN over datastore |

*(Scales differ where noted — Memory/RAG at 5000 facts, LoRA/kNN-LM at 1000; the qualitative
ordering is robust. Backbone references: TriviaQA 53.4 % at n=1000, WikiText-103 slice PPL 7.24.)*

## What each baseline shows

**RAG (retrieval-augmented, frozen weights).** Sparse BM25 over the fact declaratives, top-1
injected into the prompt. Near-perfect recall (99.4 %) and **non-destructive** by construction
(general PPL / TriviaQA are the backbone's). The cost is operational, not in competence: a
retrieval index and an injected context on **every** query (and the knowledge never enters the
weights). This is the strongest baseline and the honest point of comparison: Memory Layers trade
a small parametric tax for *no retrieval and no context bloat*.

**LoRA (low-rank weight edit).** Trained answer-only on the facts. It memorises them only
partially (82.7 % at 1000 facts) and, crucially, **catastrophically forgets**: TriviaQA collapses
53.4 % → 0.7 % and WikiText PPL rises +45 %. Editing the backbone's effective weights to inject
narrow synthetic facts destroys general competence — exactly the failure the frozen-backbone +
gate design avoids.

**kNN-LM (non-parametric interpolation).** Datastore of (hidden-state → next-token) pairs from the
fact declaratives; next-token distribution interpolated with the LM. It **fails the recall task
(0 %)**: the datastore keys come from declarative contexts (“…is HHHH”), but the test reformulates
as a question, so the question's hidden state does not match the declarative keys → wrong
neighbours → the value token is never boosted. Meanwhile the fact-only datastore **taxes general
competence**, worsening with λ (PPL +15 %→+70 %, TriviaQA 42 %→8 %). Weakest here, because naive
kNN-LM needs the test context to *continue* a stored context, which the Q&A format breaks.

## Takeaway

Only **Memory Layers + relevance gate** simultaneously (i) reaches ~100 % stored-fact recall,
(ii) preserves general competence (PPL +2.3 %, TriviaQA −0.6), and (iii) is **parametric** — no
retrieval index, no per-query context. RAG matches recall and preservation but pays a per-query
retrieval/context cost and keeps knowledge out of the weights; LoRA puts knowledge in the weights
but forgets catastrophically; kNN-LM (naive) does neither well here. This positions the
contribution as a **parametric, non-destructive** alternative for injecting a closed set of facts
into a frozen model.

*Memorizing Transformers (Wu et al. 2022) — kNN-augmented attention — is discussed in Related
Work rather than reimplemented (a faithful integration into this stack is out of scope for this
report).*
