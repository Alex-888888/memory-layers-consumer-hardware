# External baselines (v0.3.2, kNN-LM updated in v0.3.4)

How does the frozen-backbone Memory-Layers-+-gate approach compare to standard ways of giving a
frozen LLM new facts? We run three external baselines on the same synthetic sensor facts and the
same metrics. **All data is synthetic; the backbone is Qwen2.5-7B.**

## Comparison

| approach | synthetic recall | TriviaQA (general) | WikiText-103 PPL | weights | per-query cost |
|---|---|---|---|---|---|
| **Memory Layers + relevance gate** | **~100 %** (5000) | 52.5–52.8 % (−0.6) | **+2.3 %** | backbone frozen, added params | none (parametric) |
| **RAG** (BM25 sparse) | 99.4 % (5000) | = backbone 53.4 % | = backbone (non-destructive) | unchanged | retrieval + injected context (~per-query) |
| **LoRA** (r=16, all-linear) | 82.7 % (1000) | **0.7 %** 💥 | **+45 %** | modified (low-rank Δ) | none |
| **kNN-LM** (declarative datastore) | **0 %** (1000) | 42 % / 8 % | +15 % / +70 % | unchanged | per-token kNN over datastore |
| **kNN-LM** (Q&A datastore, v0.3.4) | **≤ 13 %** (100, only at λ=0.9) | — | — | unchanged | per-token kNN over datastore |

*(Scales differ where noted — Memory/RAG at 5000 facts, LoRA/kNN-LM at 1000, kNN-LM-QA at 100; the
qualitative ordering is robust. Backbone references: TriviaQA 53.4 % at n=1000, WikiText-103 slice
PPL 7.24.)*

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

**kNN-LM (non-parametric interpolation).** Datastore of (hidden-state → next-token) pairs;
next-token distribution interpolated with the LM. With a **declarative** datastore it **fails the
recall task (0 %)**: the keys come from declarative contexts (“…is HHHH”) but the test reformulates
as a question, so the question's hidden state does not match the declarative keys → wrong
neighbours. To check this 0 % was not merely a datastore-format artifact, **v0.3.4 rebuilt the
datastore in Q&A form** (key = hidden state over the *answer span* of a `question → answer`
sequence, value = answer token), so the query's question-end state matches the key preceding the
answer. Result (n=100): recall stays **0 % up to λ=0.5**, reaches **2 % at λ=0.7** and **13 % at
λ=0.9** — i.e. it only recalls anything when the kNN term nearly **overrides** the LM (λ=0.9),
which is exactly the regime that destroys the general distribution (PPL +70 %, TriviaQA → 8 % at
high λ). So the format fix lifts the artifactual 0 % but confirms kNN-LM is **structurally
inadequate** for exact-string fact recall here: it cannot inject a fact without either ignoring it
(low λ) or wrecking general competence (high λ), and even then tops out at 13 % vs ~100 %.

## Takeaway

Only **Memory Layers + relevance gate** simultaneously (i) reaches ~100 % stored-fact recall,
(ii) preserves general competence (PPL +2.3 %, TriviaQA −0.6), and (iii) is **parametric** — no
retrieval index, no per-query context. RAG matches recall and preservation but pays a per-query
retrieval/context cost and keeps knowledge out of the weights; LoRA puts knowledge in the weights
but forgets catastrophically; kNN-LM — even with a format-matched Q&A datastore — peaks at 13 %
recall only in the general-distribution-destroying regime. This positions the contribution as a
**parametric, non-destructive** alternative for injecting a closed set of facts into a frozen model.

*Memorizing Transformers (Wu et al. 2022) — kNN-augmented attention — is discussed in Related
Work rather than reimplemented (a faithful integration into this stack is out of scope for this
report).*
