# Related work (v0.3.3)

How this reconstruction sits relative to the literature on giving language models new facts. We
group prior work by *where the knowledge lives* and *what is modified*, then state our position.

## Memory layers (the substrate)

- **Lample et al. 2019, *Large Memory Layers with Product Keys*** (arXiv:1907.05242). Product-key
  memory: a large key-value store addressed by a factorised (product-key) nearest-neighbour
  lookup, so an enormous value table can be queried in sub-linear time. The addressing mechanism
  we reconstruct.
- **Berges et al. 2024, *Memory Layers at Scale*** (arXiv:2412.09764). Scales memory layers inside
  modern LLMs (shared value pool, qk-normalisation, gating). This repository is an independent,
  from-scratch reconstruction of this architecture on a single 24 GB consumer GPU.

## Injecting facts into memory layers (closest concurrent work)

- **Lin et al. 2025, *Continual Learning via Sparse Memory Finetuning*** (arXiv:2510.15103, Meta
  FAIR). The most directly related work: built on memory-layer models (Berges 2024), it **updates
  only the memory slots most activated by a new fact** (relative to pretraining usage), reducing
  interference. They report NaturalQuestions F1 dropping **89 % (full finetuning) / 71 % (LoRA) /
  11 % (sparse memory finetuning)** at matched knowledge acquisition — i.e. memory-layer sparsity
  mitigates forgetting.
  **Relation to this work.** Both exploit the sparsity of memory layers, but at different points:
  Sparse Memory Finetuning *sparsely **updates** the memory weights*; we keep the backbone **and**
  the memory **frozen** and add a small **inference-time relevance gate** that conditions the
  always-on memory output (open on stored-fact contexts, closed elsewhere). The two are
  complementary — a learned-update sparsity vs an inference-time gating — and their LoRA-forgetting
  numbers independently corroborate our LoRA baseline (`docs/BASELINES.md`).
- **Lin et al. 2025, *Learning Facts at Scale with Active Reading*** (arXiv:2508.09494, Meta FAIR).
  "Active Reading" trains a model to *study* source material with self-generated strategies,
  absorbing far more knowledge than vanilla finetuning (e.g. +313 % relative on a Wikipedia SimpleQA
  subset; releases a WikiExpert-8B). This addresses the *training-data/strategy* side of reliable
  parametric fact absorption — orthogonal to, and combinable with, an inference-time gate.

## Non-parametric / retrieval approaches

- **Lewis et al. 2020, *Retrieval-Augmented Generation*** (arXiv:2005.11401). Retrieve documents
  and condition generation on them — non-destructive (weights unchanged) but pays a retrieval +
  context cost per query. Our RAG baseline.
- **Khandelwal et al. 2020, *Generalization through Memorization: Nearest Neighbor LMs*** (kNN-LM,
  arXiv:1911.00172). Interpolate the LM's next-token distribution with a kNN distribution over a
  datastore of hidden states. Our kNN-LM baseline; we find a naive fact datastore fails the Q&A
  recall task (declarative→question mismatch) while taxing the general distribution.
- **Wu et al. 2022, *Memorizing Transformers*** (arXiv:2203.08913). kNN-augmented attention over an
  external memory of (key, value) pairs. Conceptually adjacent to kNN-LM at the attention level; we
  discuss it here rather than reimplement it (a faithful integration into this stack is out of
  scope for this report).

## Parameter-efficient finetuning

- **Hu et al. 2021, *LoRA*** (arXiv:2106.09685). Low-rank weight adapters. Our LoRA baseline.
- **Biderman et al. 2024, *LoRA Learns Less and Forgets Less*** (arXiv:2405.09673). Characterises
  LoRA's lower capacity *and* its milder forgetting **in their continual-learning regime**. Our
  setting differs (injecting a closed set of narrow synthetic facts), and there LoRA forgets
  catastrophically (TriviaQA 53.4 % → 0.7 %) — the "forgets less" claim is regime-dependent.

## Our position

This work studies the **retrofit** setting: a **pre-trained, frozen** backbone with an **added,
frozen** product-key memory, where the only trainable component is a small **learned
domain-relevance gate** on the hidden state. We are therefore neither a weight-edit method (full
finetuning, LoRA, sparse memory finetuning) nor a retrieval method (RAG, kNN-LM, Memorizing
Transformers): the contribution is the **inference-time gate** that removes the always-on memory's
general-competence tax while keeping the backbone and memory frozen — and the characterisation of
*how far that gate generalises* (`docs/GENERALIZATION.md`). The closest neighbour, Sparse Memory
Finetuning, achieves a similar "low-forgetting fact injection" goal by sparsely updating the memory
instead; comparing the two head-to-head is natural future work.
