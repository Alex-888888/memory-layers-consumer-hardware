# Threats to validity (v0.3.3)

An explicit, not-minimised list of what could make the conclusions weaker than they look. The aim
is to mark the frontier of what this reconstruction does and does not show.

## Data

- **Synthetic facts with marked statistical signatures.** The fact families (sensor hex pairs,
  config integers, service semver, node coordinates, protocol hex bytes) are easy to tell apart
  and easy to address. Real heterogeneous knowledge may be harder, and the gate's
  *distributional-cluster* generalisation (`docs/GENERALIZATION.md`) could behave differently on
  natural facts.
- **A single, simple natural-language family.** The only non-structured family (biographical
  birthplaces) is one simple paradigm; "the gate handles NL once it has seen NL" rests on that one
  example. More, and more varied, NL families would strengthen (or qualify) the claim.
- **Closed fact set.** The memory stores a *fixed* set of facts. Open-set operation, fact updating,
  and deletion are not addressed.

## Model and hardware

- **One backbone, one checkpoint per scale.** All results use Qwen2.5-7B-Instruct and a single
  trained memory per fact-count. No variation across model families, sizes, or training seeds *of
  the memory* (seeds vary the recipe and the gate, not the production memory).
- **Hardware-specific findings.** The pool ceiling (~500k) and the generation latencies are tied to
  a 24 GB RX 7900 XTX under ROCm/WSL2. On other hardware the ceiling and timings differ; the 1M
  target is explicitly deferred to more capable hardware.

## Baselines

- **Differing scales and reduced eval n.** Memory/RAG are evaluated at 5000 facts, LoRA/kNN-LM at
  1000; generative evals were reduced (n≈150–500 for some baselines) because generation is slow on
  this GPU. The qualitative ordering is robust but the exact baseline numbers would tighten with
  larger n.
- **Baselines are not exhaustively tuned.** LoRA (rank, target modules, epochs) and kNN-LM (λ,
  k, datastore construction) were swept only modestly. In particular, **kNN-LM's 0 % recall reflects
  a declarative→question datastore mismatch**, not an upper bound on kNN-LM — a Q&A-formatted
  datastore would likely recall better (while still taxing the general distribution). Stronger
  baseline tuning could narrow the gaps.
- **Memorizing Transformers not implemented.** It is discussed in Related Work only.

## Claims that rest on this specific setup

- **"No hallucination" is empirical.** The safety of opening the gate on non-stored entities comes
  from *this* memory's retrieval geometry (incoherent retrieval → collapsed decode confidence,
  `docs/GENERALIZATION.md`). A memory whose out-of-store retrieval is coherent could hallucinate;
  the property is not guaranteed by the gate.
- **Domain-level, not entity-level.** The gate detects the distributional signature of a stored-fact
  context, not whether a *specific* entity is stored. It needs ≥1 example per distributional
  cluster; a held-out natural-language family is not recovered.

## Statistics

- TriviaQA is reported at n=1000 (95 % binomial CI ±3.1 pts; see `docs/STATS.md`); the gated
  residual is within that CI. Perplexity is over 220k tokens (population-scale). Recall is sampled
  up to n=1000 and sits at the ceiling. The reported ±σ on gated TriviaQA is inter-seed
  gate-training reproducibility, not sampling error.

## What would most strengthen the work

A head-to-head comparison with Sparse Memory Finetuning (arXiv:2510.15103) on the same facts; real
(non-synthetic) heterogeneous facts; more than one backbone; and larger-n external benchmarks.
