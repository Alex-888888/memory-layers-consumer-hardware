# Threats to validity (v0.3.3, updated in v0.3.4)

An explicit, not-minimised list of what could make the conclusions weaker than they look. The aim
is to mark the frontier of what this reconstruction does and does not show.

## Data

- **Synthetic facts with marked statistical signatures.** The fact families (sensor hex pairs,
  config integers, service semver, node coordinates, protocol hex bytes) are easy to tell apart
  and easy to address. Real heterogeneous knowledge may be harder, and the gate's
  *distributional-cluster* generalisation (`docs/GENERALIZATION.md`) could behave differently on
  natural facts.
- **A single, simple natural-language family.** The only non-structured family (biographical
  birthplaces) is one simple paradigm. We show held-out NL *entities* recover once the NL family is
  seen, but with only one NL family we **cannot distinguish per-cluster from per-family coverage**
  (`docs/GENERALIZATION.md` §2). More, and more varied, NL families would settle this.
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
  1000 (kNN-LM-QA at 100); generative evals were reduced (n≈100–500 for some baselines) because
  generation is slow on this GPU. The qualitative ordering is robust but the exact baseline numbers
  would tighten with larger n.
- **Baselines are not exhaustively tuned.** LoRA (rank, target modules, epochs) and kNN-LM (λ, k,
  datastore construction) were swept only modestly. The original **kNN-LM 0 %** came from a
  declarative→question datastore mismatch; **v0.3.4 re-ran kNN-LM with a Q&A-format datastore** and
  measured **≤ 13 % recall, reached only at λ=0.9** (the general-distribution-destroying regime) —
  so the 0 % was partly a format artifact, but kNN-LM remains structurally inadequate here rather
  than merely strawmanned (`docs/BASELINES.md`). Stronger LoRA tuning could still narrow its gap.
- **Memorizing Transformers not implemented.** It is discussed in Related Work only.

## Claims that rest on this specific setup

- **No emergent entity-level safety (corrected in v0.3.4).** v0.3.1 claimed the memory exhibits
  emergent entity-level safety via a decode-confidence collapse on non-stored entities. Re-measured
  at n=360 (`docs/SAFETY_EVAL.md`), this is **not** supported: the model **confidently fabricates
  plausible, format-correct wrong values** for non-stored entities (it does not abstain), and
  decode-confidence separates stored from fake only **partially** (AUC 0.69). The one robust
  entity-level property is **no inter-fact leakage** (the model does not reproduce the unseen value
  nor pull a stored fact for an unknown entity). A deployment needing real abstention must add an
  explicit mechanism; this memory does not provide one.
- **Domain-level, not entity-level.** The gate detects the distributional signature of a stored-fact
  context, not whether a *specific* entity is stored. It needs ≥1 example per family (plausibly per
  distributional cluster); a held-out natural-language family is not recovered.

## Statistics

- TriviaQA is reported at n=1000 (95 % binomial CI ±3.1 pts; see `docs/STATS.md`); the gated
  residual is within that CI. Perplexity is over 220k tokens (population-scale). Recall is sampled
  up to n=1000 and sits at the ceiling. The reported ±σ on gated TriviaQA is inter-seed
  gate-training reproducibility, not sampling error. The decode-confidence AUC (0.69) is over
  360 stored / 360 fake (`docs/SAFETY_EVAL.md`).

## What would most strengthen the work

A head-to-head comparison with Sparse Memory Finetuning (arXiv:2510.15103) on the same facts; real
(non-synthetic) heterogeneous facts; a second, structurally different NL family (to settle
per-cluster vs per-family coverage); an explicit abstention mechanism for non-stored entities; more
than one backbone; and larger-n external benchmarks.
