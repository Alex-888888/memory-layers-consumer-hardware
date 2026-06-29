# Statistical notes (v0.3.2)

This note makes the uncertainty in the reported numbers explicit, so the claims are calibrated.

## Two kinds of variability (kept distinct)

1. **Sampling uncertainty** — the finite-eval-set error of an estimate (e.g. TriviaQA on 1000
   questions). Quantified by a binomial confidence interval.
2. **Inter-seed reproducibility** — how much a number moves when the *gate* is retrained from
   scratch with a different seed. This is what the `± …` on the gated TriviaQA figures reports —
   it is **not** the sampling error.

## TriviaQA — sampling confidence intervals (n = 1000)

The 95 % Wald binomial half-width at p ≈ 0.5 and n = 1000 is
`1.96 · √(0.5·0.5 / 1000) ≈ ±3.1 pts`.

| config | TriviaQA | 95 % CI |
|---|---|---|
| backbone | 53.4 % | [50.3, 56.5] |
| + memory, ungated | 45.2 % | [42.1, 48.3] |
| + memory, gated (v6) | 52.5 % | [49.4, 55.6] |
| + memory, gated (multi-domain) | 52.8 % | [49.7, 55.9] |

**Reading.**
- The **ungated** regression (−8.2 pts) is **statistically significant**: its CI [42.1, 48.3] is
  disjoint from the backbone's [50.3, 56.5].
- The **gated** residual (−0.6 / −0.9 pt) is **within sampling noise**: the gated CIs overlap the
  backbone CI substantially. So the honest claim is **"the gate removes the statistically
  significant regression; no significant regression remains"** — stronger and more defensible than
  "recovers 89 % of the loss".

## Inter-seed σ (gate-training reproducibility)

The reported **± 1.74** (gate v6) and **± 0.28** (multi-domain) are the standard deviation of
TriviaQA across **3 independent gate trainings** (seeds), i.e. the gate-training is reproducible.
For comparison, the 1σ *sampling* standard error at n = 1000 is ≈ ± 1.58 pts. Both are small; they
measure different things and we report both rather than conflating them.

## Perplexity

WikiText-103 perplexity is computed over **220k tokens / 108 non-overlapping 2048-token windows**
— effectively a population quantity at this scale, with negligible sampling noise. The gated
**+2.3 %** figure is deterministic given the model (no error bar needed).

## Recall

Synthetic recall sits at the ceiling: **100 % with σ = 0** across seeds {137, 7, 23} at 100 / 300
/ 1000 facts, and **1000 / 1000** on the 5000-fact production anchor. The Wilson 95 % interval for
p̂ = 1.0 at n = 1000 is **[99.6 %, 100 %]**. (Recall was measured on samples up to n = 1000; a
full 5000-fact sweep is a longer job and would only tighten an interval already at the ceiling.)

## One-line restatement

> The always-on memory causes a **statistically significant** drop in general-knowledge accuracy
> (TriviaQA −8.2 pts, CI-disjoint). The relevance gate **removes it**: the gated residual
> (−0.6 / −0.9 pt) is **within sampling noise** (CI-overlapping), and is **reproducible** across
> gate-training seeds (σ ≤ 1.74). Stored-fact recall stays at the ceiling and perplexity within
> +2.3 %.
