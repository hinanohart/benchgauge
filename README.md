# benchgauge

**Is your benchmark a measuring instrument?**

`benchgauge` is an offline, CPU-only, model-agnostic **diagnostic** for evaluation
logs. You hand it a `models × items` correctness matrix (or point it at an
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
`--log_samples` directory) and it answers two questions a leaderboard number
never does:

1. **Resolution** — can this benchmark statistically tell these models apart, and
   into how many distinguishable tiers? (`ndc`)
2. **Item quality** — which items are dead weight (no discrimination) or look
   mislabelled (the stronger models systematically miss them)?

It treats a benchmark the way metrology treats a gauge and the way psychometrics
treats a test: as an instrument whose **resolution and item quality** can be
audited. It is a *diagnostic* tool — it makes **no claim about which model is
"better"** in any absolute sense.

> Status: `v0.1.0a1` (pre-alpha). All recovery numbers quoted below come from
> **synthetic ground-truth** experiments (machine-labelled), not from real model
> runs. Treat the API as unstable.

## Install

```bash
pip install benchgauge                 # core: numpy, scipy, girth, pyyaml (torch-free)
pip install "benchgauge[irt-bayes]"    # optional Bayesian IRT backend (py-irt; needs python<3.12)
pip install "benchgauge[parquet]"      # optional parquet I/O (pyarrow)
```

The core install is deliberately **torch-free** and **statsmodels-free** so it
stays light and portable. CI covers Python 3.10–3.12 on Linux and Windows.

## Quickstart

Already have an lm-eval run with `--log_samples`? Point benchgauge at it:

```bash
benchgauge report ./lm_eval_outputs --adapter lm_eval
```

Or from Python:

```python
from benchgauge import load_evallog, ReportCard

log = load_evallog("./lm_eval_outputs", adapter="lm_eval")
card = ReportCard.from_evallog(log)
print(card.to_markdown())
```

### CLI

```
benchgauge rank    INPUT   pairwise lead verdicts (item-clustered SE + Holm)
benchgauge gauge   INPUT   resolution: how many distinguishable tiers (ndc)?
benchgauge item    INPUT   IRT item forensics: dead / suspected-mislabel / saturation
benchgauge report  INPUT   the three views as one Markdown + JSON report card
benchgauge gate            run the G1–G8 self-tests on synthetic ground truth
benchgauge selfcheck       synthetic recovery-rate demonstration
benchgauge convert INPUT --to OUT.parquet
```

**Exit codes:** `0` instrument healthy · `1` unhealthy (low resolution or no
establishable order — a *diagnostic finding, not a performance claim*) · `2`
input error · `3` abstain (too few models/items, all-missing, or the model could
not be fit — we decline to guess).

## The three views

| view | what it tells you | what it never claims |
|------|-------------------|----------------------|
| **resolution (ndc)** | how many statistically separable performance tiers this **model set** falls into | absolute benchmark quality; that a sparse/unrepresentative matrix is trustworthy |
| **rank-stability** | whether `A > B` is significant against item sampling error, with family-wise (Holm) correction | that `A` is a better *model* (out-of-distribution generalisation) |
| **item quality (IRT)** | for `n_models ≥ 25`, item difficulty / discrimination / saturation under a 1-D ability model | mislabel/contamination detection at small `n`; multi-dimensional ability |

### Honest boundaries

- **Diagnostic only.** benchgauge reports nothing about model performance gains.
- **`ndc` is relative to the model set you pass in.** Adding near-duplicate
  fine-tunes inflates apparent spread; benchgauge reports an *effective model
  count* (after merging score-correlation > 0.99 rows) to counter this.
- **"Any log as-is" is an over-claim.** Sparse or unrepresentative matrices give
  misleading resolution. benchgauge records `n_dropped` / `sparsity` and warns
  when **curation** is required, and abstains (exit 3) rather than guess.
- **Numbers are measured or synthetic, and labelled as such.** The report card
  carries `labels.synthetic` and a determinism label; `report` and `gate` contain
  no random placeholders.

## How it differs from what exists

This is **operationalisation, not a new statistic.** The underlying methods are
known:

- [GeneralizIT](https://arxiv.org/abs/2411.17880) is a general G-theory toolkit;
  benchgauge is an **eval-log-specific** adapter + IRT + rank-stability report card.
- Item-clustered / paired standard errors follow Miller
  ([arXiv:2411.00640](https://arxiv.org/abs/2411.00640)), which also shows the
  analytic SE is primary and bootstrap is unnecessary — benchgauge uses analytic
  clustered SE first and bootstrap only as a secondary robustness check.
- Benchmark-health meta-studies (e.g. arXiv:2602.11674, arXiv:2602.16763) describe
  discrimination/saturation analyses as research procedures; benchgauge packages
  them as a single turnkey CLI you can run on **any** log.

Modern harnesses (lm-eval) do report standard errors; what benchgauge adds is the
**clustered/paired SE + a resolution verdict + item forensics, integrated** into
one report card with an exit-code contract for CI.

## Sensitivity gates (G1–G8) — the lifeline

Before trusting benchgauge on a real log, it must recover **injected** ground
truth on synthetic matrices `P = sigmoid(a·(θ−b))`, `M ~ Bernoulli(P)`. `benchgauge
gate` runs the pre-registered G1–G8 suite (zero/high resolution, dead-item,
mislabel via point-biserial correlation, indistinguishable pair, saturation,
small-n robustness, and **nominal 95% coverage** ∈ [0.93, 0.97]). If the gates do
not pass, real-log analysis is blocked fail-closed. This is a structural guard
against the "pretty figure that distinguishes nothing" failure mode.

## License

MIT © hinanohart. Contributions welcome — see `CONTRIBUTING.md`.
