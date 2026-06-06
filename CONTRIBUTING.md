# Contributing to benchgauge

Thanks for your interest! benchgauge is pre-alpha; the API may change.

## Dev setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src/ tests/
```

The core install is intentionally **torch-free** and **statsmodels-free**. Keep
it that way: anything pulling in heavy/optional dependencies belongs in an extra
(`[irt-bayes]`, `[parquet]`).

## Design rules (enforced in CI)

- **Diagnostic framing only.** benchgauge audits a benchmark as a measuring
  instrument; it never claims a model is "better". Please keep the README and
  docs free of performance/marketing language — CI scans for it.
- **Deterministic outputs.** `src/benchgauge/report/` and `src/benchgauge/gate.py`
  must contain no random calls; all randomness lives in `selfcheck/synth.py`
  behind integer seeds. CI greps for this.
- **The sensitivity gates (G1–G8) are the lifeline.** Any change to the
  statistics must keep `benchgauge gate` green, including the G8 nominal-coverage
  band [0.93, 0.97]. New statistical behaviour should come with a gate or a unit
  test that injects known ground truth and recovers it.
- **`ndc` is relative to the model set** and must always be reported as such.

## Pull requests

Run `pytest -q` and `ruff check` before opening a PR. Small, focused changes are
easiest to review. New adapters should ship a pinned real-format fixture under
`tests/fixtures/` and a round-trip test.
