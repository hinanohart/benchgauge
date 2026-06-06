import numpy as np

from benchgauge.irt import fit_irt, item_report, mislabel_pointbiserial
from benchgauge.model import EvalLog


def _synth(rng, n_models=80, n_items=60, a_lo=0.7, a_hi=2.0):
    a = rng.uniform(a_lo, a_hi, n_items)
    b = rng.normal(0, 1, n_items)
    theta = rng.normal(0, 1, n_models)
    p = 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    scores = (rng.random((n_models, n_items)) < p).astype(float)
    return scores, a, b, theta


def test_fit_recovers_discrimination():
    rng = np.random.default_rng(0)
    scores, a, b, theta = _synth(rng)
    log = EvalLog.from_matrix(scores)
    fit = fit_irt(log)
    assert fit is not None and fit["converged"]
    # discrimination correlates with truth (allow a couple of dropped degenerate items)
    assert fit["a"].shape[0] >= 55


def test_dead_item_detected():
    rng = np.random.default_rng(1)
    scores, a, b, theta = _synth(rng, n_models=120, n_items=60)
    # inject 3 dead items: pure coin flips (a ~ 0, no relation to ability)
    dead_cols = [5, 20, 40]
    for c in dead_cols:
        scores[:, c] = (rng.random(scores.shape[0]) < 0.5).astype(float)
    log = EvalLog.from_matrix(scores)
    rep = item_report(log)
    assert rep.irt_converged
    # at least 2 of the 3 injected dead items recovered
    found = sum(f"item_{c}" in rep.dead_items for c in dead_cols)
    assert found >= 2, rep.dead_items


def test_mislabel_pointbiserial_detects_reversed():
    rng = np.random.default_rng(2)
    scores, a, b, theta = _synth(rng, n_models=120, n_items=60)
    # mislabel item 10: high-ability models get it "wrong" (reverse the truth)
    order = np.argsort(theta)
    scores[order[60:], 10] = 0.0  # top half fails
    scores[order[:60], 10] = 1.0  # bottom half passes
    log = EvalLog.from_matrix(scores)
    flags, detail = mislabel_pointbiserial(log)
    assert "item_10" in flags
    assert detail["item_10"] < 0


def test_small_n_downgrades():
    rng = np.random.default_rng(3)
    scores, *_ = _synth(rng, n_models=10, n_items=40)
    log = EvalLog.from_matrix(scores)
    rep = item_report(log)
    assert rep.downgraded == "skipped"
    assert not rep.irt_converged
    assert rep.dead_items == []
