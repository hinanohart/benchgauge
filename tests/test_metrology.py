import numpy as np

from benchgauge.metrology import effective_models, resolution
from benchgauge.model import EvalLog
from benchgauge.rankstability import rank
from benchgauge.results import INSUFFICIENT, RESOLVED


def _synth(rng, thetas, n_items=400, a=1.5):
    M = len(thetas)
    scores = np.zeros((M, n_items))
    for i, th in enumerate(thetas):
        p = 1.0 / (1.0 + np.exp(-a * (th - 0.0)))
        scores[i] = rng.random(n_items) < p
    return EvalLog.from_matrix(scores, model_ids=tuple(f"m{i}" for i in range(M)))


def test_ndc_high_when_models_separated():
    rng = np.random.default_rng(0)
    log = _synth(rng, thetas=[-2.5, -1.2, 0.0, 1.2, 2.5], n_items=600)
    res = resolution(rank(log), log)
    assert res.ndc >= 5
    assert res.verdict == RESOLVED


def test_ndc_one_when_no_difference():
    rng = np.random.default_rng(1)
    log = _synth(rng, thetas=[0.0] * 5, n_items=400)
    res = resolution(rank(log), log)
    assert res.ndc < 2
    assert res.verdict == INSUFFICIENT


def test_effective_models_merges_duplicates():
    rng = np.random.default_rng(2)
    base = (rng.random(300) < 0.6).astype(float)
    dup = base.copy()
    other = (rng.random(300) < 0.3).astype(float)
    log = EvalLog.from_matrix(np.vstack([base, dup, other]), model_ids=("a", "a_copy", "b"))
    eff = effective_models(log)
    assert eff["effective_n"] == 2
    assert eff["merged"]
