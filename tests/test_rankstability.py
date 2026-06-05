import numpy as np

from benchgauge.model import EvalLog
from benchgauge.rankstability import rank
from benchgauge.rankstability.paired import holm_adjust
from benchgauge.results import DISTINGUISHABLE, NOT_DISTINGUISHABLE


def _synth(rng, thetas, n_items=400, a=1.5, b=0.0):
    M = len(thetas)
    scores = np.zeros((M, n_items))
    for i, th in enumerate(thetas):
        p = 1.0 / (1.0 + np.exp(-a * (th - b)))
        scores[i] = rng.random(n_items) < p
    return EvalLog.from_matrix(scores, model_ids=tuple(f"m{i}" for i in range(M)))


def test_clear_difference_is_established():
    rng = np.random.default_rng(0)
    log = _synth(rng, thetas=[1.5, -1.5], n_items=400)
    res = rank(log)
    pair = res.pairs[0]
    assert pair.verdict == DISTINGUISHABLE
    assert pair.lead == "m0"  # higher theta
    assert pair.ci95[0] > 0  # CI excludes zero


def test_no_difference_is_not_established():
    rng = np.random.default_rng(1)
    log = _synth(rng, thetas=[0.0, 0.0], n_items=400)
    res = rank(log)
    pair = res.pairs[0]
    assert pair.verdict == NOT_DISTINGUISHABLE
    assert pair.lead is None
    assert pair.ci95[0] < 0 < pair.ci95[1]  # CI brackets zero


def test_mdd_positive():
    rng = np.random.default_rng(2)
    log = _synth(rng, thetas=[0.5, 0.0], n_items=300)
    res = rank(log)
    assert res.pairs[0].mdd > 0


def test_holm_monotone_and_correct():
    # smallest p must stay <= its raw*(m); adjusted p monotone non-decreasing in p
    p = np.array([0.001, 0.02, 0.5, 0.9])
    reject, adj = holm_adjust(p, alpha=0.05)
    assert reject[0]
    assert not reject[3]
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_full_family_correction_reduces_false_leads():
    # 6 truly-equal models -> with FWER control we expect ~no established pairs
    rng = np.random.default_rng(3)
    log = _synth(rng, thetas=[0.0] * 6, n_items=300)
    res = rank(log)
    assert res.n_established == 0
