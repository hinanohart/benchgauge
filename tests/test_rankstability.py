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


def test_lead_view_orients_margin_to_named_leader():
    # regression (M1): when the leader is the second model (pair.b), the stored
    # mean_diff (a - b) is negative; lead_view must present a POSITIVE lead margin
    # and a CI oriented to the leader, so output never shows "leader wins by -x".
    log = _synth(np.random.default_rng(7), thetas=[-1.5, 1.5], n_items=400)
    p = rank(log).pairs[0]
    assert p.verdict == DISTINGUISHABLE
    assert p.lead == "m1" == p.b
    assert p.mean_diff < 0  # internal (a-b) order is negative
    leader, loser, margin, ci = p.lead_view()
    assert (leader, loser) == ("m1", "m0")
    assert margin > 0
    assert ci[0] > 0 and ci[1] > 0  # CI excludes zero, oriented to the leader
    # the a-oriented case (leader == pair.a) is returned unchanged
    p2 = rank(_synth(np.random.default_rng(8), thetas=[1.5, -1.5], n_items=400)).pairs[0]
    assert p2.lead == "m0" == p2.a
    leader2, _, margin2, _ = p2.lead_view()
    assert leader2 == "m0" and margin2 > 0


def test_bootstrap_se_tracks_analytic_se():
    # the SECONDARY paired item-cluster bootstrap SE should track the PRIMARY
    # analytic clustered SE (operationalizes the otherwise-unexercised estimator).
    from benchgauge.rankstability.paired import bootstrap_se, pair_stats

    log = _synth(np.random.default_rng(11), thetas=[0.6, -0.6], n_items=400)
    s = pair_stats(log.scores, log.mask, 0, 1, 0.05)
    assert s is not None
    bse = bootstrap_se(log.scores, log.mask, 0, 1, np.random.default_rng(12), n_boot=2000)
    assert bse is not None
    assert abs(bse - s["se"]) / s["se"] < 0.15
