"""Item-quality forensics: dead items, suspected mislabels, saturation.

- **mislabel** is detected with a point-biserial (leave-one-out item-total)
  correlation < 0: "the more able models systematically miss this item". This is
  classical test theory -- numpy only, no IRT needed, backend-independent -- and
  is therefore the PRIMARY mislabel signal, available even when IRT is skipped.
  (A negative IRT discrimination is *not* used: girth/py-irt constrain a > 0, so
  "negative a" is dead code.)
- **dead items** (a ~ 0) and **saturation** (test information collapses at high
  ability) require a fitted 2PL and are suppressed for small model sets.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from benchgauge.irt.backends import IRTBackend
from benchgauge.irt.fit import fit_irt
from benchgauge.model import EvalLog
from benchgauge.results import ItemReport

N_FULL_IRT = 25  # below this, suppress a-based flags (a posterior too diffuse)
DEAD_A_THRESH = 0.4  # IRT discrimination below this is "barely informative"
MISLABEL_R = -0.15  # point-biserial threshold
SATURATION_RATIO = 0.2


def test_information(a: np.ndarray, b: np.ndarray, theta_grid: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-a[:, None] * (theta_grid[None, :] - b[:, None])))
    return (a[:, None] ** 2 * p * (1.0 - p)).sum(axis=0)


def mislabel_pointbiserial(
    log: EvalLog, r_thresh: float = MISLABEL_R, alpha: float = 0.05
) -> tuple[list[str], dict]:
    scores, mask = log.scores, log.mask
    flags: list[str] = []
    detail: dict = {}
    for j in range(log.n_items):
        obs = mask[:, j]
        rows = np.where(obs)[0]
        if rows.size < 5:
            continue
        col = scores[rows, j]
        if np.std(col) == 0:
            continue
        proxy = np.full(rows.size, np.nan)
        for k, i in enumerate(rows):
            others = mask[i].copy()
            others[j] = False
            if others.any():
                proxy[k] = scores[i, others].mean()
        good = ~np.isnan(proxy)
        if good.sum() < 5:
            continue
        c, pr = col[good], proxy[good]
        if np.std(c) == 0 or np.std(pr) == 0:
            continue
        r = float(np.corrcoef(c, pr)[0, 1])
        detail[log.item_ids[j]] = r
        n_eff = int(good.sum())
        if r < r_thresh and n_eff > 2:
            t = r * np.sqrt((n_eff - 2) / max(1e-9, 1.0 - r * r))
            if float(stats.t.cdf(t, n_eff - 2)) < alpha:
                flags.append(log.item_ids[j])
    return flags, detail


def item_report(log: EvalLog, backend: IRTBackend | None = None) -> ItemReport:
    n = log.n_models
    mislabel, mis_detail = mislabel_pointbiserial(log)

    fit = fit_irt(log, backend) if n >= 15 else None
    if fit is None:
        return ItemReport(
            dead_items=[],
            suspected_mislabel=mislabel,
            saturation_theta=None,
            irt_backend="none",
            irt_converged=False,
            n_models=n,
            downgraded="skipped" if n < 15 else "irt_not_fit",
            detail={"pointbiserial": mis_detail},
        )

    a, b, theta = fit["a"], fit["b"], fit["theta"]
    used = fit["item_ids"]
    dead: list[str] = []
    saturation_theta = None
    downgraded = None

    if n >= N_FULL_IRT:
        p = 1.0 / (1.0 + np.exp(-a[:, None] * (theta[None, :] - b[:, None])))
        info_contrib = (a[:, None] ** 2 * p * (1.0 - p)).mean(axis=1)
        info_eps = 0.05 * float(info_contrib.max()) if info_contrib.size else 0.0
        for i in range(len(a)):
            if a[i] < DEAD_A_THRESH and info_contrib[i] <= info_eps:
                dead.append(used[i])
        # saturation: test information at a frontier ability (theta=2) vs the
        # centre (theta=0). Fixed reference points (not data quantiles) keep the
        # criterion stable and matched to the pre-registered gate.
        th_ref, th_high = 0.0, 2.0
        info = test_information(a, b, np.array([th_ref, th_high]))
        if info[0] > 1e-9 and info[1] / info[0] < SATURATION_RATIO:
            saturation_theta = th_high
    else:
        downgraded = "a_flag_suppressed"

    return ItemReport(
        dead_items=dead,
        suspected_mislabel=mislabel,
        saturation_theta=saturation_theta,
        irt_backend=fit["backend"],
        irt_converged=True,
        n_models=n,
        downgraded=downgraded,
        detail={"pointbiserial": mis_detail, "binarised": fit["binarised"]},
    )
