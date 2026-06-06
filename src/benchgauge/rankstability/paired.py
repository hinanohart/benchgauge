"""Paired, item-clustered standard errors for model-vs-model comparisons.

Items are the sampling/cluster unit. For two models A and B the per-item
difference d_i = score(A,i) - score(B,i) over commonly-observed items has
mean d-bar and clustered SE = sd(d)/sqrt(n) (Miller, arXiv:2411.00640, eq.4/7).
This automatically accounts for the A-B correlation across items.

Analytic clustered SE is the PRIMARY estimator: Miller shows the CLT applies to
any finite-variance eval with enough items and that bootstrap is unnecessary.
A paired item-cluster bootstrap (``bootstrap_se``) is provided as an opt-in
SECONDARY robustness check for small-n / 0-1 saturated regimes; it is not on the
default ``rank`` path. Holm-Bonferroni controls the family-wise error rate across
all C(n,2) pairs and is implemented here so the core stays statsmodels-free.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

Z_POWER_80 = float(stats.norm.ppf(0.80))  # 0.8416, for the MDD power term


def pair_stats(scores: np.ndarray, mask: np.ndarray, i: int, j: int, alpha: float) -> dict | None:
    """Clustered paired statistics for models i vs j. Returns None if < 2 common items."""
    both = mask[i] & mask[j]
    n = int(both.sum())
    if n < 2:
        return None
    d = scores[i, both] - scores[j, both]
    dbar = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    zc = float(stats.norm.ppf(1 - alpha / 2))
    if se == 0.0:
        # all per-item differences identical (e.g. both models tied on every item)
        z = 0.0 if dbar == 0.0 else float(np.sign(dbar)) * np.inf
        p = 1.0 if dbar == 0.0 else 0.0
        ci = (dbar, dbar)
    else:
        z = dbar / se
        p = float(2.0 * stats.norm.sf(abs(z)))
        ci = (dbar - zc * se, dbar + zc * se)
    mdd = (zc + Z_POWER_80) * se
    return {
        "n": n,
        "dbar": dbar,
        "se": se,
        "z": float(z),
        "p": float(p),
        "ci": (float(ci[0]), float(ci[1])),
        "mdd": float(mdd),
    }


def holm_adjust(pvals: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down. Returns (reject_bool, adjusted_pvalues)."""
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(pvals)
    p_adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvals[idx] * (m - rank)))
        p_adj[idx] = running
    return p_adj <= alpha, p_adj


def bootstrap_se(
    scores: np.ndarray,
    mask: np.ndarray,
    i: int,
    j: int,
    rng: np.random.Generator,
    n_boot: int = 1000,
) -> float | None:
    """SECONDARY paired item-cluster bootstrap SE (resample items with replacement)."""
    both = mask[i] & mask[j]
    n = int(both.sum())
    if n < 2:
        return None
    d = scores[i, both] - scores[j, both]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    return float(np.std(means, ddof=1))
