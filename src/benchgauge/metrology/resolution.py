"""Resolution (ndc): how many distinguishable performance tiers does this
benchmark carve the model set into?

ndc = number of connected components of the graph whose edges connect models
that are NOT statistically distinguishable (rank verdict != DISTINGUISHABLE).
Two models sharing an edge cannot be separated, so they belong to the same tier;
the number of components is the number of resolvable tiers.

This is deliberately the circular-reasoning-safe definition: it makes NO
measurement-repeatability claim and never improves merely by adding items. It is
relative to the specific model set, which is stated in every report.
"""

from __future__ import annotations

import numpy as np

from benchgauge.metrology.cluster import _UF, effective_models
from benchgauge.model import EvalLog
from benchgauge.results import DISTINGUISHABLE, INSUFFICIENT, MARGINAL, RESOLVED, ResolutionResult


def resolution(
    rank_result, log: EvalLog, fail_under: int = 2, dedup_thresh: float = 0.99
) -> ResolutionResult:
    ids = list(log.model_ids)
    m = len(ids)
    idx = {mid: i for i, mid in enumerate(ids)}
    uf = _UF(m)
    for p in rank_result.pairs:
        if p.verdict != DISTINGUISHABLE:
            uf.union(idx[p.a], idx[p.b])
    groups_idx = uf.groups()
    # order tiers by mean observed score (descending) for readability
    means = []
    for g in groups_idx:
        vals = log.scores[g][:, :]
        msk = log.mask[g][:, :]
        mean = float(vals[msk].mean()) if msk.any() else 0.0
        means.append(mean)
    order = np.argsort(means)[::-1]
    tiers = [[ids[i] for i in groups_idx[o]] for o in order]
    ndc = len(groups_idx)

    if ndc < fail_under:
        verdict = INSUFFICIENT
    elif ndc < 5:
        verdict = MARGINAL
    else:
        verdict = RESOLVED

    eff = effective_models(log, thresh=dedup_thresh)
    note = (
        f"ndc={ndc} is relative to this set of {m} models "
        f"(effective {eff['effective_n']} after merging score-corr>{dedup_thresh} near-duplicates). "
        f"It is a count of statistically separable tiers, not an absolute benchmark quality."
    )
    return ResolutionResult(
        ndc=ndc, verdict=verdict, tiers=tiers, effective_n_models=eff["effective_n"], note=note
    )
