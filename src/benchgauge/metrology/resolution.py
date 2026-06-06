"""Resolution (ndc): how many distinguishable performance tiers does this
benchmark carve the model set into?

ndc = number of tiers produced by walking models up the ability axis (by mean
score) and opening a new tier whenever a model is statistically distinguishable
(Holm-corrected) from the current tier's anchor. This greedy, deterministic count
is intransitivity-safe and does not collapse to 1 on a smooth ability continuum.

This is deliberately the circular-reasoning-safe definition: it makes NO
measurement-repeatability claim and never improves merely by adding items. It is
relative to the specific model set, which is stated in every report.
"""

from __future__ import annotations

import numpy as np

from benchgauge.metrology.cluster import effective_models
from benchgauge.model import EvalLog
from benchgauge.results import DISTINGUISHABLE, INSUFFICIENT, MARGINAL, RESOLVED, ResolutionResult


def resolution(
    rank_result, log: EvalLog, fail_under: int = 2, dedup_thresh: float = 0.99
) -> ResolutionResult:
    ids = list(log.model_ids)
    m = len(ids)
    # mean observed score per model (ascending order = the ability axis)
    means = np.array(
        [float(log.scores[i][log.mask[i]].mean()) if log.mask[i].any() else 0.0 for i in range(m)]
    )
    order = list(np.argsort(means, kind="stable"))  # low score -> high score

    # statistically distinguishable pairs (Holm-corrected rank verdicts)
    dist = {frozenset((p.a, p.b)) for p in rank_result.pairs if p.verdict == DISTINGUISHABLE}

    def distinguishable(i: int, j: int) -> bool:
        return frozenset((ids[i], ids[j])) in dist

    # Greedy sequential tiering up the ability axis: a new tier opens when a
    # model is statistically distinguishable from the *anchor* (lowest member)
    # of the current tier. ndc = number of tiers. This counts resolvable steps
    # and -- unlike connected components of the not-distinguishable graph -- does
    # NOT collapse to 1 on a smooth continuum of abilities (intransitivity-safe).
    tiers_idx: list[list[int]] = []
    if order:
        anchor = order[0]
        current = [order[0]]
        for k in order[1:]:
            if distinguishable(anchor, k):
                tiers_idx.append(current)
                current = [k]
                anchor = k
            else:
                current.append(k)
        tiers_idx.append(current)
    ndc = len(tiers_idx)

    # high score -> low score for display
    tiers = [[ids[i] for i in t] for t in tiers_idx][::-1]

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
        f"It counts statistically separable tiers along the ability axis, "
        f"not an absolute benchmark quality."
    )
    return ResolutionResult(
        ndc=ndc, verdict=verdict, tiers=tiers, effective_n_models=eff["effective_n"], note=note
    )
