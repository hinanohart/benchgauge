"""Public ``rank`` entry point: pairwise lead verdicts with FWER control."""

from __future__ import annotations

from math import nan

import numpy as np

from benchgauge.errors import AbstainError
from benchgauge.model import EvalLog
from benchgauge.rankstability.paired import holm_adjust, pair_stats
from benchgauge.results import (
    ABSTAIN,
    DISTINGUISHABLE,
    NOT_DISTINGUISHABLE,
    RankPair,
    RankResult,
)


def rank(log: EvalLog, alpha: float = 0.05, min_items: int = 1) -> RankResult:
    """Rank-stability verdict for every model pair.

    For each pair we test H0: mean per-item score difference == 0 with an
    item-clustered SE, then Holm-correct across all C(n,2) pairs. A pair is
    ``DISTINGUISHABLE`` only if the corrected test rejects H0.
    """
    m = log.n_models
    if m < 2:
        raise AbstainError(f"rank needs >= 2 models, got {m}")
    if log.n_items < min_items:
        raise AbstainError(f"rank needs >= {min_items} items, got {log.n_items}")

    ids = log.model_ids
    scores, mask = log.scores, log.mask
    pair_idx = [(i, j) for i in range(m) for j in range(i + 1, m)]
    raw = [pair_stats(scores, mask, i, j, alpha) for (i, j) in pair_idx]

    valid = [(k, s) for k, s in enumerate(raw) if s is not None]
    if valid:
        reject, p_adj = holm_adjust(np.array([s["p"] for _, s in valid]), alpha)
    else:
        reject, p_adj = np.zeros(0, bool), np.zeros(0, float)
    adj_by_k = {k: (bool(reject[v]), float(p_adj[v])) for v, (k, _) in enumerate(valid)}

    pairs: list[RankPair] = []
    for k, (i, j) in enumerate(pair_idx):
        s = raw[k]
        if s is None:
            pairs.append(
                RankPair(
                    a=ids[i],
                    b=ids[j],
                    mean_diff=nan,
                    se=nan,
                    z=nan,
                    p_raw=nan,
                    p_holm=nan,
                    ci95=(nan, nan),
                    mdd=nan,
                    n_items=0,
                    lead=None,
                    verdict=ABSTAIN,
                )
            )
            continue
        rej, padj = adj_by_k[k]
        if rej:
            lead = ids[i] if s["dbar"] > 0 else ids[j]
            verdict = DISTINGUISHABLE
        else:
            lead, verdict = None, NOT_DISTINGUISHABLE
        pairs.append(
            RankPair(
                a=ids[i],
                b=ids[j],
                mean_diff=s["dbar"],
                se=s["se"],
                z=s["z"],
                p_raw=s["p"],
                p_holm=padj,
                ci95=s["ci"],
                mdd=s["mdd"],
                n_items=s["n"],
                lead=lead,
                verdict=verdict,
            )
        )
    return RankResult(pairs=pairs, alpha=alpha, n_models=m, n_items=log.n_items)
