"""Near-duplicate model detection.

Leaderboards are full of near-identical fine-tunes. If many rows are ~copies,
the spread of model scores is inflated and the resolution (ndc) looks better
than it is. We merge models whose score vectors correlate above a threshold and
report an "effective model count".
"""

from __future__ import annotations

import numpy as np

from benchgauge.model import EvalLog


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())


def effective_models(log: EvalLog, thresh: float = 0.99, min_common: int = 3) -> dict:
    """Return {effective_n, groups (lists of model ids), merged (bool)}."""
    m = log.n_models
    scores, mask = log.scores, log.mask
    uf = _UF(m)
    for i in range(m):
        for j in range(i + 1, m):
            both = mask[i] & mask[j]
            if int(both.sum()) < min_common:
                continue
            a, b = scores[i, both], scores[j, both]
            sa, sb = np.std(a), np.std(b)
            if sa == 0.0 or sb == 0.0:
                if np.array_equal(a, b):
                    uf.union(i, j)
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > thresh:
                uf.union(i, j)
    groups_idx = uf.groups()
    groups = [[log.model_ids[i] for i in g] for g in groups_idx]
    return {
        "effective_n": len(groups_idx),
        "groups": groups,
        "merged": any(len(g) > 1 for g in groups_idx),
    }
