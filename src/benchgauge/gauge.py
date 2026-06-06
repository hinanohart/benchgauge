"""Instrument-health gauge: resolution (柱①) + IRT item forensics (柱③).

``gauge`` is the one-call "is this benchmark a healthy measuring instrument?"
entry point. It runs rank-stability, derives the resolution (ndc), and -- when
there are enough models -- the IRT item forensics, and folds them into a single
:class:`~benchgauge.results.GaugeResult`. It makes no performance claim; an
"unhealthy" verdict is a diagnostic statement about the benchmark, not a model.
"""

from __future__ import annotations

from benchgauge.errors import AbstainError
from benchgauge.irt.forensics import item_report
from benchgauge.metrology.resolution import resolution
from benchgauge.model import EvalLog
from benchgauge.rankstability.verdict import rank
from benchgauge.results import GaugeResult


def gauge(
    log: EvalLog,
    *,
    alpha: float = 0.05,
    fail_under: int = 2,
    dedup_thresh: float = 0.99,
    with_irt: bool = True,
) -> GaugeResult:
    """Compute the instrument-health summary for an EvalLog.

    Parameters
    ----------
    log : EvalLog
    alpha : float
        Family-wise significance level for rank-stability (Holm-corrected).
    fail_under : int
        An instrument is "healthy" only if it resolves at least this many tiers
        (ndc >= fail_under) and is not saturated.
    dedup_thresh : float
        Score-correlation above which near-duplicate models are merged when
        reporting the effective model count.
    with_irt : bool
        Run the IRT item forensics (柱③). Skipped automatically for small model
        sets inside :func:`item_report`.
    """
    if log.n_models < 2:
        raise AbstainError(f"gauge needs >= 2 models, got {log.n_models}")
    if log.n_items < 1:
        raise AbstainError("gauge needs >= 1 item")

    rr = rank(log, alpha=alpha)
    res = resolution(rr, log, fail_under=fail_under, dedup_thresh=dedup_thresh)
    item = item_report(log) if with_irt else None
    saturated = bool(item is not None and item.saturation_theta is not None)
    healthy = bool(res.ndc >= fail_under and not saturated)
    return GaugeResult(resolution=res, item=item, healthy=healthy)
