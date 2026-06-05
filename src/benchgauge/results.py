"""Result value-objects and verdict vocabulary (internal namespace).

These are intentionally not exported from the top-level package; only EvalLog,
load_evallog, gauge, rank and ReportCard are public. Everything here is the
structured payload those functions return.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# resolution verdicts
RESOLVED = "RESOLVED"
MARGINAL = "MARGINAL"
INSUFFICIENT = "INSUFFICIENT"
# pairwise rank verdicts
DISTINGUISHABLE = "DISTINGUISHABLE"
NOT_DISTINGUISHABLE = "NOT_DISTINGUISHABLE"
ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class RankPair:
    a: str
    b: str
    mean_diff: float  # mean(score_a - score_b) over commonly-observed items
    se: float
    z: float
    p_raw: float
    p_holm: float
    ci95: tuple[float, float]
    mdd: float  # minimum detectable difference (Miller eq.9, power 0.8)
    n_items: int
    lead: str | None  # id of the leading model, or None if not established
    verdict: str  # DISTINGUISHABLE / NOT_DISTINGUISHABLE / ABSTAIN

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "mean_diff": self.mean_diff,
            "se": self.se,
            "z": self.z,
            "p_raw": self.p_raw,
            "p_holm": self.p_holm,
            "ci95": list(self.ci95),
            "mdd": self.mdd,
            "n_items": self.n_items,
            "lead": self.lead,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class RankResult:
    pairs: list[RankPair]
    alpha: float
    n_models: int
    n_items: int

    @property
    def n_established(self) -> int:
        return sum(1 for p in self.pairs if p.verdict == DISTINGUISHABLE)

    @property
    def establishes_any_order(self) -> bool:
        return self.n_established > 0

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "n_models": self.n_models,
            "n_items": self.n_items,
            "n_established": self.n_established,
            "pairs": [p.to_dict() for p in self.pairs],
        }


@dataclass(frozen=True)
class ResolutionResult:
    ndc: int  # number of distinguishable performance tiers (relative to model set)
    verdict: str  # RESOLVED / MARGINAL / INSUFFICIENT
    tiers: list[list[str]]  # model ids grouped by tier
    effective_n_models: int  # after near-duplicate (corr>thresh) merge
    note: str

    def to_dict(self) -> dict:
        return {
            "ndc": self.ndc,
            "resolution_verdict": self.verdict,
            "tiers": self.tiers,
            "effective_n_models": self.effective_n_models,
            "note": self.note,
        }


@dataclass(frozen=True)
class ItemReport:
    dead_items: list[str]
    suspected_mislabel: list[str]
    saturation_theta: float | None
    irt_backend: str
    irt_converged: bool
    n_models: int
    downgraded: str | None  # None / "a_flag_suppressed" / "theta_only" / "skipped"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dead_items": self.dead_items,
            "suspected_mislabel": self.suspected_mislabel,
            "saturation_theta": self.saturation_theta,
            "irt_backend": self.irt_backend,
            "irt_converged": self.irt_converged,
            "n_models": self.n_models,
            "downgraded": self.downgraded,
        }


@dataclass(frozen=True)
class GaugeResult:
    resolution: ResolutionResult
    item: ItemReport | None
    healthy: bool

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution.to_dict(),
            "item": self.item.to_dict() if self.item else None,
            "healthy": self.healthy,
        }
