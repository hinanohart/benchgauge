"""The Benchmark Report Card: the three views as one Markdown + JSON artefact.

A ReportCard bundles the resolution verdict (柱①), the rank-stability table
(柱②) and the IRT item forensics (柱③) computed from a single EvalLog. The JSON
form (``reportcard/1``) is the machine contract; the Markdown is rendered
deterministically from the same data (no second source of truth, no RNG).
"""

from __future__ import annotations

from dataclasses import dataclass

from benchgauge.irt.forensics import item_report
from benchgauge.metrology.resolution import resolution
from benchgauge.model import EvalLog
from benchgauge.rankstability.verdict import rank
from benchgauge.results import GaugeResult, RankResult

SCHEMA_VERSION = "reportcard/1"


@dataclass(frozen=True)
class ReportCard:
    gauge: GaugeResult
    rank: RankResult
    input_summary: dict
    synthetic: bool

    @classmethod
    def from_evallog(
        cls,
        log: EvalLog,
        *,
        alpha: float = 0.05,
        fail_under: int = 2,
        dedup_thresh: float = 0.99,
        with_irt: bool = True,
        synthetic: bool | None = None,
    ) -> ReportCard:
        rr = rank(log, alpha=alpha)
        res = resolution(rr, log, fail_under=fail_under, dedup_thresh=dedup_thresh)
        item = item_report(log) if with_irt else None
        saturated = bool(item is not None and item.saturation_theta is not None)
        g = GaugeResult(
            resolution=res, item=item, healthy=bool(res.ndc >= fail_under and not saturated)
        )
        if synthetic is None:
            synthetic = bool(log.provenance.get("synthetic", False))
        n_dropped = int(log.provenance.get("n_dropped", int((~log.mask).sum())))
        input_summary = {
            "n_models": log.n_models,
            "n_items": log.n_items,
            "n_dropped": n_dropped,
            "sparsity": round(float(log.sparsity), 4),
            "score_type": "binary" if log.is_binary else "continuous",
        }
        return cls(gauge=g, rank=rr, input_summary=input_summary, synthetic=bool(synthetic))

    # ---- serialisation ---------------------------------------------------
    def to_json(self) -> dict:
        g = self.gauge
        item = g.item
        return {
            "schema_version": SCHEMA_VERSION,
            "input": self.input_summary,
            "metrology": g.resolution.to_dict(),
            "rank_stability": {
                "n_models": self.rank.n_models,
                "n_items": self.rank.n_items,
                "alpha": self.rank.alpha,
                "n_established": self.rank.n_established,
                "pairs": [p.to_dict() for p in self.rank.pairs],
            },
            "item_quality": item.to_dict() if item else None,
            "labels": {
                # analytic clustered SE is the default path => deterministic output
                "determinism": "DETERMINISTIC",
                "uncertainty_reported": True,
                "synthetic": self.synthetic,
            },
            "healthy": g.healthy,
        }

    def to_markdown(self, explain: bool = False) -> str:
        g = self.gauge
        res = g.resolution
        item = g.item
        inp = self.input_summary
        L: list[str] = []
        L.append("# Benchmark Report Card")
        L.append("")
        synth_note = " (synthetic ground truth)" if self.synthetic else ""
        L.append(
            f"- **Input**: {inp['n_models']} models x {inp['n_items']} items"
            f" ({inp['score_type']}, sparsity {inp['sparsity']:.1%}, "
            f"{inp['n_dropped']} cells unobserved){synth_note}"
        )
        health = "HEALTHY" if g.healthy else "UNHEALTHY"
        L.append(f"- **Instrument health**: **{health}**")
        L.append("")

        # 柱① resolution
        L.append(f"## Resolution (ndc) - {res.verdict}")
        L.append(
            f"This benchmark separates the model set into **{res.ndc}** "
            f"distinguishable tier(s) (effective models: {res.effective_n_models})."
        )
        if explain:
            L.append("")
            L.append(f"> {res.note}")
        L.append("")

        # 柱② rank-stability
        L.append("## Rank-stability")
        L.append(
            f"{self.rank.n_established} of {len(self.rank.pairs)} model pairs have a "
            f"statistically established order (Holm-corrected, alpha={self.rank.alpha})."
        )
        shown = [p for p in self.rank.pairs if p.verdict == "DISTINGUISHABLE"][:8]
        if shown:
            L.append("")
            L.append("| leader | over | mean diff | 95% CI | p (Holm) |")
            L.append("|---|---|---|---|---|")
            for p in shown:
                lead = p.lead or "-"
                other = p.b if p.lead == p.a else p.a
                L.append(
                    f"| {lead} | {other} | {p.mean_diff:+.3f} | "
                    f"[{p.ci95[0]:+.3f}, {p.ci95[1]:+.3f}] | {p.p_holm:.3g} |"
                )
        L.append("")

        # 柱③ item quality
        L.append("## Item quality (IRT forensics)")
        if item is None or item.irt_backend == "none":
            why = "too few models" if (item is None or item.n_models < 15) else "IRT not fit"
            L.append(f"_IRT item forensics skipped ({why})._")
            if item is not None and item.suspected_mislabel:
                L.append("")
                L.append(
                    f"Point-biserial mislabel candidates ({len(item.suspected_mislabel)}): "
                    + ", ".join(item.suspected_mislabel[:10])
                )
        else:
            L.append(f"- backend: `{item.irt_backend}` (converged: {item.irt_converged})")
            L.append(f"- dead items (a~0): {len(item.dead_items)}")
            L.append(f"- suspected mislabels: {len(item.suspected_mislabel)}")
            if item.saturation_theta is not None:
                L.append(
                    f"- **saturated**: test information collapses above "
                    f"theta~{item.saturation_theta:.2f}"
                )
            if item.downgraded:
                L.append(f"- note: a-flags downgraded ({item.downgraded})")
            if explain and item.dead_items:
                L.append("")
                L.append("Dead items: " + ", ".join(item.dead_items[:10]))
        L.append("")

        # honest footer
        L.append("---")
        L.append(
            "_Diagnostic only; no performance claim. `ndc` is relative to this model set. "
            "Numbers are measured" + (" from synthetic ground truth." if self.synthetic else ".")
        )
        return "\n".join(L)
