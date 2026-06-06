"""G1-G8 sensitivity gates -- benchgauge's lifeline.

Each gate injects known ground truth into a synthetic matrix and checks that the
analysis recovers it. If the suite does not pass, real-log analysis should be
considered unreliable (the CLI blocks it fail-closed). This module contains NO
random calls; all stochasticity lives in ``selfcheck.synth`` behind integer
seeds, so gate outcomes are reproducible and carry no random placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchgauge.irt import item_report, mislabel_pointbiserial
from benchgauge.metrology import resolution
from benchgauge.model import EvalLog
from benchgauge.rankstability import rank
from benchgauge.rankstability.paired import pair_stats
from benchgauge.results import DISTINGUISHABLE
from benchgauge.selfcheck import synth


@dataclass(frozen=True)
class GateOutcome:
    name: str
    passed: bool
    summary: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "summary": self.summary}


def _log(scores) -> EvalLog:
    return EvalLog.from_matrix(scores)


def g1_zero_resolution(reps=40, base=1000) -> GateOutcome:
    hits = 0
    for r in range(reps):
        s = synth.scenario_null(base + r)["scores"]
        log = _log(s)
        if resolution(rank(log), log).ndc < 2:
            hits += 1
    rate = hits / reps
    return GateOutcome(
        "G1_zero_resolution", rate >= 0.95, f"ndc<2 in {rate:.0%} of reps (need >=95%)"
    )


def g2_high_resolution(reps=40, base=2000) -> GateOutcome:
    thetas = np.linspace(-2.5, 2.5, 6)
    ndc_ok = 0
    powers = []
    for r in range(reps):
        s = synth.scenario_separated(base + r, thetas)["scores"]
        log = _log(s)
        rr = rank(log)
        if resolution(rr, log).ndc >= 5:
            ndc_ok += 1
        powers.append(rr.n_established / len(rr.pairs))
    rate = ndc_ok / reps
    power = float(np.mean(powers))
    return GateOutcome(
        "G2_high_resolution",
        rate >= 0.9 and power >= 0.8,
        f"ndc>=5 in {rate:.0%} (need >=90%), pair power {power:.0%} (need >=80%)",
    )


def g3_dead_item(reps=12, base=3000, k=5) -> GateOutcome:
    recalls, fprs = [], []
    for r in range(reps):
        sc = synth.scenario_dead(base + r, n_models=90, n_items=50, k=k)
        dead = {f"item_{c}" for c in sc["dead_cols"]}
        log = _log(sc["scores"])
        flagged = set(item_report(log).dead_items)
        recalls.append(len(flagged & dead) / len(dead))
        fprs.append(len(flagged - dead) / (log.n_items - len(dead)))
    mr, mf = float(np.mean(recalls)), float(np.mean(fprs))
    return GateOutcome(
        "G3_dead_item",
        mr >= 0.8 and mf <= 0.1,
        f"recall {mr:.0%} (need >=80%), FPR {mf:.0%} (need <=10%)",
    )


def g4_mislabel(reps=15, base=4000, k=4) -> GateOutcome:
    recalls, fprs = [], []
    for r in range(reps):
        sc = synth.scenario_mislabel(base + r, n_models=90, n_items=50, k=k)
        mis = {f"item_{c}" for c in sc["mis_cols"]}
        log = _log(sc["scores"])
        flags, _ = mislabel_pointbiserial(log)
        flagged = set(flags)
        recalls.append(len(flagged & mis) / len(mis))
        fprs.append(len(flagged - mis) / (log.n_items - len(mis)))
    mr, mf = float(np.mean(recalls)), float(np.mean(fprs))
    return GateOutcome(
        "G4_mislabel",
        mr >= 0.8 and mf <= 0.05,
        f"recall {mr:.0%} (need >=80%), FP {mf:.1%} (need <=5%)",
    )


def g5_indistinguishable(reps=500, base=5000) -> GateOutcome:
    false = 0
    for r in range(reps):
        s = synth.scenario_two_equal(base + r)["scores"]
        if rank(_log(s)).pairs[0].verdict == DISTINGUISHABLE:
            false += 1
    rate = false / reps
    return GateOutcome(
        "G5_indistinguishable",
        rate <= 0.075,
        f"false-lead rate {rate:.1%} (need <=~alpha=5%)",
    )


def g6_saturation(reps=10, base=6000) -> GateOutcome:
    det = 0
    for r in range(reps):
        s = synth.scenario_saturation(base + r, n_models=90, n_items=50)["scores"]
        if item_report(_log(s)).saturation_theta is not None:
            det += 1
    rate = det / reps
    return GateOutcome("G6_saturation", rate >= 0.9, f"saturation detected {rate:.0%} (need >=90%)")


def g7_small_n(base=7000) -> GateOutcome:
    # n=8: a-based flags must be suppressed (posterior too diffuse to trust)
    s = synth.scenario_separated(base, np.linspace(-2, 2, 8))["scores"]
    rep = item_report(_log(s))
    suppressed = rep.dead_items == [] and rep.downgraded is not None
    # n=8 null: family correction must keep false leads at/below alpha
    reps = 200
    false = 0
    for r in range(reps):
        sc = synth.scenario_separated(base + 100 + r, np.zeros(8))["scores"]
        if rank(_log(sc)).n_established > 0:
            false += 1
    fp = false / reps
    return GateOutcome(
        "G7_small_n",
        suppressed and fp <= 0.075,
        f"n=8 a-flags suppressed={suppressed}, null FP {fp:.1%} (need <=~5%)",
    )


def g8_nominal_coverage(reps=1000, base=8000) -> GateOutcome:
    mu = synth.coverage_truth()
    cov = 0
    for r in range(reps):
        s = synth.coverage_sample(base + r)["scores"]
        log = _log(s)
        st = pair_stats(log.scores, log.mask, 0, 1, 0.05)
        if st is None:  # never for a 2-model dense sample, but keep type-safe
            continue
        lo, hi = st["ci"]
        if lo <= mu <= hi:
            cov += 1
    rate = cov / reps
    return GateOutcome(
        "G8_nominal_coverage",
        0.93 <= rate <= 0.97,
        f"95% CI coverage {rate:.1%} (need in [93%,97%])",
    )


ALL_GATES = (
    g1_zero_resolution,
    g2_high_resolution,
    g3_dead_item,
    g4_mislabel,
    g5_indistinguishable,
    g6_saturation,
    g7_small_n,
    g8_nominal_coverage,
)


def run_gates() -> dict:
    outcomes = [g() for g in ALL_GATES]
    return {
        "gates": [o.to_dict() for o in outcomes],
        "all_passed": all(o.passed for o in outcomes),
    }
