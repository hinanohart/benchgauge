"""Command-line interface.

    benchgauge rank    INPUT
    benchgauge gauge   INPUT
    benchgauge item    INPUT
    benchgauge report  INPUT
    benchgauge gate
    benchgauge selfcheck
    benchgauge convert INPUT --to OUT

Exit-code contract (see errors.py):
    0  instrument healthy / command succeeded
    1  instrument unhealthy (low resolution / no establishable order) -- a
       diagnostic finding, NOT a performance claim
    2  input error
    3  abstain (too few models/items, all-missing, IRT could not be fit)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchgauge.errors import AbstainError, BenchgaugeError, InputError


def _load(args):
    from benchgauge.ingest import load_evallog

    return load_evallog(args.input, adapter=args.adapter)


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"written: {out}")
    else:
        print(text)


# ---- subcommands ---------------------------------------------------------
def _cmd_rank(args) -> int:
    from benchgauge.rankstability.verdict import rank

    log = _load(args)
    rr = rank(log, alpha=args.alpha)
    if args.format == "json":
        _emit(json.dumps(rr.to_dict(), indent=2, ensure_ascii=False), args.out)
    else:
        lines = [
            f"rank-stability: {rr.n_established}/{len(rr.pairs)} pairs established "
            f"(alpha={rr.alpha}, Holm-corrected)"
        ]
        for p in rr.pairs:
            if p.verdict == "DISTINGUISHABLE":
                lines.append(
                    f"  {p.lead} > {p.b if p.lead == p.a else p.a}  "
                    f"(diff {p.mean_diff:+.3f}, CI [{p.ci95[0]:+.3f},{p.ci95[1]:+.3f}], "
                    f"p_holm={p.p_holm:.3g})"
                )
        if rr.n_established == 0:
            lines.append("  (no pair order could be established at this alpha)")
        _emit("\n".join(lines), args.out)
    return 0 if rr.n_established > 0 else 1


def _cmd_gauge(args) -> int:
    from benchgauge.gauge import gauge

    log = _load(args)
    g = gauge(log, alpha=args.alpha, fail_under=args.fail_under, with_irt=not args.no_irt)
    if args.format == "json":
        _emit(json.dumps(g.to_dict(), indent=2, ensure_ascii=False), args.out)
    else:
        res = g.resolution
        n_total = sum(len(t) for t in res.tiers)
        lines = [
            f"resolution: ndc={res.ndc} ({res.verdict}); "
            f"models={n_total} (effective {res.effective_n_models} after near-dup merge)",
            f"instrument health: {'HEALTHY' if g.healthy else 'UNHEALTHY'} "
            f"(fail-under ndc>={args.fail_under})",
        ]
        if g.item is not None and g.item.irt_backend != "none":
            lines.append(
                f"item forensics: {len(g.item.dead_items)} dead, "
                f"{len(g.item.suspected_mislabel)} suspected mislabels"
                + (", SATURATED" if g.item.saturation_theta is not None else "")
            )
        _emit("\n".join(lines), args.out)
    return 0 if g.healthy else 1


def _cmd_item(args) -> int:
    from benchgauge.irt.forensics import item_report

    log = _load(args)
    if log.n_models < 2:
        raise AbstainError(f"item forensics needs >= 2 models, got {log.n_models}")
    ir = item_report(log)
    if args.format == "json":
        _emit(json.dumps(ir.to_dict(), indent=2, ensure_ascii=False), args.out)
    else:
        lines = [
            f"IRT backend: {ir.irt_backend} (converged={ir.irt_converged}, n_models={ir.n_models})",
            f"dead items (a~0): {len(ir.dead_items)}",
            f"suspected mislabels (point-biserial<0): {len(ir.suspected_mislabel)}",
        ]
        if ir.saturation_theta is not None:
            lines.append(f"SATURATED: information collapses above theta~{ir.saturation_theta:.2f}")
        if ir.downgraded:
            lines.append(f"note: {ir.downgraded}")
        if args.explain:
            if ir.dead_items:
                lines.append("  dead: " + ", ".join(ir.dead_items[:20]))
            if ir.suspected_mislabel:
                lines.append("  mislabel: " + ", ".join(ir.suspected_mislabel[:20]))
        _emit("\n".join(lines), args.out)
    return 0


def _cmd_report(args) -> int:
    from benchgauge.report import ReportCard

    log = _load(args)
    card = ReportCard.from_evallog(
        log, alpha=args.alpha, fail_under=args.fail_under, with_irt=not args.no_irt
    )
    if args.format == "json":
        _emit(json.dumps(card.to_json(), indent=2, ensure_ascii=False), args.out)
    else:
        _emit(card.to_markdown(explain=args.explain), args.out)
    return 0 if card.gauge.healthy else 1


def _cmd_gate(args) -> int:
    from benchgauge.gate import run_gates

    res = run_gates()
    print("benchgauge sensitivity gates (synthetic ground truth):")
    for o in res["gates"]:
        print(f"  [{'PASS' if o['passed'] else 'FAIL'}] {o['name']}: {o['summary']}")
    print(f"all gates passed: {res['all_passed']}")
    return 0 if res["all_passed"] else 1


def _cmd_selfcheck(args) -> int:
    import numpy as np

    from benchgauge.irt.forensics import item_report, mislabel_pointbiserial
    from benchgauge.metrology.resolution import resolution
    from benchgauge.model import EvalLog
    from benchgauge.rankstability.verdict import rank
    from benchgauge.selfcheck import synth

    rows = []
    ok = True

    s = synth.scenario_separated(20260606, np.linspace(-2.5, 2.5, 6))["scores"]
    log = EvalLog.from_matrix(s)
    ndc = resolution(rank(log), log).ndc
    r = ndc >= 5
    ok &= r
    rows.append(("resolution (6 separated models)", f"ndc={ndc} (expect >=5)", r))

    sc = synth.scenario_dead(20260607, k=6)
    dead = {f"item_{c}" for c in sc["dead_cols"]}
    flagged = set(item_report(EvalLog.from_matrix(sc["scores"])).dead_items)
    rec = len(flagged & dead) / len(dead)
    r = rec >= 0.8
    ok &= r
    rows.append(("dead-item recovery", f"recall={rec:.0%} (expect >=80%)", r))

    sc = synth.scenario_mislabel(20260608, k=4)
    mis = {f"item_{c}" for c in sc["mis_cols"]}
    flags, _ = mislabel_pointbiserial(EvalLog.from_matrix(sc["scores"]))
    rec = len(set(flags) & mis) / len(mis)
    r = rec >= 0.8
    ok &= r
    rows.append(("mislabel recovery", f"recall={rec:.0%} (expect >=80%)", r))

    print("benchgauge selfcheck (synthetic ground-truth recovery):")
    for name, stat, passed in rows:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {stat}")
    return 0 if ok else 1


def _cmd_convert(args) -> int:
    from benchgauge.ingest import load_evallog
    from benchgauge.ingest.native import save_native

    log = load_evallog(args.input, adapter=args.adapter)
    save_native(log, args.to)
    print(
        f"converted {args.input} -> {args.to} "
        f"({log.n_models} models x {log.n_items} items, sparsity {log.sparsity:.1%})"
    )
    return 0


# ---- parser --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchgauge",
        description="Is your benchmark a measuring instrument? Offline diagnostic for eval logs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_input(sp, with_irt_flag=True):
        sp.add_argument("input", help="path to an eval log (native JSON/parquet or adapter dir)")
        sp.add_argument(
            "--adapter", default=None, help="force an adapter (lm_eval/open_llm_leaderboard)"
        )
        sp.add_argument("--format", choices=["text", "json"], default="text")
        sp.add_argument("--out", default=None, help="write output to a file instead of stdout")
        sp.add_argument("--alpha", type=float, default=0.05)
        if with_irt_flag:
            sp.add_argument("--no-irt", action="store_true", help="skip IRT item forensics")
        else:
            sp.set_defaults(no_irt=True)

    sp = sub.add_parser("rank", help="pairwise lead verdicts (item-clustered SE + Holm)")
    add_input(sp, with_irt_flag=False)
    sp.set_defaults(func=_cmd_rank)

    sp = sub.add_parser("gauge", help="resolution: how many distinguishable tiers (ndc)?")
    add_input(sp)
    sp.add_argument("--fail-under", type=int, default=2, dest="fail_under")
    sp.set_defaults(func=_cmd_gauge)

    sp = sub.add_parser("item", help="IRT item forensics (dead / mislabel / saturation)")
    add_input(sp)
    sp.add_argument("--explain", action="store_true")
    sp.set_defaults(func=_cmd_item, explain=False, fail_under=2)

    sp = sub.add_parser("report", help="three views as one Markdown + JSON report card")
    add_input(sp)
    sp.add_argument("--fail-under", type=int, default=2, dest="fail_under")
    sp.add_argument("--explain", action="store_true")
    sp.set_defaults(func=_cmd_report)

    sp = sub.add_parser("gate", help="run the G1-G8 sensitivity self-tests")
    sp.set_defaults(func=_cmd_gate)

    sp = sub.add_parser("selfcheck", help="synthetic recovery-rate demonstration")
    sp.set_defaults(func=_cmd_selfcheck)

    sp = sub.add_parser("convert", help="convert an adapter input to native EvalLog")
    sp.add_argument("input")
    sp.add_argument("--to", required=True, help="output path (.json or .parquet)")
    sp.add_argument("--adapter", default=None)
    sp.set_defaults(func=_cmd_convert)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AbstainError as e:
        print(f"benchgauge: abstain: {e}", file=sys.stderr)
        return 3
    except InputError as e:
        print(f"benchgauge: input error: {e}", file=sys.stderr)
        return 2
    except BenchgaugeError as e:
        print(f"benchgauge: error: {e}", file=sys.stderr)
        return int(getattr(e, "exit_code", 2))
    except FileNotFoundError as e:
        print(f"benchgauge: input error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
