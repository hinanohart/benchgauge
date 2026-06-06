"""The sensitivity-gate suite is benchgauge's lifeline; CI must keep it green.

This is the slowest test (it fits girth IRT models on synthetic data several
times) but it is the structural guard against shipping an instrument that
produces pretty figures while distinguishing nothing.
"""

from benchgauge.gate import run_gates


def test_all_sensitivity_gates_pass():
    res = run_gates()
    failed = [g["name"] + ": " + g["summary"] for g in res["gates"] if not g["passed"]]
    assert res["all_passed"], "failed gates -> " + " | ".join(failed)


def test_g8_coverage_in_band():
    # the nominal-coverage gate is the CI-claim lifeline; assert it explicitly
    from benchgauge.gate import g8_nominal_coverage

    o = g8_nominal_coverage()
    assert o.passed, o.summary
