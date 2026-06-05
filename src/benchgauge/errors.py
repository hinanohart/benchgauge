"""Exception hierarchy and exit-code contract for benchgauge.

Exit codes (see cli.py):
    0  instrument healthy / analysis succeeded
    1  instrument unhealthy (low resolution or unstable rank) -- a diagnostic
       finding, NOT a performance claim
    2  input error (bad file, malformed matrix)
    3  abstain (too few models/items, all-missing, IRT non-convergence) -- we
       decline to guess rather than emit a misleading number
"""

from __future__ import annotations


class BenchgaugeError(Exception):
    """Base class for all benchgauge errors."""

    exit_code = 2


class InputError(BenchgaugeError):
    """Malformed input / file that cannot be parsed into an EvalLog."""

    exit_code = 2


class AbstainError(BenchgaugeError):
    """We decline to analyse: too few models/items, all-missing, or the model
    could not be fit. Abstaining is a first-class, honest outcome."""

    exit_code = 3


class GateFailure(BenchgaugeError):
    """A sensitivity gate (G1-G8) failed to recover injected ground truth.
    Real-log analysis is blocked fail-closed when this happens."""

    exit_code = 1
