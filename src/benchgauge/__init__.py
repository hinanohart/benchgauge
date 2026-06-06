"""benchgauge — a measuring-instrument audit for evaluation logs.

Public API (kept deliberately small):

    from benchgauge import EvalLog, load_evallog, gauge, rank, ReportCard

benchgauge is a *diagnostic* tool. It makes no claim about which model is
"better" in any absolute sense; it reports whether a benchmark can statistically
tell models apart on the items it actually contains, and which items behave as
dead weight or look mislabelled. All numbers are either measured from the log
you pass in or generated from synthetic ground truth (machine-labelled).
"""

__version__ = "0.1.0a3"

from benchgauge.model import EvalLog, ItemMeta  # noqa: E402

__all__ = [
    "__version__",
    "EvalLog",
    "ItemMeta",
    "load_evallog",
    "gauge",
    "rank",
    "ReportCard",
]


def __getattr__(name: str):  # lazy to keep import light and avoid cycles
    if name == "load_evallog":
        from benchgauge.ingest import load_evallog

        return load_evallog
    if name == "gauge":
        from benchgauge.gauge import gauge

        return gauge
    if name == "rank":
        from benchgauge.rankstability.verdict import rank

        return rank
    if name == "ReportCard":
        from benchgauge.report import ReportCard

        return ReportCard
    raise AttributeError(f"module 'benchgauge' has no attribute {name!r}")
