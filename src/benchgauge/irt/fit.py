"""Fit a 2PL model to an EvalLog via the active backend.

IRT requires complete data, so when the matrix is sparse we fit on the
complete-column submatrix (items observed by *every* model) and record how many
items were dropped. Continuous scores are thresholded at 0.5 (with a flag).
"""

from __future__ import annotations

import warnings

import numpy as np

from benchgauge.irt.backends import IRTBackend, default_backend
from benchgauge.model import EvalLog

MIN_MODELS_FOR_IRT = 15
MIN_ITEMS_FOR_IRT = 10


def _prepare(log: EvalLog):
    complete = log.mask.all(axis=0)
    used_idx = np.where(complete)[0]
    sub = log.scores[:, used_idx]
    binarised = bool(not log.is_binary)
    binary = (sub >= 0.5).astype(int)
    used_items = tuple(log.item_ids[j] for j in used_idx)
    return binary, used_items, binarised


def fit_irt(log: EvalLog, backend: IRTBackend | None = None) -> dict | None:
    """Return a fit dict, or None if IRT is not applicable / did not converge."""
    backend = backend or default_backend()
    binary, used_items, binarised = _prepare(log)
    if log.n_models < MIN_MODELS_FOR_IRT or len(used_items) < MIN_ITEMS_FOR_IRT:
        return None
    data = binary.T  # items x persons
    # drop degenerate items (all-correct / all-wrong) -- 2PL is undefined there
    keep = ~(np.all(data == data[:, :1], axis=1))
    if keep.sum() < MIN_ITEMS_FOR_IRT:
        return None
    data = data[keep]
    used_items = tuple(it for it, k in zip(used_items, keep, strict=True) if k)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = backend.fit_2pl(data)
    except Exception:  # noqa: BLE001 - any backend failure -> abstain on pillar 3
        return None
    if not res["converged"]:
        return None
    return {
        "a": res["a"],
        "b": res["b"],
        "theta": res["theta"],
        "item_ids": used_items,
        "backend": backend.name,
        "converged": True,
        "n_models": log.n_models,
        "binarised": binarised,
    }
