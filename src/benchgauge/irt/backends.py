"""IRT backends behind a small Protocol so the torch-free core (girth) and the
optional Bayesian backend (py-irt, in the ``[irt-bayes]`` extra) are
interchangeable. Only girth is imported by the core install.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IRTBackend(Protocol):
    name: str

    def fit_2pl(self, data: np.ndarray) -> dict:
        """Fit a 2PL model. ``data`` is (n_items, n_persons) of 0/1.
        Returns {a:(n_items,), b:(n_items,), theta:(n_persons,), converged:bool}."""
        ...


class GirthBackend:
    name = "girth"

    def fit_2pl(self, data: np.ndarray) -> dict:
        from girth import ability_eap, twopl_mml

        out = twopl_mml(data)
        a = np.asarray(out["Discrimination"], dtype=float)
        b = np.asarray(out["Difficulty"], dtype=float)
        theta = np.asarray(ability_eap(data, out["Difficulty"], out["Discrimination"]), dtype=float)
        converged = bool(
            np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(theta))
        )
        return {"a": a, "b": b, "theta": theta, "converged": converged}


def default_backend() -> IRTBackend:
    return GirthBackend()
