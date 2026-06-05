"""EvalLog -- the canonical intermediate representation.

A benchmark run is modelled as a (models x items) matrix of scores plus a
boolean observation mask. Sparsity / missingness is a first-class citizen:
``scores`` always holds numbers, ``mask`` records whether each cell was
actually observed. We never encode "missing" as NaN inside ``scores`` once an
EvalLog is constructed -- the two-channel representation avoids the classic
"is this 0 a real zero or a hole?" ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "evallog/1"


@dataclass(frozen=True)
class ItemMeta:
    """Optional per-item provenance. All fields are optional so callers can
    build an EvalLog from a bare matrix."""

    item_id: str
    source_benchmark: str | None = None
    release_date: str | None = None
    subtask: str | None = None
    gold: str | None = None


@dataclass(frozen=True)
class EvalLog:
    """Immutable (models x items) evaluation log.

    Attributes
    ----------
    schema_version : str
        Always ``"evallog/1"`` for this release.
    model_ids : tuple[str, ...]
        Row labels, length ``M``. Must be unique.
    item_ids : tuple[str, ...]
        Column labels, length ``N``. Must be unique.
    scores : np.ndarray
        ``(M, N)`` float array. Binary {0,1} or continuous [0,1]. Cells where
        ``mask`` is False are not meaningful (filled with 0.0 by convention).
    mask : np.ndarray
        ``(M, N)`` bool array. True = observed.
    item_meta : dict[str, ItemMeta]
        Optional, keyed by item_id.
    provenance : dict
        Free-form ingest provenance, e.g. ``{"adapter", "source_path",
        "ingested_at", "n_dropped", "sparsity"}``.
    """

    schema_version: str
    model_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    scores: np.ndarray
    mask: np.ndarray
    item_meta: dict[str, ItemMeta] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        m, n = len(self.model_ids), len(self.item_ids)
        if len(set(self.model_ids)) != m:
            raise ValueError("model_ids must be unique")
        if len(set(self.item_ids)) != n:
            raise ValueError("item_ids must be unique")

        scores = np.asarray(self.scores, dtype=float)
        mask = np.asarray(self.mask, dtype=bool)
        if scores.shape != (m, n):
            raise ValueError(f"scores shape {scores.shape} != (n_models={m}, n_items={n})")
        if mask.shape != (m, n):
            raise ValueError(f"mask shape {mask.shape} != (n_models={m}, n_items={n})")
        if np.isnan(scores[mask]).any():
            raise ValueError("scores contain NaN at observed (mask=True) cells")
        # masked-out cells are normalised to 0.0 so downstream code never trips on NaN
        scores = np.where(mask, scores, 0.0)
        scores.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "mask", mask)

    # ---- convenience -----------------------------------------------------
    @property
    def n_models(self) -> int:
        return len(self.model_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def sparsity(self) -> float:
        """Fraction of cells that are NOT observed."""
        total = self.scores.size
        return float(1.0 - self.mask.sum() / total) if total else 0.0

    @property
    def is_binary(self) -> bool:
        obs = self.scores[self.mask]
        if obs.size == 0:
            return True
        return bool(np.all(np.isin(obs, (0.0, 1.0))))

    @classmethod
    def from_matrix(
        cls,
        scores,
        model_ids=None,
        item_ids=None,
        mask=None,
        item_meta=None,
        provenance=None,
    ) -> EvalLog:
        """Build an EvalLog from a raw matrix.

        NaN cells (when ``mask`` is not given) are treated as unobserved.
        """
        scores = np.asarray(scores, dtype=float)
        if scores.ndim != 2:
            raise ValueError(f"scores must be 2-D (models x items), got ndim={scores.ndim}")
        m, n = scores.shape
        if model_ids is None:
            model_ids = tuple(f"model_{i}" for i in range(m))
        if item_ids is None:
            item_ids = tuple(f"item_{j}" for j in range(n))
        model_ids = tuple(model_ids)
        item_ids = tuple(item_ids)
        if mask is None:
            mask = ~np.isnan(scores)
        else:
            mask = np.asarray(mask, dtype=bool)
        n_dropped = int((~mask).sum())
        prov = dict(provenance or {})
        prov.setdefault("adapter", "from_matrix")
        prov.setdefault("n_dropped", n_dropped)
        prov.setdefault("sparsity", float(1.0 - mask.sum() / mask.size) if mask.size else 0.0)
        return cls(
            schema_version=SCHEMA_VERSION,
            model_ids=model_ids,
            item_ids=item_ids,
            scores=np.where(mask, np.nan_to_num(scores, nan=0.0), 0.0),
            mask=mask,
            item_meta=dict(item_meta or {}),
            provenance=prov,
        )
