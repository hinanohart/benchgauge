"""Adapter for Open LLM Leaderboard ``details`` parquet files.

Each details parquet is one model on one task, one row per example, with an
``acc``/``acc_norm`` column. We align across files by example index and read
the model id from the parent directory (HF lays these out per-model). Requires
the optional ``[parquet]`` extra (pyarrow). This adapter is secondary; lm_eval
is the primary, dependency-free path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchgauge.errors import AbstainError, InputError
from benchgauge.model import EvalLog, ItemMeta

_METRIC_COLS = ("acc", "acc_norm", "exact_match", "score")


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


class OpenLLMLeaderboardAdapter:
    name = "open_llm_leaderboard"

    def _detail_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix == ".parquet" else []
        if path.is_dir():
            return sorted(path.rglob("*.parquet"))
        return []

    def sniff(self, path: Path) -> bool:
        if not _has_pyarrow():
            return False
        files = self._detail_files(path)
        if not files:
            return False
        import pyarrow.parquet as pq

        for f in files:
            try:
                cols = set(pq.read_schema(f).names)
            except Exception:  # noqa: BLE001 - corrupt parquet => not ours
                return False
            # a benchgauge-native long parquet has model+item+score; route that
            # to native, not here.
            if {"model", "item", "score"}.issubset(cols):
                return False
            if any(c in cols for c in _METRIC_COLS):
                return True
        return False

    def load(self, path: Path) -> EvalLog:
        if not _has_pyarrow():
            raise InputError("open_llm_leaderboard adapter requires the [parquet] extra (pyarrow)")
        import pyarrow.parquet as pq

        path = Path(path)
        root = path if path.is_dir() else path.parent
        files = self._detail_files(path)
        if not files:
            raise InputError(f"no .parquet details files under {path}")
        table: dict[str, dict[str, float]] = {}
        item_meta: dict[str, ItemMeta] = {}
        for f in files:
            t = pq.read_table(f)
            cols = t.column_names
            metric = next((c for c in _METRIC_COLS if c in cols), None)
            if metric is None:
                continue
            d = t.to_pydict()
            model = f.parent.name if f.parent != root else f.stem
            task = f.stem
            vals = d[metric]
            golds = d.get("gold") or d.get("target")
            for idx, v in enumerate(vals):
                if v is None:
                    continue
                try:
                    s = float(v)
                except (TypeError, ValueError):
                    continue
                item_id = f"{task}:{idx}"
                table.setdefault(model, {})[item_id] = s
                if item_id not in item_meta:
                    item_meta[item_id] = ItemMeta(
                        item_id=item_id,
                        source_benchmark=task,
                        gold=str(golds[idx]) if golds is not None else None,
                    )
        if not table:
            raise AbstainError(f"no usable details rows parsed from {path}")
        models = sorted(table)
        items = sorted({iid for row in table.values() for iid in row})
        mi = {m: i for i, m in enumerate(models)}
        ii = {it: j for j, it in enumerate(items)}
        scores = np.zeros((len(models), len(items)), dtype=float)
        mask = np.zeros((len(models), len(items)), dtype=bool)
        for m, row in table.items():
            for it, s in row.items():
                scores[mi[m], ii[it]] = s
                mask[mi[m], ii[it]] = True
        return EvalLog(
            schema_version="evallog/1",
            model_ids=tuple(models),
            item_ids=tuple(items),
            scores=scores,
            mask=mask,
            item_meta=item_meta,
            provenance={
                "adapter": "open_llm_leaderboard",
                "source_path": str(path),
                "sparsity": float(1.0 - mask.sum() / mask.size),
            },
        )
