"""Adapter for EleutherAI lm-evaluation-harness ``--log_samples`` output.

lm-eval writes one JSONL file per (model, task), with one line per example.
Each line carries a ``doc_id`` and one or more metric fields (``acc``,
``exact_match``, ``acc_norm`` ...). We align examples across models by
``"{task}:{doc_id}"`` and build a (models x items) correctness matrix.

Model id resolution order:
    1. an explicit ``model``/``model_name`` field on the line, else
    2. the immediate parent directory name (lm-eval's per-model subdir), else
    3. the filename token before ``__`` / ``samples_``.
Task resolution order:
    1. an explicit ``task``/``task_name`` field on the line, else
    2. parsed from the filename (``samples_<task>_<timestamp>.jsonl``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from benchgauge.errors import AbstainError, InputError
from benchgauge.model import EvalLog, ItemMeta

_METRIC_KEYS = ("acc", "exact_match", "acc_norm", "em", "correct", "score")
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T")


def _task_from_filename(fname: str) -> str:
    stem = fname[:-6] if fname.endswith(".jsonl") else fname
    if stem.startswith("samples_"):
        stem = stem[len("samples_") :]
    head, _, tail = stem.rpartition("_")
    if head and _TS.match(tail):
        return head
    return stem


def _model_from_path(path: Path, root: Path) -> str:
    parent = path.parent
    if parent != root and parent.name:
        return parent.name
    name = path.name
    if "__" in name:
        return name.split("__", 1)[0]
    if name.startswith("samples_"):
        return root.name or "model"
    return name[:-6] if name.endswith(".jsonl") else name


def _score_from_line(line: dict) -> float | None:
    for k in _METRIC_KEYS:
        if k in line and line[k] is not None:
            v = line[k]
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


class LmEvalAdapter:
    name = "lm_eval"

    def _sample_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix == ".jsonl" else []
        if path.is_dir():
            return sorted(path.rglob("*.jsonl"))
        return []

    def sniff(self, path: Path) -> bool:
        files = self._sample_files(path)
        if not files:
            return False
        for f in files:
            try:
                with f.open(encoding="utf-8") as fh:
                    first = fh.readline()
                if not first.strip():
                    continue
                line = json.loads(first)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                return False
            if isinstance(line, dict) and "doc_id" in line and _score_from_line(line) is not None:
                return True
        return False

    def load(self, path: Path) -> EvalLog:
        path = Path(path)
        root = path if path.is_dir() else path.parent
        files = self._sample_files(path)
        if not files:
            raise InputError(f"no lm-eval *.jsonl sample files found under {path}")
        # model -> {item_id: score}
        table: dict[str, dict[str, float]] = {}
        item_meta: dict[str, ItemMeta] = {}
        n_lines = n_dropped = 0
        for f in files:
            task_fn = _task_from_filename(f.name)
            with f.open(encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    n_lines += 1
                    try:
                        ln = json.loads(raw)
                    except json.JSONDecodeError:
                        n_dropped += 1
                        continue
                    if not isinstance(ln, dict) or "doc_id" not in ln:
                        n_dropped += 1
                        continue
                    score = _score_from_line(ln)
                    if score is None:
                        n_dropped += 1
                        continue
                    model = str(
                        ln.get("model") or ln.get("model_name") or _model_from_path(f, root)
                    )
                    task = str(ln.get("task") or ln.get("task_name") or task_fn)
                    item_id = f"{task}:{ln['doc_id']}"
                    table.setdefault(model, {})[item_id] = float(score)
                    if item_id not in item_meta:
                        item_meta[item_id] = ItemMeta(
                            item_id=item_id,
                            source_benchmark=task,
                            gold=str(ln["target"]) if "target" in ln else None,
                        )
        if not table:
            raise AbstainError(f"no usable lm-eval samples parsed from {path}")
        models = sorted(table)
        items = sorted({iid for row in table.values() for iid in row})
        if len(models) < 1 or len(items) < 1:
            raise AbstainError("lm-eval log produced an empty matrix")
        mi = {m: i for i, m in enumerate(models)}
        ii = {it: j for j, it in enumerate(items)}
        scores = np.zeros((len(models), len(items)), dtype=float)
        mask = np.zeros((len(models), len(items)), dtype=bool)
        for m, row in table.items():
            for it, s in row.items():
                scores[mi[m], ii[it]] = s
                mask[mi[m], ii[it]] = True
        sparsity = float(1.0 - mask.sum() / mask.size)
        return EvalLog(
            schema_version="evallog/1",
            model_ids=tuple(models),
            item_ids=tuple(items),
            scores=scores,
            mask=mask,
            item_meta=item_meta,
            provenance={
                "adapter": "lm_eval",
                "source_path": str(path),
                "n_lines": n_lines,
                "n_dropped": n_dropped,
                "sparsity": sparsity,
            },
        )
