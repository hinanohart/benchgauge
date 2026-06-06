"""Native (canonical) persistence for EvalLog.

JSON is the default and only hard dependency. A long-format parquet
(model, item, score, observed) is supported when the optional ``[parquet]``
extra (pyarrow) is installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchgauge.errors import InputError
from benchgauge.model import SCHEMA_VERSION, EvalLog, ItemMeta


def save_native(log: EvalLog, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".parquet":
        _save_parquet(log, path)
        return
    payload = {
        "schema_version": log.schema_version,
        "model_ids": list(log.model_ids),
        "item_ids": list(log.item_ids),
        "scores": log.scores.tolist(),
        "mask": log.mask.astype(bool).tolist(),
        "item_meta": {k: vars(v) for k, v in log.item_meta.items()},
        "provenance": log.provenance,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_native(path: str | Path) -> EvalLog:
    path = Path(path)
    if path.suffix == ".parquet":
        return _load_parquet(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InputError(f"{path} is not valid benchgauge JSON: {e}") from e
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise InputError(
            f"{path} is not a benchgauge native log (schema_version != {SCHEMA_VERSION!r})"
        )
    missing = {"model_ids", "item_ids", "scores", "mask"} - set(raw)
    if missing:
        raise InputError(f"{path} native log is missing required keys: {sorted(missing)}")
    meta = {
        k: ItemMeta(**v) if isinstance(v, dict) else ItemMeta(item_id=k)
        for k, v in raw.get("item_meta", {}).items()
    }
    return EvalLog(
        schema_version=raw["schema_version"],
        model_ids=tuple(raw["model_ids"]),
        item_ids=tuple(raw["item_ids"]),
        scores=np.asarray(raw["scores"], dtype=float),
        mask=np.asarray(raw["mask"], dtype=bool),
        item_meta=meta,
        provenance=raw.get("provenance", {}),
    )


def _save_parquet(log: EvalLog, path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised only with [parquet]
        raise InputError("parquet support requires the [parquet] extra (pyarrow)") from e
    rows_model, rows_item, rows_score, rows_obs = [], [], [], []
    for i, mid in enumerate(log.model_ids):
        for j, iid in enumerate(log.item_ids):
            rows_model.append(mid)
            rows_item.append(iid)
            rows_score.append(float(log.scores[i, j]))
            rows_obs.append(bool(log.mask[i, j]))
    table = pa.table(
        {"model": rows_model, "item": rows_item, "score": rows_score, "observed": rows_obs}
    )
    table = table.replace_schema_metadata({"schema_version": SCHEMA_VERSION})
    pq.write_table(table, path)


def _load_parquet(path: Path) -> EvalLog:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover
        raise InputError("parquet support requires the [parquet] extra (pyarrow)") from e
    table = pq.read_table(path)
    cols = set(table.column_names)
    if not {"model", "item", "score"}.issubset(cols):
        raise InputError(f"{path} parquet must have columns model,item,score (got {cols})")
    d = table.to_pydict()
    models = list(dict.fromkeys(d["model"]))
    items = list(dict.fromkeys(d["item"]))
    mi = {m: i for i, m in enumerate(models)}
    ii = {it: j for j, it in enumerate(items)}
    scores = np.zeros((len(models), len(items)), dtype=float)
    mask = np.zeros((len(models), len(items)), dtype=bool)
    obs = d.get("observed", [True] * len(d["model"]))
    for m, it, s, o in zip(d["model"], d["item"], d["score"], obs, strict=True):
        scores[mi[m], ii[it]] = float(s)
        mask[mi[m], ii[it]] = bool(o)
    return EvalLog.from_matrix(
        scores,
        model_ids=tuple(models),
        item_ids=tuple(items),
        mask=mask,
        provenance={"adapter": "native_parquet", "source_path": str(path)},
    )
