"""Ingest entry point: ``load_evallog(path, adapter=None)``.

Dispatch order:
    1. explicit ``adapter=`` name -> use it,
    2. native benchgauge JSON/parquet (by extension + schema check),
    3. sniff registered adapters; exactly one match -> use it,
       zero or many matches -> abstain (we do not guess).
"""

from __future__ import annotations

import json
from pathlib import Path

from benchgauge.errors import AbstainError, InputError
from benchgauge.ingest.adapters import ADAPTERS, get_adapter
from benchgauge.ingest.native import load_native, save_native
from benchgauge.model import SCHEMA_VERSION, EvalLog

__all__ = ["load_evallog", "save_native", "load_native"]


def _looks_native_json(path: Path) -> bool:
    # Parse and inspect the schema_version key rather than substring-matching a
    # byte window: a foreign JSON that merely contains "evallog/1" early (or a
    # native log whose keys are reordered) must not be mis-routed.
    if path.suffix != ".json":
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(obj, dict) and str(obj.get("schema_version", "")).startswith("evallog")


def load_evallog(path: str | Path, adapter: str | None = None) -> EvalLog:
    path = Path(path)
    if not path.exists():
        raise InputError(f"path does not exist: {path}")

    if adapter is not None:
        return get_adapter(adapter).load(path)

    # a .json file is meant to be a native benchgauge log; failing to parse it is
    # an input error (exit 2), not an abstention (exit 3).
    if path.is_file() and path.suffix == ".json":
        if _looks_native_json(path):
            return load_native(path)
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise InputError(f"{path}: not valid JSON ({e})") from e
        raise InputError(
            f"{path}: valid JSON but not a benchgauge native log "
            f"(missing schema_version {SCHEMA_VERSION!r})"
        )
    if path.suffix == ".parquet" and _maybe_native_parquet(path):
        return load_native(path)

    matches = [a for a in ADAPTERS if a.sniff(path)]
    if len(matches) == 1:
        return matches[0].load(path)
    if not matches:
        raise AbstainError(
            f"no adapter recognised {path}; pass --adapter explicitly "
            f"(known: {[a.name for a in ADAPTERS]})"
        )
    raise AbstainError(
        f"ambiguous input {path}: matched {[a.name for a in matches]}; pass --adapter"
    )


def _maybe_native_parquet(path: Path) -> bool:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return False
    try:
        cols = set(pq.read_schema(path).names)
    except Exception:  # noqa: BLE001
        return False
    return {"model", "item", "score"}.issubset(cols)
