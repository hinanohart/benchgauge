"""Parquet round-trip and the open_llm_leaderboard adapter (optional [parquet])."""

import numpy as np
import pytest

from benchgauge.ingest import load_evallog
from benchgauge.ingest.native import load_native, save_native
from benchgauge.model import EvalLog

pytest.importorskip("pyarrow")


def test_native_parquet_round_trip(tmp_path):
    log = EvalLog.from_matrix(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        model_ids=("m1", "m2"),
        item_ids=("i1", "i2", "i3"),
    )
    p = tmp_path / "log.parquet"
    save_native(log, p)
    back = load_native(p)
    assert back.model_ids == log.model_ids
    np.testing.assert_array_equal(back.scores, log.scores)
    np.testing.assert_array_equal(back.mask, log.mask)


def test_native_parquet_autodispatch(tmp_path):
    log = EvalLog.from_matrix([[1, 0], [0, 1]])
    p = tmp_path / "x.parquet"
    save_native(log, p)
    back = load_evallog(p)
    assert back.n_models == 2


def test_open_llm_leaderboard_adapter(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # two "models" as per-model details parquets with an `acc` column
    for name, accs in (("modelA", [1, 1, 0, 1]), ("modelB", [1, 0, 0, 0])):
        d = tmp_path / name
        d.mkdir()
        pq.write_table(pa.table({"acc": [float(a) for a in accs]}), d / "arc.parquet")
    log = load_evallog(tmp_path, adapter="open_llm_leaderboard")
    assert log.n_models == 2
    assert log.n_items == 4
    assert log.scores[0].tolist() == [1.0, 1.0, 0.0, 1.0]
