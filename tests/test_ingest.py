from pathlib import Path

import numpy as np
import pytest

from benchgauge.errors import AbstainError
from benchgauge.ingest import load_evallog, load_native, save_native
from benchgauge.model import EvalLog

FIX = Path(__file__).parent / "fixtures"


def test_native_round_trip(tmp_path):
    log = EvalLog.from_matrix(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        model_ids=("m1", "m2"),
        item_ids=("i1", "i2", "i3"),
    )
    p = tmp_path / "log.json"
    save_native(log, p)
    back = load_native(p)
    assert back.model_ids == log.model_ids
    assert back.item_ids == log.item_ids
    np.testing.assert_array_equal(back.scores, log.scores)
    np.testing.assert_array_equal(back.mask, log.mask)


def test_native_round_trip_with_missing(tmp_path):
    log = EvalLog.from_matrix([[1.0, np.nan], [0.0, 1.0]])
    p = tmp_path / "log.json"
    save_native(log, p)
    back = load_native(p)
    np.testing.assert_array_equal(back.mask, log.mask)
    assert back.sparsity == log.sparsity


def test_lm_eval_adapter_round_trip():
    log = load_evallog(FIX / "lm_eval", adapter="lm_eval")
    assert log.model_ids == ("modelA", "modelB")
    assert log.n_items == 5
    # modelA accs were [1,1,1,0,1]
    a = log.scores[0]
    assert a.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0]
    b = log.scores[1]
    assert b.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert log.provenance["adapter"] == "lm_eval"


def test_load_evallog_autodispatch_lm_eval():
    log = load_evallog(FIX / "lm_eval")  # no adapter -> sniff
    assert log.n_models == 2


def test_load_evallog_native_autodispatch(tmp_path):
    log = EvalLog.from_matrix([[1, 0], [0, 1]])
    p = tmp_path / "x.json"
    save_native(log, p)
    back = load_evallog(p)
    assert back.n_models == 2


def test_abstain_on_unknown(tmp_path):
    p = tmp_path / "mystery.txt"
    p.write_text("not an eval log", encoding="utf-8")
    with pytest.raises(AbstainError):
        load_evallog(p)
