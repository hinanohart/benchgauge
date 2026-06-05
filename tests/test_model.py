import numpy as np
import pytest

from benchgauge.model import EvalLog, ItemMeta


def test_from_matrix_basic():
    log = EvalLog.from_matrix([[1, 0, 1], [0, 1, 1]])
    assert log.n_models == 2
    assert log.n_items == 3
    assert log.mask.all()
    assert log.sparsity == 0.0
    assert log.is_binary


def test_from_matrix_nan_becomes_mask():
    log = EvalLog.from_matrix([[1.0, np.nan], [0.0, 1.0]])
    assert log.mask[0, 1] == False  # noqa: E712
    assert log.mask.sum() == 3
    assert log.sparsity == 0.25  # 1 missing cell out of 4
    # masked score is normalised to 0.0, never NaN
    assert not np.isnan(log.scores).any()


def test_explicit_mask():
    scores = [[1.0, 0.5], [0.2, 0.9]]
    mask = [[True, False], [True, True]]
    log = EvalLog.from_matrix(scores, mask=mask)
    assert log.scores[0, 1] == 0.0  # masked -> normalised
    assert not log.is_binary


def test_frozen_and_readonly():
    import dataclasses

    log = EvalLog.from_matrix([[1, 0]])
    with pytest.raises(ValueError):
        log.scores[0, 0] = 5.0  # array is write-protected
    with pytest.raises(dataclasses.FrozenInstanceError):
        log.model_ids = ("x",)  # frozen dataclass


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        EvalLog.from_matrix([[1, 0], [0, 1]], model_ids=("m", "m"))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        EvalLog(
            schema_version="evallog/1",
            model_ids=("a",),
            item_ids=("x", "y"),
            scores=np.zeros((1, 3)),
            mask=np.ones((1, 3), bool),
        )


def test_nan_at_observed_rejected():
    with pytest.raises(ValueError):
        EvalLog(
            schema_version="evallog/1",
            model_ids=("a",),
            item_ids=("x",),
            scores=np.array([[np.nan]]),
            mask=np.array([[True]]),
        )


def test_item_meta():
    meta = {"item_0": ItemMeta(item_id="item_0", source_benchmark="arc")}
    log = EvalLog.from_matrix([[1, 0]], item_meta=meta)
    assert log.item_meta["item_0"].source_benchmark == "arc"
