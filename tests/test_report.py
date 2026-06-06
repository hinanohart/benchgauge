import numpy as np

from benchgauge.model import EvalLog
from benchgauge.report import ReportCard
from benchgauge.selfcheck import synth


def test_healthy_report_card():
    s = synth.scenario_separated(1, np.linspace(-2.5, 2.5, 6))["scores"]
    card = ReportCard.from_evallog(EvalLog.from_matrix(s))
    assert card.gauge.healthy
    j = card.to_json()
    assert j["schema_version"] == "reportcard/1"
    assert j["metrology"]["ndc"] >= 5
    assert j["labels"]["determinism"] == "DETERMINISTIC"
    md = card.to_markdown()
    assert "Diagnostic only" in md
    assert "Report Card" in md


def test_unhealthy_report_card():
    s = synth.scenario_null(2)["scores"]
    card = ReportCard.from_evallog(EvalLog.from_matrix(s))
    assert not card.gauge.healthy
    assert card.to_json()["metrology"]["ndc"] < 2


def test_with_irt_false_skips_item():
    s = synth.scenario_separated(3, np.linspace(-2, 2, 6))["scores"]
    card = ReportCard.from_evallog(EvalLog.from_matrix(s), with_irt=False)
    assert card.gauge.item is None
    assert card.to_json()["item_quality"] is None


def test_synthetic_label_flows_through():
    s = synth.scenario_separated(4, np.linspace(-2, 2, 6))["scores"]
    card = ReportCard.from_evallog(EvalLog.from_matrix(s), synthetic=True)
    assert card.to_json()["labels"]["synthetic"] is True
    assert "synthetic ground truth" in card.to_markdown()
