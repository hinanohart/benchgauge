from pathlib import Path

import numpy as np

from benchgauge.cli import main
from benchgauge.ingest.native import save_native
from benchgauge.model import EvalLog

FIX = Path(__file__).parent / "fixtures"


def test_report_healthy_exit0():
    assert main(["report", str(FIX / "healthy.json"), "--no-irt"]) == 0


def test_gauge_unhealthy_exit1():
    assert main(["gauge", str(FIX / "unhealthy.json"), "--no-irt"]) == 1


def test_rank_json_output(capsys):
    rc = main(["rank", str(FIX / "healthy.json"), "--format", "json"])
    assert rc in (0, 1)
    assert '"pairs"' in capsys.readouterr().out


def test_item_runs_exit0():
    assert main(["item", str(FIX / "healthy.json")]) == 0


def test_report_markdown_has_diagnostic(capsys):
    main(["report", str(FIX / "healthy.json"), "--no-irt"])
    assert "Diagnostic only" in capsys.readouterr().out


def test_abstain_single_model_exit3(tmp_path):
    p = tmp_path / "one.json"
    save_native(EvalLog.from_matrix(np.array([[1.0, 0.0, 1.0, 0.0]])), p)
    assert main(["report", str(p)]) == 3


def test_missing_input_exit2():
    assert main(["report", "/no/such/benchgauge_xyz.json"]) == 2


def test_convert_round_trip(tmp_path):
    out = tmp_path / "out.json"
    assert main(["convert", str(FIX / "healthy.json"), "--to", str(out)]) == 0
    assert out.exists()
