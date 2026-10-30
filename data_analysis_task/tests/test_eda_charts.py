import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eda_charts as ec


def test_concentration_finds_half(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "CHARTS", str(tmp_path))
    dom = pd.DataFrame({"domain": list("abcd"), "articles": [50, 30, 10, 10],
                        "cumulative_share_pct": [50.0, 80.0, 90.0, 100.0]})
    path, half = ec.concentration(dom)
    assert half == 1 and os.path.exists(path)


def test_need_reports_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "OUT", str(tmp_path))
    try:
        ec.need("nope.csv")
        assert False
    except SystemExit as e:
        assert "domain_analysis.py" in str(e)
