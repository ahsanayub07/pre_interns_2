import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import domain_analysis as da


def test_pct_basic():
    assert da.pct(1, 4) == 25.0
    assert da.pct(1, 3) == 33.33


def test_pct_zero_whole():
    assert da.pct(5, 0) == 0.0


def test_write_csv_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(da.config, "OUTPUT_DIR", str(tmp_path))
    p = da.write_csv("x.csv", ["a", "b"], [[1, 2]])
    assert open(p).read().splitlines() == ["a,b", "1,2"]
    assert not os.path.exists(p + ".partial")
