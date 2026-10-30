import os, sys
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import score_review as sr

FILL = " The report was published on Monday and the details were shared with the public by the team after a long review of the facts."


def files(tmp_path, labels, same_ids=True):
    smp = pd.DataFrame({"review_id": ["1", "2", "3"], "id": ["10", "20", "30"], "domain": ["a.com"] * 3,
                        "title": ["Oil prices rise as OPEC cuts output", "=boom quiet day", "Shop page"],
                        "lead": ["Crude and natural gas supply tightened." + FILL, "Nothing special." + FILL, "x"]})
    lab = pd.DataFrame({"review_id": ["1", "2", "3"], "article_id": ["10", "20", "30" if same_ids else "99"],
                        "your_label": labels})
    s, l = tmp_path / "s.csv", tmp_path / "l.csv"
    smp.to_csv(s, index=False); lab.to_csv(l, index=False)
    return str(l), str(s)


def test_end_to_end(tmp_path):
    l, s = files(tmp_path, ["energy", "other", "not_news"])
    sr.main(["--labels", l, "--sample", s, "--out", str(tmp_path / "o")])
    rep = (tmp_path / "o" / "step9_report.txt").read_text(encoding="utf-8")
    assert "2 of 2 correct" in rep and "not_news: 1" in rep
    det = pd.read_csv(tmp_path / "o" / "score_details.csv", dtype=str)
    assert det["title"].str.startswith("=").sum() == 0      # CSV-injection guard


def test_bad_label_and_mismatch(tmp_path):
    l, s = files(tmp_path, ["energy", "sports", "other"])
    with pytest.raises(SystemExit):
        sr.load_inputs(l, s)
    l, s = files(tmp_path, ["energy", "other", "other"], same_ids=False)
    with pytest.raises(SystemExit):
        sr.load_inputs(l, s)


def test_wilson():
    lo, hi = sr.wilson(80, 100)
    assert 0.70 < lo < 0.80 < hi < 0.88
