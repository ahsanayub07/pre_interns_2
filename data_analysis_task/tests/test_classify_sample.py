import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classify_sample as cs

FILL = " The report was published on Monday and the details were shared with the public by the team after a long review of the facts."


def frame():
    return pd.DataFrame({
        "id": ["1", "2", "3"], "domain": ["a.com"] * 3, "published_date": ["2026-01-01"] * 3,
        "title": ["Oil prices rise as OPEC cuts output", "=cmd() quiet day", "Short"],
        "body": ["Crude and natural gas supply tightened." + FILL, "Nothing special." + FILL, "x"]})


def test_classify_and_summarise():
    df = cs.classify_frame(frame())
    s = cs.summarise(df)
    assert s["n"] == 3 and int(s["dist"].sum()) == 3
    assert df.loc[0, "category"] == "energy" and df.loc[2, "category"] == "uncertain"
    assert s["uncertain_reasons"].sum() == 1


def test_safe_cell():
    assert cs.safe_cell("=SUM(A1)") == "'=SUM(A1)" and cs.safe_cell("ok") == "ok" and cs.safe_cell(None) == ""


def test_compare_with_previous(tmp_path):
    df = cs.classify_frame(frame())
    prev = df[["id", "category"]].copy()
    prev.loc[prev["id"] == "1", "category"] = "technology"          # pretend the old run said technology
    path = tmp_path / "prev.csv"
    prev.to_csv(path, index=False)
    table, changed, both = cs.compare_with_previous(df, str(path))
    assert (changed, both) == (1, 3) and table.loc["technology", "energy"] == 1
