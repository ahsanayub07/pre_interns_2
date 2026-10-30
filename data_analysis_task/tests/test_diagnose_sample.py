import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diagnose_sample as ds


def frame():
    cats = ["other"] * 6 + ["technology"] * 4
    return pd.DataFrame({
        "domain": ["a.com"] * 5 + ["b.com"] * 5, "category": cats,
        "title": ["DJI drone review", "Drone camera review", "Home Co | Company Overview & News", "Quiet day", "Quiet day",
                  "AI chip news", "AI model review", "Chip deals", "AI tools", "AI app"],
        "reason": ["x"] * 4 + ["weak evidence: technology: app, AI (title)"] + ["x"] * 5})


def test_title_words_skips_stopwords_and_site_suffix():
    w = dict(ds.title_words(["The drone review - ZDNET", "Drone camera review of the year"]))
    assert w["drone"] == 2 and "the" not in w and "zdnet" not in w


def test_template_table_counts_phrases():
    rows = {r[0]: r[1] for r in ds.template_table(frame())}
    assert rows["review"] == 3 and rows["company overview"] == 1


def test_domain_table_percentages():
    t = ds.domain_table(frame(), min_articles=5)
    assert t.loc["a.com", "other"] == 100 and t.loc["b.com", "technology"] == 80 and t.loc["a.com", "n"] == 5


def test_weak_evidence_topics():
    topics, words = ds.weak_evidence_topics(frame())
    assert topics["technology"] == 1 and dict(words) == {"app": 1, "AI": 1}
