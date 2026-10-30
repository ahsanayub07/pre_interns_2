import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import text_profile as tp
import topic_keywords as tk


def hits(text, topic):
    return tp.find_hits(text, tp.compile_topic(tk.TOPICS[topic]))


def test_whole_words_only():
    assert hits("The toil of the boil", "energy") == set()
    assert hits("Oil prices rose", "energy") == {"oil"}


def test_plural_allowed():
    assert hits("Several investors and stocks", "market_stocks") == {"investor", "stock"}


def test_acronym_is_case_sensitive():
    assert hits("He said it was fine", "technology") == set()
    assert hits("New AI chips", "technology") == {"AI", "chip"}


def test_phrase_and_no_double_count():
    assert hits("Federal Reserve raised interest rates; the federal reserve spoke", "market_stocks") \
        == {"federal reserve", "interest rate"}


def test_domain_of():
    assert tp.domain_of("https://www.Wired.com/story/x") == "wired.com"
    assert tp.domain_of("http://bbc.com:80/n") == "bbc.com"
    assert tp.domain_of("junk") == "(unknown)"


def make_csv(tmp_path, n):
    rows = [{"id": i, "url": f"https://a.com/{i}", "source": "a.com", "title": f"t{i}",
             "body": "x" * 50, "published_date": "2026-01-01", "clean_word_count": 9} for i in range(1, n + 1)]
    p = tmp_path / "ml.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_sample_is_exact_and_repeatable(tmp_path):
    p = make_csv(tmp_path, 100)
    a, total = tp.sample_articles(p, 10, 42, 30, 20)
    b, _ = tp.sample_articles(p, 10, 42, 30, 20)
    c, _ = tp.sample_articles(p, 10, 7, 30, 20)
    assert total == 100 and len(a) == 10
    assert list(a["id"]) == list(b["id"])
    assert list(a["id"]) != list(c["id"])


def test_sample_independent_of_chunk_size(tmp_path):
    p = make_csv(tmp_path, 100)
    a, _ = tp.sample_articles(p, 10, 42, 100, 20)
    b, _ = tp.sample_articles(p, 10, 42, 7, 20)
    assert set(a["id"]) == set(b["id"])


def test_profile_counts():
    s = pd.DataFrame({"title": ["Oil surges", "Quiet day"], "body": ["energy and coal and oil", "nothing here"]})
    rows, counters, ntop = tp.profile(s, 100)
    assert rows["energy"]["any_pct"] == 50.0 and rows["energy"]["strong_pct"] == 50.0
    assert ntop == {0: 1, 1: 1}


def test_blockchain_counts_inside_technology():
    s = pd.DataFrame({"title": ["Bitcoin jumps", "Quiet day"], "body": ["ethereum and crypto news", "nothing"]})
    rows, _, _ = tp.profile(s, 100)
    assert rows["technology"]["any_pct"] == 0.0
    assert rows["blockchain_technology"]["any_pct"] == 50.0
    assert rows["technology_incl_blockchain"]["any_pct"] == 50.0


def test_main_survives_a_cp1252_console(tmp_path):
    """Regression: non-English titles crashed printing on a Windows console."""
    import subprocess
    d = tmp_path / "v"; d.mkdir()
    rows = [{"id": i, "url": f"https://a.com/{i}", "source": "a.com", "title": "الأخبار 日本語 — café %d" % i,
             "body": "x" * 50, "published_date": "2026-01-01", "clean_word_count": 9} for i in range(1, 40)]
    pd.DataFrame(rows).to_csv(d / "data_ml_ready.csv", index=False, encoding="utf-8")
    code = ("import sys; sys.path.insert(0, %r); import config; config.VALIDATION_DIR=%r; "
            "config.OUTPUT_DIR=%r; config.SAMPLE_SIZE=30; import text_profile as t; t.main()"
            % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), str(d), str(tmp_path / "out")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, env={**os.environ, "PYTHONIOENCODING": "cp1252"})
    assert r.returncode == 0, r.stderr.decode(errors="replace")[-400:]
    assert "café" in open(tmp_path / "out" / "step4_report.txt", encoding="utf-8").read()


def test_exclude_ids_are_never_drawn(tmp_path):
    p = make_csv(tmp_path, 100)
    dev, _ = tp.sample_articles(p, 30, 42, 25, 20)
    dev_ids = set(dev["id"])
    fresh, total = tp.sample_articles(p, 20, 7, 25, 20, exclude_ids=dev_ids)
    assert total == 100 and len(fresh) == 20 and not (set(fresh["id"]) & dev_ids)
