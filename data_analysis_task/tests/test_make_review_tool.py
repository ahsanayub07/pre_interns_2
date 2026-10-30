import csv
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_review_tool as mr


def sample(n=12, evil=False):
    titles = [f"Title {i}" for i in range(1, n + 1)]
    if evil:
        titles[0] = '</script><script>document.title="HACKED"</script>'
    return pd.DataFrame({"id": [str(i) for i in range(1, n + 1)], "title": titles,
                         "body": [f"Body text number {i}. " * 30 for i in range(1, n + 1)]})


def test_items_are_shuffled_numbered_and_trimmed():
    a = mr.make_items(sample(), 1)
    b = mr.make_items(sample(), 1)
    c = mr.make_items(sample(), 2)
    assert [i["rid"] for i in a] == list(range(1, 13))
    assert [i["id"] for i in a] == [i["id"] for i in b] and [i["id"] for i in a] != [i["id"] for i in c]
    assert all(len(i["lead"]) <= mr.SHOW_CHARS for i in a)


def test_article_text_cannot_break_out_of_the_script_block():
    page = mr.build_html(mr.make_items(sample(evil=True), 1))
    assert page.count("</script>") == mr.PAGE.count("</script>")      # only the page's own closing tag
    assert "HACKED" in page                                           # still present, but as harmless data


def test_prediction_is_not_in_the_tool():
    page = mr.build_html(mr.make_items(sample(), 1)).lower()
    assert "category_pred" not in page and "predicted" not in page and "confidence" not in page


def test_run_id_is_stable_for_the_same_sample():
    a = mr.build_html(mr.make_items(sample(), 1))
    b = mr.build_html(mr.make_items(sample(), 1))
    assert a == b


def test_tool_works_in_a_real_browser(tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    path = tmp_path / "tool.html"
    path.write_text(mr.build_html(mr.make_items(sample(evil=True), 1)), encoding="utf-8")
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(path.as_uri())
        assert "Article 1 of 12" in page.inner_text("#prog")
        assert page.title() == "Blind labelling tool"                  # the injected script did not run
        page.keyboard.press("2")                                       # technology, then moves on
        page.keyboard.press("6")                                       # other
        assert "Article 3 of 12" in page.inner_text("#prog") and "labelled: 2" in page.inner_text("#prog")
        page.reload()                                                  # progress survives a reload
        assert "labelled: 2" in page.inner_text("#prog")
        page.keyboard.press("ArrowLeft")
        assert "Article 2 of 12" in page.inner_text("#prog")
        with page.expect_download() as dl:
            page.click("#export")
        target = tmp_path / "labels.csv"
        dl.value.save_as(str(target))
        rows = list(csv.DictReader(open(target, encoding="utf-8")))
        assert [r["your_label"] for r in rows] == ["technology", "other"] and len(rows) == 2
        assert set(rows[0]) == {"review_id", "article_id", "your_label"}
        assert not errors
        browser.close()


def test_tag_changes_download_name():
    page = mr.build_html(mr.make_items(sample(), 1), "round2")
    assert 'my_labels_round2.csv' in page and "__TAG__" not in page
    assert 'my_labels.csv' in mr.build_html(mr.make_items(sample(), 1))
