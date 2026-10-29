import os
import sys
from datetime import date, datetime, time, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from validation_rules import (DuplicateIndex, alnum_ratio, check_content, check_fields,  # noqa: E402
                              check_flags, check_times, check_url, domain_matches,
                              has_errors, is_missing, is_placeholder_title,
                              is_valid_http_url, normalize_url, signature_key,
                              validate_row, validate_schema)
from validation_rules import ACCEPT, QUARANTINE, WARNING, disposition  # noqa: E402
from report_generator import (Outputs, Stats, build_report_row, check_partitions,  # noqa: E402
                              issue_breakdown, ml_exclusion_reasons, safe_cell, write_table)

GOOD = [("id", "bigint"), ("url", "text"), ("title", "text"), ("body", "text"),
        ("source", "character varying"), ("published_date", "date")]
ALL = {"id", "url", "title", "body", "source", "author", "published_date",
       "published_time", "imported_at"}


def rules(issues):
    return [(i["rule_id"], i["severity"], i["field"]) for i in issues]


def ids(issues):
    return [i["rule_id"] for i in issues]


# ---- schema -------------------------------------------------------------------------
def test_complete_required_columns_pass():
    assert not has_errors(validate_schema(GOOD))


def test_missing_optional_columns_are_info_only():
    issues = validate_schema(GOOD)
    assert ("SCHEMA_003", "INFO", "author") in rules(issues)
    assert all(i["severity"] == "INFO" for i in issues)


def test_missing_required_column_is_error():
    issues = validate_schema([c for c in GOOD if c[0] != "title"])
    assert ("SCHEMA_002", "ERROR", "title") in rules(issues)
    assert has_errors(issues)


def test_duplicate_column_names_are_error():
    assert ("SCHEMA_001", "ERROR", "url") in rules(validate_schema(GOOD + [("url", "text")]))


def test_wrong_required_type_is_error_wrong_optional_is_warning():
    cols = [("id", "text") if c[0] == "id" else c for c in GOOD]
    cols = [("published_date", "text") if c[0] == "published_date" else c for c in cols]
    issues = validate_schema(cols)
    assert ("SCHEMA_004", "ERROR", "id") in rules(issues)
    assert ("SCHEMA_004", "WARNING", "published_date") in rules(issues)


def test_empty_table_definition_reports_all_required_missing():
    missing = {i["field"] for i in validate_schema([]) if i["rule_id"] == "SCHEMA_002"}
    assert missing == {"id", "url", "title", "body"}


# ---- missing values and types -------------------------------------------------------
def test_missing_value_detection():
    assert is_missing(None) and is_missing("") and is_missing("   ")
    assert is_missing(float("nan")) and is_missing("​  ")
    assert not is_missing("Title") and not is_missing(0)


def test_blank_required_fields_are_errors():
    row = {"id": 7, "url": "", "title": None, "body": "  ", "author": "A"}
    found = check_fields(row, ALL, 7)
    assert sorted(i["field"] for i in found if i["rule_id"] == "FIELD_001") == \
        ["body", "title", "url"]
    assert all(i["article_id"] == 7 for i in found)


def test_missing_author_is_info_not_error():
    found = check_fields({"id": 1, "url": "u", "title": "t", "body": "b", "author": None},
                         ALL, 1)
    assert rules(found) == [("FIELD_002", "INFO", "author")]


def test_non_text_value_is_type_warning():
    found = check_fields({"id": 1, "url": "u", "title": 123, "body": "b", "author": "a"},
                         ALL, 1)
    assert ("TYPE_001", "WARNING", "title") in rules(found)


def test_unicode_title_is_accepted():
    row = {"id": 1, "url": "u", "title": "Pokémon کی خبریں", "body": "b", "author": "a"}
    assert check_fields(row, ALL, 1) == []


def test_missing_optional_column_skips_rule():
    assert check_fields({"id": 1, "url": "u", "title": "t", "body": "b"},
                        {"id", "url", "title", "body"}, 1) == []


# ---- URLs ---------------------------------------------------------------------------
def test_valid_urls_with_ports_queries_fragments():
    for url in ("https://example.com/story/1", "http://news.example.org/",
                "https://example.com:8443/a/b?x=1&y=2#top", "HTTPS://EXAMPLE.COM/A",
                "https://sub.domain.example.co.uk/p?q=é", "http://192.168.1.10:8080/x"):
        assert is_valid_http_url(url), url


def test_invalid_urls():
    for url in ("not-a-url", "ftp://example.com/file", "http://", "https://localhost/x",
                "https://exa mple.com", "http://example.com:99999/", "//example.com/x",
                " https://example.com", "https://example..com/x", "mailto:a@b.com",
                "", None, 42):
        assert not is_valid_http_url(url), url


def test_invalid_url_is_fatal_error():
    found = check_url({"id": 3, "url": "not-a-url", "source": "x.com"}, ALL, 3)
    assert rules(found) == [("URL_001", "ERROR", "url")]


def test_domain_policy():
    assert domain_matches("bbc.com", "www.bbc.com")
    assert domain_matches("BBC.com", "news.bbc.com")
    assert not domain_matches("bbc.com", "notbbc.com")
    assert not domain_matches("bbc.com", "bbc.com.evil.net")


def test_domain_mismatch_is_warning_and_blank_source_skipped():
    row = {"id": 1, "url": "https://www.a.com/x", "source": "b.com"}
    assert rules(check_url(row, ALL, 1)) == [("URL_002", "WARNING", "source")]
    assert check_url({"id": 1, "url": "https://a.com/x", "source": ""}, ALL, 1) == []
    assert check_url({"id": 1, "url": "https://www.a.com/x", "source": "a.com"}, ALL, 1) == []


def test_blank_url_not_reported_twice():
    assert check_url({"id": 1, "url": " ", "source": "a.com"}, ALL, 1) == []


# ---- timestamps ---------------------------------------------------------------------
IMPORTED = datetime(2026, 10, 28, 6, 0, tzinfo=timezone.utc)
LONG = "The council met on Tuesday to discuss the new budget. " * 8


def times(**kw):
    row = {"id": 1, "published_date": date(2026, 10, 20), "published_time": None,
           "imported_at": IMPORTED}
    row.update(kw)
    return check_times(row, ALL, 1)


def test_normal_dates_have_no_issue():
    assert times() == []


def test_missing_dates_are_info_only():
    found = times(published_date=None, imported_at=None)
    assert sorted(rules(found)) == [("TIME_001", "INFO", "imported_at"),
                                    ("TIME_005", "INFO", "published_date")]


def test_unparseable_values():
    found = times(published_date="yesterday", imported_at="soon")
    assert ids(found) == ["TIME_002", "TIME_002"]


def test_iso_text_is_parsed():
    assert times(published_date="2026-10-20", imported_at="2026-10-28T06:00:00Z") == []


def test_implausibly_old_date():
    assert ids(times(published_date=date(1985, 5, 1))) == ["TIME_004"]


def test_future_date_beyond_tolerance_is_warning():
    assert ids(times(published_date=date(2026, 10, 30))) == ["TIME_003"]
    assert times(published_date=date(2026, 10, 29)) == []        # within 1-day tolerance


def test_timezone_offset_is_respected():
    # 23:30 at -08:00 on Oct 27 = 07:30 UTC on Oct 28, which is after 06:00 UTC import.
    late = time(23, 30, tzinfo=timezone(timedelta(hours=-8)))
    assert ids(times(published_date=date(2026, 10, 27), published_time=late)) == ["TIME_003"]
    # Same wall-clock time at +00:00 is before the import.
    early = time(23, 30, tzinfo=timezone.utc)
    assert times(published_date=date(2026, 10, 27), published_time=early) == []


def test_naive_time_is_not_assumed_utc():
    assert times(published_date=date(2026, 10, 28), published_time=time(23, 0)) == []


# ---- whole row ----------------------------------------------------------------------
def test_validate_row_collects_all_rules():
    row = {"id": 9, "url": "not-a-url", "title": "", "body": LONG, "source": "a.com",
           "author": None, "published_date": None, "published_time": None,
           "imported_at": IMPORTED}
    assert sorted(ids(validate_row(row, ALL))) == \
        ["FIELD_001", "FIELD_002", "TIME_005", "URL_001"]


def test_validate_row_clean_row_has_no_issues():
    row = {"id": 1, "url": "https://a.com/x", "title": "A good headline", "body": LONG,
           "source": "a.com", "author": "Jane", "published_date": date(2026, 10, 1),
           "published_time": None, "imported_at": IMPORTED}
    assert validate_row(row, ALL) == []



# ---- content quality (Step 5) -------------------------------------------------------
def content(**kw):
    row = {"id": 1, "title": "A proper headline here", "body": LONG}
    row.update(kw)
    return check_content(row, {"id", "title", "body"}, 1)


def test_good_content_has_no_issue():
    assert content() == []


def test_short_title_and_short_body_are_warnings():
    assert ids(content(title="Short")) == ["TEXT_001"]
    assert ids(content(body="A brief but possibly legitimate update.")) == ["TEXT_002"]


def test_blank_values_are_left_to_the_field_rule():
    assert content(title="  ", body=None) == []


def test_placeholder_titles():
    for title in ("Home", "Untitled", "Page not found", "Just a moment...", "404", "N/A",
                  "  ACCESS DENIED  ", "!!!???", "-----"):
        assert is_placeholder_title(title), title
    assert not is_placeholder_title("Home prices rise again")
    assert not is_placeholder_title("Pokémon کی خبریں")
    assert "TEXT_004" in ids(content(title="Untitled"))


def test_error_text_in_short_body_is_flagged():
    body = "Access denied. Please enable JavaScript and complete the CAPTCHA to continue. " * 3
    assert "TEXT_003" in ids(content(body=body))
    assert "TEXT_003" in ids(content(body="403 Forbidden. " * 15))


def test_error_words_inside_a_long_article_are_not_flagged():
    body = LONG * 6 + " The old page was reported as page not found by readers."
    assert len(body) > 1500
    assert "TEXT_003" not in ids(content(body=body))


def test_mostly_punctuation_body():
    assert "TEXT_005" in ids(content(body="... --- ... *** ### !!! ??? ... --- ... *** ###"))
    assert alnum_ratio("") == 0.0
    assert alnum_ratio("اردو خبریں 123") > 0.9
    assert "TEXT_005" not in ids(content())


def test_content_rules_skipped_when_columns_absent():
    assert check_content({"id": 1, "title": "x", "body": "y"}, {"id"}, 1) == []


def test_preprocessing_flags_are_reused_as_warnings():
    row = {"id": 5, "is_short": True, "is_non_article": False, "has_html_tags": True,
           "has_boilerplate": None, "has_suspicious_encoding": True}
    found = check_flags(row, set(row), 5)
    assert ids(found) == ["FLAG_001", "FLAG_003", "FLAG_005"]
    assert all(i["severity"] == "WARNING" for i in found)
    assert check_flags(row, {"id"}, 5) == []


# ---- duplicates (Step 5) ------------------------------------------------------------
def test_normalize_url_treats_cosmetic_differences_as_same():
    base = normalize_url("https://example.com/story/1")
    for variant in ("http://www.Example.com/story/1/", "https://example.com/story/1#comments",
                    "https://example.com:443/story/1", "https://example.com/story/1?utm_source=x",
                    "https://example.com/story/1?fbclid=abc"):
        assert normalize_url(variant) == base, variant


def test_normalize_url_keeps_real_differences():
    assert normalize_url("https://example.com/story/1") != normalize_url("https://example.com/story/2")
    assert normalize_url("https://example.com/a?id=1") != normalize_url("https://example.com/a?id=2")
    assert normalize_url("https://example.com/a?x=1&y=2") == normalize_url("https://example.com/a?y=2&x=1")
    assert normalize_url("https://example.com:8080/a") != normalize_url("https://example.com/a")
    assert normalize_url("https://example.com/Story") != normalize_url("https://example.com/story")
    assert normalize_url("not-a-url") is None and normalize_url(None) is None


def test_signature_ignores_case_and_whitespace():
    assert signature_key("Same  Title", "Body\ntext") == signature_key("same title", "body text")
    assert signature_key("A", "x") != signature_key("A", "y")
    assert signature_key(" ", None) is None


def dup_rows():
    return [
        {"id": 1, "url": "https://a.com/x", "title": "Same", "body": "Same body text",
         "body_hash": "h1" * 16, "duplicate_group_size": 2, "is_duplicate": False},
        {"id": 2, "url": "http://www.a.com/x/", "title": "same", "body": "same  body text",
         "body_hash": "h1" * 16, "duplicate_group_size": 2, "is_duplicate": True},
        {"id": 3, "url": "https://b.com/y", "title": "Other title", "body": "Same body text",
         "body_hash": "h1" * 16, "duplicate_group_size": 3, "is_duplicate": True},
        {"id": 4, "url": "https://c.com/z", "title": "Unique", "body": "Unique body",
         "body_hash": "h2" * 16, "duplicate_group_size": 1, "is_duplicate": False},
    ]


def run_index(rows):
    index = DuplicateIndex()
    for r in rows:
        index.add(r)
    return {r["id"]: index.check(r) for r in rows}


def test_duplicate_index_reports_url_and_signature_candidates():
    found = run_index(dup_rows())
    assert ids(found[1]) == ["DUP_001", "DUP_002"]
    assert ids(found[2]) == ["DUP_001", "DUP_002"]
    assert found[1][0]["group"] == found[2][0]["group"]          # same group id
    assert found[4] == []


def test_same_body_with_different_title_is_a_weaker_candidate():
    found = run_index(dup_rows())
    assert ids(found[3]) == ["DUP_003"]
    assert "later copy" in found[3][0]["message"]
    assert found[3][0]["severity"] == "WARNING"


def test_duplicate_rows_are_never_fatal():
    from config import FATAL_RULES
    for issues in run_index(dup_rows()).values():
        assert not any(i["rule_id"] in FATAL_RULES for i in issues)


def test_invalid_urls_and_blank_rows_do_not_form_duplicate_groups():
    rows = [{"id": 1, "url": "not-a-url", "title": "", "body": ""},
            {"id": 2, "url": "not-a-url", "title": " ", "body": None}]
    assert run_index(rows) == {1: [], 2: []}


def test_empty_dataset_and_missing_optional_columns():
    index = DuplicateIndex()
    assert index.check({"id": 1}) == []
    rows = [{"id": 1, "url": "https://a.com/x", "title": "T", "body": "B"}]   # no hash columns
    assert run_index(rows) == {1: []}


# ---- dispositions and reports (Step 6) ---------------------------------------------
def issue(rule, severity="WARNING"):
    return {"rule_id": rule, "severity": severity, "scope": "row", "article_id": 1,
            "field": "x", "message": "m", "group": None}


def test_disposition_accept_warning_quarantine():
    assert disposition([]) == ACCEPT
    assert disposition([issue("FIELD_002", "INFO"), issue("TIME_005", "INFO")]) == ACCEPT
    assert disposition([issue("TEXT_002")]) == WARNING
    assert disposition([issue("DUP_002"), issue("FIELD_002", "INFO")]) == WARNING
    assert disposition([issue("TEXT_002"), issue("URL_001", "ERROR")]) == QUARANTINE
    assert disposition([issue("FIELD_001", "ERROR")]) == QUARANTINE


def test_missing_author_short_content_and_duplicates_are_never_fatal():
    for rule in ("FIELD_002", "TEXT_002", "DUP_001", "DUP_002", "DUP_003"):
        assert disposition([issue(rule)]) != QUARANTINE


def test_safe_cell_neutralises_formulas_only():
    assert safe_cell("=SUM(A1)") == "'=SUM(A1)"
    assert safe_cell("@cmd") == "'@cmd" and safe_cell("+1") == "'+1"
    assert safe_cell("Normal title") == "Normal title"
    assert safe_cell(42) == 42 and safe_cell(None) is None


def test_report_row_has_unique_codes_and_issue_count():
    row = {"id": 5, "url": "u", "title": "=x", "source": "a.com"}
    out = build_report_row(row, [issue("FIELD_001", "ERROR"), issue("FIELD_001", "ERROR"),
                                 issue("TEXT_002")], QUARANTINE)
    assert out[0] == 5 and out[3] == "'=x"
    assert out[-3:] == [QUARANTINE, 3, "FIELD_001;TEXT_002"]


def test_stats_distinguish_issue_counts_from_affected_articles():
    stats = Stats(["id", "source", "tags", "author"])
    two_fields = [issue("FIELD_001", "ERROR"), issue("FIELD_001", "ERROR")]
    stats.add_row({"id": 1, "source": "a.com", "tags": [], "author": "x"}, two_fields, QUARANTINE)
    stats.add_row({"id": 2, "source": "a.com", "tags": ["t"], "author": None}, [], ACCEPT)
    assert stats.issues_by_rule["FIELD_001"] == 2
    assert stats.articles_by_rule["FIELD_001"] == 1
    assert stats.with_any_issue == 1 and stats.rows == 2
    assert stats.missing["tags"] == 1 and stats.missing["author"] == 1   # [] and None
    done = {r[0]: r for r in stats.completeness_rows()}
    assert done["tags"][4] == 50.0 and done["id"][4] == 100.0


def test_domain_rows_rates():
    stats = Stats(["id", "source"])
    stats.add_row({"id": 1, "source": "a.com"}, [issue("URL_001", "ERROR")], QUARANTINE)
    stats.add_row({"id": 2, "source": "a.com"}, [], ACCEPT)
    (name, n, acc, warn, quar, _, qpct, invalid, ipct, *_), = stats.domain_rows()
    assert (name, n, acc, warn, quar, qpct, invalid, ipct) == ("a.com", 2, 1, 0, 1, 50.0, 1, 50.0)


def test_empty_dataset_statistics_do_not_divide_by_zero():
    stats = Stats(["id", "source"])
    assert stats.domain_rows() == []
    assert stats.completeness_rows()[0][4] == ""


def test_outputs_commit_and_rollback(tmp_path):
    out = Outputs(str(tmp_path / "ok"))
    out.write_row([1] + [""] * 11, ACCEPT)
    out.write_issue(issue("TEXT_002"))
    out.close(commit=True)
    names = sorted(p.name for p in (tmp_path / "ok").iterdir())
    assert "data_accepted.csv" in names and not any(n.endswith(".partial") for n in names)
    failed = Outputs(str(tmp_path / "bad"))
    failed.close(commit=False)
    assert list((tmp_path / "bad").iterdir()) == []          # nothing half-written is kept
    write_table(str(tmp_path), "t.csv", ["a"], [["=x"]])
    assert "'=x" in (tmp_path / "t.csv").read_text(encoding="utf-8-sig")


# ---- verification helpers (Step 8) -------------------------------------------------
def test_partition_check_passes_when_exact():
    result = check_partitions([1, 2, 3, 4], [1, 2], [3], [4])
    assert result["partitions_disjoint"] and result["partitions_cover_all_exactly"]
    assert result["ids_unique_in_all"] and result["rows_in_all"] == 4


def test_partition_check_detects_overlap_gap_and_duplicates():
    assert not check_partitions([1, 2, 3], [1, 2], [2, 3], [])["partitions_disjoint"]
    assert not check_partitions([1, 2, 3], [1], [2], [])["partitions_cover_all_exactly"]
    assert not check_partitions([1, 1, 2], [1], [1], [2])["ids_unique_in_all"]
    assert check_partitions([], [], [], [])["partitions_cover_all_exactly"]       # empty dataset


def test_issue_breakdown_separates_new_findings_from_reused_ones(tmp_path):
    path = tmp_path / "issues.csv"
    path.write_text(
        "rule_id,severity,scope,article_id,field,group,message\n"
        "SCHEMA_003,INFO,dataset,,author,,Optional column not present\n"
        "FLAG_001,WARNING,row,1,is_short,,short\n"
        "URL_002,WARNING,row,2,source,,mismatch\n"
        "FLAG_002,WARNING,row,2,is_non_article,,non-article\n"
        "FIELD_002,INFO,row,3,author,,missing author\n"
        "DUP_003,WARNING,row,4,body,g1,same body\n", encoding="utf-8-sig")
    result = issue_breakdown(str(path))
    assert result["issue_rows"] == 6
    assert result["article_ids"] == {1, 2, 3, 4}               # dataset-level row has no id
    assert result["warning_articles"] == 3
    assert result["warning_articles_with_new_finding"] == 1    # only article 2 (URL_002)
    assert result["warning_articles_only_preprocessing_findings"] == 2


# ---- ML-ready selection (Step 9) ---------------------------------------------------
def _write_reports(tmp_path, all_rows, issue_rows):
    allp, issp = tmp_path / "all.csv", tmp_path / "issues.csv"
    allp.write_text("article_id,validation_disposition\n" +
                    "".join(f"{i},{d}\n" for i, d in all_rows), encoding="utf-8-sig")
    issp.write_text("rule_id,severity,scope,article_id,field,group,message\n" +
                    "".join(f"{r},{s},row,{i},x,{g},m\n" for r, s, i, g in issue_rows),
                    encoding="utf-8-sig")
    return str(issp), str(allp)


def test_ml_selection_excludes_quarantine_and_bad_content_but_keeps_label_warnings(tmp_path):
    issues, alls = _write_reports(
        tmp_path,
        [(1, "ACCEPT"), (2, "QUARANTINE"), (3, "WARNING"), (4, "WARNING"), (5, "WARNING")],
        [("FLAG_003", "WARNING", 3, ""), ("TEXT_005", "WARNING", 3, ""),
         ("URL_002", "WARNING", 4, ""), ("TEXT_001", "WARNING", 4, ""),
         ("FIELD_002", "INFO", 5, "")])
    reasons = ml_exclusion_reasons(issues, alls)
    assert reasons[2] == ["QUARANTINE"]
    assert reasons[3] == ["FLAG_003", "TEXT_005"]
    assert 1 not in reasons and 4 not in reasons and 5 not in reasons   # kept


def test_ml_selection_keeps_first_url_copy_only(tmp_path):
    issues, alls = _write_reports(
        tmp_path, [(7, "WARNING"), (8, "WARNING"), (9, "WARNING")],
        [("DUP_001", "WARNING", 9, "g1"), ("DUP_001", "WARNING", 7, "g1"),
         ("DUP_001", "WARNING", 8, "g1")])
    reasons = ml_exclusion_reasons(issues, alls)
    assert 7 not in reasons
    assert reasons[8] == ["DUP_001_LATER_COPY"] and reasons[9] == ["DUP_001_LATER_COPY"]


def test_ml_selection_ignores_dataset_level_issues_and_can_require_author(tmp_path, monkeypatch):
    issues, alls = _write_reports(
        tmp_path, [(1, "ACCEPT"), (2, "ACCEPT")],
        [("SCHEMA_003", "INFO", "", ""), ("FIELD_002", "INFO", 2, "")])
    assert ml_exclusion_reasons(issues, alls) == {}
    import config
    monkeypatch.setattr(config, "ML_REQUIRE_AUTHOR", True)
    assert ml_exclusion_reasons(issues, alls) == {2: ["FIELD_002"]}
    