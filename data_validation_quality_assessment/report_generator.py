"""Report writers and quality statistics for the validation stage (no database code)."""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import config
from validation_rules import ACCEPT, QUARANTINE, WARNING, is_missing

ROW_COLUMNS = ["article_id", "url", "source", "title", "author", "published_date",
               "published_time", "imported_at", "clean_word_count",
               "validation_disposition", "validation_issue_count", "validation_issue_codes"]
ISSUE_COLUMNS = ["rule_id", "severity", "scope", "article_id", "field", "group", "message"]
ROW_FILES = {"all": "data_validation_all.csv", ACCEPT: "data_accepted.csv",
             WARNING: "data_warning_review.csv", QUARANTINE: "data_quarantine.csv"}
ISSUES_FILE = "data_validation_issues.csv"
COMPLETENESS_FILE = "data_field_completeness.csv"
DOMAIN_FILE = "domain_quality_summary.csv"
SUMMARY_FILE = "data_validation_summary.json"

# Which rules need which optional column (to report checks that could not run).
RULES_NEEDING = {
    "source": ["URL_002"], "author": ["FIELD_002"],
    "published_date": ["TIME_002", "TIME_003", "TIME_004", "TIME_005"],
    "imported_at": ["TIME_001", "TIME_002", "TIME_003"],
    "published_time": ["TIME_003 (exact-time comparison; date-only comparison still runs)"],
    "is_short": ["FLAG_001"], "is_non_article": ["FLAG_002"], "has_html_tags": ["FLAG_003"],
    "has_boilerplate": ["FLAG_004"], "has_suspicious_encoding": ["FLAG_005"],
    "duplicate_group_size": ["DUP_003"], "body_hash": ["DUP_003 (group id)"],
}


def safe_cell(value):
    """Stops spreadsheet programs from running text as a formula (CSV injection):
    text starting with = + - @ gets a leading apostrophe. Only the report is changed."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def build_report_row(row, issues, disposition):
    codes = list(dict.fromkeys(i["rule_id"] for i in issues))     # unique, in order
    out = {"article_id": row.get("id")}
    for col in ROW_COLUMNS[1:9]:
        out[col] = safe_cell(row.get(col))
    out.update(validation_disposition=disposition, validation_issue_count=len(issues),
               validation_issue_codes=";".join(codes))
    return [out[c] for c in ROW_COLUMNS]


def is_blank(value):
    return is_missing(value) or (isinstance(value, (list, tuple)) and not value)


class Outputs:
    """Writes the CSV reports as '<name>.partial' and renames them only when the whole
    run succeeded, so a failed run never leaves half-written reports behind."""

    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.directory, self.counts, self._files = directory, Counter(), {}
        for key, name in {**ROW_FILES, "issues": ISSUES_FILE}.items():
            handle = open(os.path.join(directory, name + ".partial"), "w",
                          encoding="utf-8-sig", newline="")
            writer = csv.writer(handle)
            writer.writerow(ISSUE_COLUMNS if key == "issues" else ROW_COLUMNS)
            self._files[key] = (name, handle, writer)

    def write_row(self, report_row, disposition):
        for key in ("all", disposition):
            self._files[key][2].writerow(report_row)
            self.counts[key] += 1

    def write_issue(self, issue):
        self._files["issues"][2].writerow([safe_cell(issue.get(c)) for c in ISSUE_COLUMNS])
        self.counts["issues"] += 1

    def close(self, commit):
        for name, handle, _ in self._files.values():
            handle.close()
            partial = os.path.join(self.directory, name + ".partial")
            if commit:
                os.replace(partial, os.path.join(self.directory, name))
            else:
                os.remove(partial)

    def sha256(self, key):
        digest = hashlib.sha256()
        with open(os.path.join(self.directory, self._files[key][0]), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()


class Stats:
    """Counters filled row by row, so memory use does not grow with the table size."""

    def __init__(self, columns):
        self.columns = list(columns)
        self.rows = 0
        self.dispositions = Counter()
        self.issues_by_rule, self.issues_by_severity = Counter(), Counter()
        self.articles_by_rule = Counter()          # distinct articles per rule
        self.with_any_issue = self.with_warning_or_error = 0
        self.missing = Counter()
        self.by_source = defaultdict(Counter)

    def add_dataset_issues(self, issues):
        for i in issues:
            self.issues_by_rule[i["rule_id"]] += 1
            self.issues_by_severity[i["severity"]] += 1

    def add_row(self, row, issues, disposition):
        self.rows += 1
        self.dispositions[disposition] += 1
        for col in self.columns:
            if is_blank(row.get(col)):
                self.missing[col] += 1
        rules = set()
        for i in issues:
            self.issues_by_rule[i["rule_id"]] += 1
            self.issues_by_severity[i["severity"]] += 1
            rules.add(i["rule_id"])
        for rule in rules:
            self.articles_by_rule[rule] += 1
        if issues:
            self.with_any_issue += 1
        if any(i["severity"] in ("ERROR", "WARNING") for i in issues):
            self.with_warning_or_error += 1
        if "source" in self.columns:
            src = self.by_source[row.get("source") or "(blank)"]
            src["articles"] += 1
            src[disposition] += 1
            for key, rule in (("invalid_url", "URL_001"), ("missing_published_date", "TIME_005"),
                              ("short_content", "TEXT_002"), ("extraction_failure", "TEXT_003"),
                              ("missing_author", "FIELD_002"), ("non_article_flag", "FLAG_002")):
                src[key] += rule in rules
            src["duplicate_candidate"] += any(r.startswith("DUP_") for r in rules)

    # ---- tables -----------------------------------------------------------------------
    def completeness_rows(self):
        total = self.rows
        return [[c, total, self.missing[c], total - self.missing[c],
                 round((total - self.missing[c]) * 100 / total, 2) if total else ""]
                for c in self.columns]

    def domain_rows(self):
        def pct(n, d):
            return round(n * 100 / d, 2) if d else 0.0
        out = []
        for source, c in sorted(self.by_source.items(), key=lambda kv: (-kv[1]["articles"], kv[0])):
            n = c["articles"]
            out.append([source, n, c[ACCEPT], c[WARNING], c[QUARANTINE],
                        pct(c[WARNING], n), pct(c[QUARANTINE], n),
                        c["invalid_url"], pct(c["invalid_url"], n),
                        c["missing_published_date"], pct(c["missing_published_date"], n),
                        c["short_content"], pct(c["short_content"], n),
                        c["extraction_failure"], c["duplicate_candidate"],
                        c["missing_author"], c["non_article_flag"]])
        return out


DOMAIN_COLUMNS = ["source", "articles", "accepted", "warning", "quarantine", "warning_pct",
                  "quarantine_pct", "invalid_url", "invalid_url_rate_pct",
                  "missing_published_date", "missing_published_date_rate_pct",
                  "short_content_text002", "short_content_rate_pct",
                  "extraction_failure_candidates", "duplicate_candidates", "missing_author",
                  "non_article_flag"]
COMPLETENESS_COLUMNS = ["column", "total_rows", "missing", "non_missing", "completeness_pct"]


# ---------------------------------------------------------------- verification ------
# Rules whose findings preprocessing had already produced (its flags and hash groups).
ALSO_FOUND_BY_PREPROCESSING = {"FLAG_001", "FLAG_002", "FLAG_003", "FLAG_004", "FLAG_005",
                               "DUP_002", "DUP_003"}


def read_ids(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [int(r["article_id"]) for r in csv.DictReader(fh)]


def check_partitions(all_ids, accepted, warning, quarantine):
    """The three partitions must not overlap and must cover every row exactly once."""
    sa, sw, sq, sall = set(accepted), set(warning), set(quarantine), set(all_ids)
    return {
        "rows_in_all": len(all_ids),
        "ids_unique_in_all": len(sall) == len(all_ids),
        "partitions_disjoint": not (sa & sw or sa & sq or sw & sq),
        "partitions_cover_all_exactly": (sa | sw | sq == sall
                                         and len(accepted) + len(warning) + len(quarantine)
                                         == len(all_ids)),
    }


def issue_breakdown(path):
    """Read the issue log once and split the WARNING articles into those that only carry
    findings preprocessing had already made and those with at least one new finding."""
    ids, warn = set(), {}                       # warn: article_id -> has a new finding
    rows = 0
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            rows += 1
            if not r["article_id"]:
                continue                        # dataset-level (schema) finding
            aid = int(r["article_id"])
            ids.add(aid)
            if r["severity"] == "WARNING":
                new = r["rule_id"] not in ALSO_FOUND_BY_PREPROCESSING
                warn[aid] = warn.get(aid, False) or new
    new_count = sum(warn.values())
    return {"issue_rows": rows, "article_ids": ids, "warning_articles": len(warn),
            "warning_articles_with_new_finding": new_count,
            "warning_articles_only_preprocessing_findings": len(warn) - new_count}


ML_READY_FILE = "data_ml_ready.csv"
ML_EXCLUDED_FILE = "data_ml_excluded.csv"
ML_SUMMARY_FILE = "data_ml_ready_summary.json"


def ml_exclusion_reasons(issues_path, all_path):
    """Why each article is NOT ml-ready, decided only from the validation reports.

    Returns {article_id: [reason, ...]}. Articles that are not listed are ml-ready, apart
    from later duplicate copies, which the export finds with the preprocessing flag.
    """
    reasons, url_groups = {}, {}
    with open(all_path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["validation_disposition"] == "QUARANTINE":
                reasons.setdefault(int(r["article_id"]), []).append("QUARANTINE")
    with open(issues_path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if not r["article_id"]:
                continue
            aid, rule = int(r["article_id"]), r["rule_id"]
            if rule in config.ML_EXCLUDE_RULES or (config.ML_REQUIRE_AUTHOR and rule == "FIELD_002"):
                reasons.setdefault(aid, []).append(rule)
            elif rule == "DUP_001" and config.ML_DROP_LATER_DUPLICATES:
                url_groups.setdefault(r["group"], []).append(aid)
    for members in url_groups.values():                      # repeated normalised URL
        for aid in sorted(members)[1:]:
            reasons.setdefault(aid, []).append("DUP_001_LATER_COPY")
    return reasons


def write_table(directory, name, header, rows):
    path = os.path.join(directory, name)
    with open(path + ".partial", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows([[safe_cell(v) for v in r] for r in rows])
    os.replace(path + ".partial", path)


def config_snapshot():
    return {
        "min_title_chars_warning": config.MIN_TITLE_CHARS_WARNING,
        "min_content_chars_warning": config.MIN_CONTENT_CHARS_WARNING,
        "earliest_plausible_date": config.EARLIEST_PLAUSIBLE_DATE,
        "time_future_tolerance_days": config.TIME_FUTURE_TOLERANCE_DAYS,
        "error_patterns": config.ERROR_PATTERNS,
        "error_pattern_max_chars": config.ERROR_PATTERN_MAX_CHARS,
        "placeholder_titles": sorted(config.PLACEHOLDER_TITLES),
        "min_alnum_ratio": config.MIN_ALNUM_RATIO,
        "fatal_rules": sorted(config.FATAL_RULES),
        "batch_size": config.BATCH_SIZE,
    }


def build_summary(stats, *, input_total, row_limit, dry_run, available, started, finished,
                  hashes=None, files=None):
    n = stats.rows
    pct = lambda x: round(x * 100 / n, 2) if n else 0.0                 # noqa: E731
    skipped = {col: rules for col, rules in RULES_NEEDING.items() if col not in available}
    partitions = sum(stats.dispositions[k] for k in (ACCEPT, WARNING, QUARANTINE))
    return {
        "run": {"started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
                "input_table": config.INPUT_TABLE, "ruleset_version": config.RULESET_VERSION,
                "row_limit": row_limit, "dry_run": dry_run},
        "config": config_snapshot(),
        "input_rows_in_table": input_total,
        "input_rows_validated": n,
        "dispositions": {"accept": stats.dispositions[ACCEPT],
                         "warning": stats.dispositions[WARNING],
                         "quarantine": stats.dispositions[QUARANTINE]},
        "disposition_pct": {"accept": pct(stats.dispositions[ACCEPT]),
                            "warning": pct(stats.dispositions[WARNING]),
                            "quarantine": pct(stats.dispositions[QUARANTINE])},
        "reconciliation": {"partitions_add_up_to_input": partitions == n,
                           "partition_total": partitions},
        "issues": {"total": sum(stats.issues_by_rule.values()),
                   "by_severity": dict(sorted(stats.issues_by_severity.items())),
                   "by_rule": dict(sorted(stats.issues_by_rule.items())),
                   "distinct_articles_by_rule": dict(sorted(stats.articles_by_rule.items())),
                   "distinct_articles_with_any_issue": stats.with_any_issue,
                   "distinct_articles_with_warning_or_error": stats.with_warning_or_error},
        "rates_pct_of_validated_rows": {
            "invalid_url_URL_001": pct(stats.articles_by_rule["URL_001"]),
            "missing_published_date_TIME_005": pct(stats.articles_by_rule["TIME_005"]),
            "short_content_TEXT_002": pct(stats.articles_by_rule["TEXT_002"]),
            "extraction_failure_candidates_TEXT_003": pct(stats.articles_by_rule["TEXT_003"])},
        "duplicate_candidates": {
            "DUP_001_normalised_url": stats.articles_by_rule["DUP_001"],
            "DUP_002_title_and_body": stats.articles_by_rule["DUP_002"],
            "DUP_003_body_only": stats.articles_by_rule["DUP_003"]},
        "checks_not_performed_missing_optional_columns": skipped,
        "files": files or {},
        "issues_csv_sha256": (hashes or {}).get("issues"),
    }


def write_summary(directory, summary):
    path = os.path.join(directory, SUMMARY_FILE)
    with open(path + ".partial", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    os.replace(path + ".partial", path)


def utc_now():
    return datetime.now(timezone.utc)
