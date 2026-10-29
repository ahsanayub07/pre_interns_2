"""Validation stage entry point (read-only on the database).

    python validate.py --schema-check              inspect the table, change nothing
    python validate.py --dry-run [--limit N]       validate and print results, write no files
    python validate.py --run [--limit N]           validate and write the reports
    python validate.py --verify                    re-check the reports against the database
    python validate.py --export-ml-ready           write the curated ML-ready dataset (a file)
"""
import argparse
import csv
import json
import os
from collections import Counter
import sys
import time

from psycopg2.extras import RealDictCursor

import config
from db import get_readonly_connection
from report_generator import (ML_EXCLUDED_FILE, ML_READY_FILE, ML_SUMMARY_FILE,
                              COMPLETENESS_COLUMNS, COMPLETENESS_FILE, DOMAIN_COLUMNS,
                              DOMAIN_FILE, ISSUES_FILE, ROW_FILES, SUMMARY_FILE, Outputs, Stats,
                              build_report_row, build_summary, check_partitions,
                              issue_breakdown, ml_exclusion_reasons, read_ids,
                              utc_now, write_summary, write_table)
from validation_rules import (ACCEPT, QUARANTINE, WARNING, DuplicateIndex, disposition,
                              has_errors, validate_row, validate_schema)

COLUMNS_SQL = """
SELECT column_name, data_type FROM information_schema.columns
WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position"""
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_output")


def read_schema(conn):
    schema, table = config.INPUT_TABLE.split(".")
    with conn.cursor() as cur:
        cur.execute(COLUMNS_SQL, (schema, table))
        columns = cur.fetchall()
        cur.execute(f"SELECT COUNT(*) FROM {config.INPUT_TABLE}")
        rows = cur.fetchone()[0]
    return columns, rows


def stream_rows(conn, columns, limit, name):
    """Yield rows ordered by id, batch by batch, from a server-side cursor."""
    sql = f"SELECT {', '.join(columns)} FROM {config.INPUT_TABLE} ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with conn.cursor(name=name, cursor_factory=RealDictCursor) as cur:
        cur.itersize = config.BATCH_SIZE
        cur.execute(sql)
        while True:
            batch = cur.fetchmany(config.BATCH_SIZE)
            if not batch:
                return
            yield from batch


def print_schema(columns, rows, issues):
    print(f"Table {config.INPUT_TABLE}: {rows:,} rows, {len(columns)} columns\n")
    for name, dtype in columns:
        print(f"  {name:<26}{dtype}")
    print("\nSchema findings:")
    for i in issues:
        print(f"  [{i['severity']}] {i['rule_id']}  {i['message']}")
    if not issues:
        print("  none")
    errors = sum(i["severity"] == "ERROR" for i in issues)
    print(f"\nResult: {'FAILED' if has_errors(issues) else 'OK'} ({errors} schema errors)")


def schema_check():
    conn = get_readonly_connection()
    try:
        columns, rows = read_schema(conn)
    finally:
        conn.close()
    issues = validate_schema(columns)
    print_schema(columns, rows, issues)
    return 1 if has_errors(issues) else 0


def run(limit, dry_run, output_dir):
    started, t0 = utc_now(), time.time()
    conn = get_readonly_connection()
    try:
        columns, total = read_schema(conn)
        schema_issues = validate_schema(columns)
        if has_errors(schema_issues):
            print_schema(columns, total, schema_issues)
            print("\nValidation stopped: fix the schema errors first. No reports written.")
            return 1
        present = {c for c, _ in columns}
        select_cols = [c for c, _ in columns
                       if c in config.REQUIRED_COLUMNS or c in config.OPTIONAL_COLUMNS]
        available = set(select_cols)
        expected = total if limit is None else min(limit, total)
        mode = "DRY RUN (no files written)" if dry_run else "FULL RUN"
        print(f"{mode}: {config.INPUT_TABLE}, {expected:,} of {total:,} rows, "
              f"batch size {config.BATCH_SIZE:,}\n")

        # Pass 1: count normalised URLs and title/body signatures (small digests only).
        index, seen = DuplicateIndex(), 0
        for row in stream_rows(conn, ["id", "url", "title", "body"], limit, "dup_pass"):
            index.add(row)
            seen += 1
            if seen % 20000 == 0:
                print(f"  pass 1 (duplicate index): {seen:,} rows")
        print(f"  pass 1 done: {seen:,} rows")

        # Pass 2: validate each row, assign its disposition, write the reports.
        stats, last_id = Stats(select_cols), None
        stats.add_dataset_issues(schema_issues)
        outputs = None if dry_run else Outputs(output_dir)
        ok = False
        try:
            if outputs:
                for issue in schema_issues:
                    outputs.write_issue(issue)
            for row in stream_rows(conn, select_cols, limit, "validate_pass"):
                if last_id is not None and row["id"] <= last_id:
                    raise RuntimeError(f"ids not strictly increasing at {row['id']}")
                last_id = row["id"]
                issues = validate_row(row, available) + index.check(row)
                decision = disposition(issues)
                stats.add_row(row, issues, decision)
                if outputs:
                    outputs.write_row(build_report_row(row, issues, decision), decision)
                    for issue in issues:
                        outputs.write_issue(issue)
                if stats.rows % 20000 == 0:
                    print(f"  pass 2 (validation): {stats.rows:,} rows")
            partitions = sum(stats.dispositions[k] for k in (ACCEPT, WARNING, QUARANTINE))
            if not (stats.rows == seen == expected == partitions):
                raise RuntimeError(f"row counts do not reconcile: pass1={seen}, "
                                   f"pass2={stats.rows}, expected={expected}, "
                                   f"partitions={partitions}")
            ok = True
        finally:
            if outputs:
                outputs.close(commit=ok)
    finally:
        conn.close()

    finished = utc_now()
    hashes = files = None
    if outputs:
        write_table(output_dir, COMPLETENESS_FILE, COMPLETENESS_COLUMNS, stats.completeness_rows())
        if "source" in present:
            write_table(output_dir, DOMAIN_FILE, DOMAIN_COLUMNS, stats.domain_rows())
        hashes = {"issues": outputs.sha256("issues")}
        files = {name: outputs.counts[key] for key, name in {**ROW_FILES, "issues": ISSUES_FILE}.items()}
    summary = build_summary(stats, input_total=total, row_limit=limit, dry_run=dry_run,
                            available=available, started=started, finished=finished,
                            hashes=hashes, files=files)
    if outputs:
        write_summary(output_dir, summary)
    print_results(summary, time.time() - t0, None if dry_run else output_dir)
    return 0


def print_results(summary, seconds, output_dir):
    n, d, p = summary["input_rows_validated"], summary["dispositions"], summary["disposition_pct"]
    print(f"\nValidated {n:,} of {summary['input_rows_in_table']:,} rows in {seconds:.1f} s")
    print(f"  ACCEPT     {d['accept']:>9,}  ({p['accept']}%)")
    print(f"  WARNING    {d['warning']:>9,}  ({p['warning']}%)")
    print(f"  QUARANTINE {d['quarantine']:>9,}  ({p['quarantine']}%)")
    rec = summary["reconciliation"]
    print(f"  partitions add up to input: {rec['partitions_add_up_to_input']} "
          f"({rec['partition_total']:,} = {n:,})")
    iss = summary["issues"]
    print(f"\nIssues: {iss['total']:,} in total {iss['by_severity']}")
    print(f"Distinct articles with any issue: {iss['distinct_articles_with_any_issue']:,}; "
          f"with a warning or error: {iss['distinct_articles_with_warning_or_error']:,}")
    print("\nRule      issues  distinct articles")
    for rule, count in sorted(iss["by_rule"].items()):
        print(f"  {rule:<9}{count:>7,}  {iss['distinct_articles_by_rule'].get(rule, 0):>7,}")
    skipped = summary["checks_not_performed_missing_optional_columns"]
    if skipped:
        print(f"\nChecks not performed (optional column missing): {skipped}")
    if output_dir:
        print(f"\nReports written to {output_dir}")
        for name, count in summary["files"].items():
            print(f"  {name:<32}{count:>9,} rows")


RECONCILIATION_FILE = "data_validation_reconciliation.json"

RAW_VS_CLEANED_SQL = """
SELECT COUNT(*),
       COUNT(*) FILTER (WHERE a.title       IS DISTINCT FROM c.title),
       COUNT(*) FILTER (WHERE a.author      IS DISTINCT FROM c.author),
       COUNT(*) FILTER (WHERE a.description IS DISTINCT FROM c.description),
       COUNT(*) FILTER (WHERE a.body        IS DISTINCT FROM c.body),
       COUNT(*) FILTER (WHERE a.source      IS DISTINCT FROM c.source),
       COUNT(*) FILTER (WHERE a.tags        IS DISTINCT FROM c.tags)
FROM public.articles a JOIN public.articles_cleaned c ON c.id = a.id"""
ONLY_IN_SQL = ("SELECT COUNT(*) FROM public.{x} a WHERE NOT EXISTS "
               "(SELECT 1 FROM public.{y} c WHERE c.id = a.id)")


def verify(output_dir):
    """Re-check the written reports against the database (read-only) and print the
    raw -> cleaned -> validated comparison."""
    needed = [SUMMARY_FILE, ISSUES_FILE] + list(ROW_FILES.values())
    missing = [n for n in needed if not os.path.exists(os.path.join(output_dir, n))]
    if missing:
        print(f"Reports not found in {output_dir}: {missing}\nRun --run first.")
        return 1
    path = lambda name: os.path.join(output_dir, name)                    # noqa: E731
    with open(path(SUMMARY_FILE), encoding="utf-8") as fh:
        summary = json.load(fh)
    parts = check_partitions(*(read_ids(path(ROW_FILES[k]))
                               for k in ("all", ACCEPT, WARNING, QUARANTINE)))
    issues = issue_breakdown(path(ISSUES_FILE))
    all_ids = set(read_ids(path(ROW_FILES["all"])))

    conn = get_readonly_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM public.articles_cleaned")
            db_ids = {r[0] for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) FROM public.articles")
            raw_total = cur.fetchone()[0]
            cur.execute(RAW_VS_CLEANED_SQL)
            joined, d_title, d_author, d_desc, d_body, d_source, d_tags = cur.fetchone()
            cur.execute(ONLY_IN_SQL.format(x="articles", y="articles_cleaned"))
            only_raw = cur.fetchone()[0]
            cur.execute(ONLY_IN_SQL.format(x="articles_cleaned", y="articles"))
            only_cleaned = cur.fetchone()[0]
    finally:
        conn.close()

    full_run = summary["run"]["row_limit"] is None
    d = summary["dispositions"]
    checks = [
        ("every row is in exactly one partition (no overlap, no gaps)",
         parts["ids_unique_in_all"] and parts["partitions_disjoint"]
         and parts["partitions_cover_all_exactly"]),
        ("partition sizes match the summary file",
         parts["rows_in_all"] == summary["input_rows_validated"]
         and d["accept"] + d["warning"] + d["quarantine"] == parts["rows_in_all"]),
        ("reported ids are exactly the ids in articles_cleaned" if full_run
         else "reported ids all exist in articles_cleaned", all_ids == db_ids if full_run
         else all_ids <= db_ids),
        ("every article id in the issue log exists in the input", issues["article_ids"] <= db_ids),
        ("issue log row count matches the summary", issues["issue_rows"] == summary["issues"]["total"]),
    ]
    print("Verification (read-only)")
    for ok, text in ((c[1], c[0]) for c in checks):
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    print(f"""
RAW -> CLEANED -> VALIDATED
  raw articles (public.articles)            {raw_total:>9,}
  cleaned (public.articles_cleaned)         {len(db_ids):>9,}   only in raw: {only_raw:,} | only in cleaned: {only_cleaned:,}

  Changes made by the PREPROCESSING stage (raw vs cleaned, {joined:,} matched rows):
    title {d_title:,} | author {d_author:,} | description {d_desc:,} | body {d_body:,} | source {d_source:,} | tags {d_tags:,}

  Changes made by VALIDATION: none. It is read-only: no row was added, changed or deleted.
  It only labelled each cleaned article:
    ACCEPT     {d['accept']:>9,}
    WARNING    {d['warning']:>9,}   {issues['warning_articles_with_new_finding']:,} have at least one finding that is new in validation
                                    (URL_002, TEXT_001/003/005, TIME_004, ...); {issues['warning_articles_only_preprocessing_findings']:,} carry only
                                    findings preprocessing had already flagged (FLAG_*, duplicates)
    QUARANTINE {d['quarantine']:>9,}
    total      {d['accept'] + d['warning'] + d['quarantine']:>9,}""")
    result = {"checks": [{"check": c[0], "passed": bool(c[1])} for c in checks],
              "raw_rows": raw_total, "cleaned_rows": len(db_ids),
              "rows_only_in_raw": only_raw, "rows_only_in_cleaned": only_cleaned,
              "preprocessing_changed_rows": {"title": d_title, "author": d_author,
                                             "description": d_desc, "body": d_body,
                                             "source": d_source, "tags": d_tags},
              "validation_changed_rows_in_database": 0, "dispositions": d,
              "warning_articles_with_new_finding": issues["warning_articles_with_new_finding"],
              "warning_articles_only_preprocessing_findings":
                  issues["warning_articles_only_preprocessing_findings"],
              "verified_utc": utc_now().isoformat()}
    with open(path(RECONCILIATION_FILE), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    failed = [c[0] for c in checks if not c[1]]
    print(f"\nResult: {'FAILED: ' + '; '.join(failed) if failed else 'ALL CHECKS PASSED'}")
    return 1 if failed else 0


ML_COLUMNS = ["id", "url", "source", "title", "author", "published_date", "body",
              "clean_word_count"]


def export_ml_ready(output_dir):
    """Write the curated ML-ready dataset as a separate file. The database is only read."""
    path = lambda name: os.path.join(output_dir, name)                    # noqa: E731
    needed = [ISSUES_FILE, ROW_FILES["all"], SUMMARY_FILE]
    missing = [n for n in needed if not os.path.exists(path(n))]
    if missing:
        print(f"Reports not found in {output_dir}: {missing}\nRun --run first.")
        return 1
    with open(path(SUMMARY_FILE), encoding="utf-8") as fh:
        summary = json.load(fh)
    if summary["run"]["row_limit"] is not None:
        print("The reports come from a --limit run. Run the full validation first (--run).")
        return 1
    reasons = ml_exclusion_reasons(path(ISSUES_FILE), path(ROW_FILES["all"]))
    known_ids = set(read_ids(path(ROW_FILES["all"])))

    conn = get_readonly_connection()
    try:
        columns, total = read_schema(conn)
        present = {c for c, _ in columns}
        cols = [c for c in ML_COLUMNS if c in present]
        later_dup = "is_duplicate" in present and config.ML_DROP_LATER_DUPLICATES
        select_cols = cols + (["is_duplicate"] if later_dup else [])
        kept = excluded = 0
        by_reason, by_source = Counter(), {"kept": Counter(), "excluded": Counter()}
        ready_path, excl_path = path(ML_READY_FILE), path(ML_EXCLUDED_FILE)
        with open(ready_path + ".partial", "w", encoding="utf-8", newline="") as ready, \
                open(excl_path + ".partial", "w", encoding="utf-8-sig", newline="") as excl:
            ready_w, excl_w = csv.writer(ready), csv.writer(excl)
            ready_w.writerow(cols)
            excl_w.writerow(["article_id", "url", "source", "title", "reasons"])
            for row in stream_rows(conn, select_cols, None, "ml_export"):
                if row["id"] not in known_ids:
                    raise RuntimeError("The table changed since the validation run (new id "
                                       f"{row['id']}). Run --run again, then export.")
                why = list(reasons.get(row["id"], []))
                if later_dup and row.get("is_duplicate") is True:
                    why.append("DUPLICATE_LATER_COPY")
                src = row.get("source") or "(blank)"
                if why:
                    excluded += 1
                    excl_w.writerow([row["id"], row.get("url"), src, row.get("title"),
                                     ";".join(dict.fromkeys(why))])
                    by_reason.update(set(why))
                    by_source["excluded"][src] += 1
                else:
                    kept += 1
                    ready_w.writerow([row.get(c) for c in cols])
                    by_source["kept"][src] += 1
        if kept + excluded != len(known_ids):
            raise RuntimeError(f"kept {kept} + excluded {excluded} != validated {len(known_ids)}")
    except BaseException:
        for name in (ML_READY_FILE, ML_EXCLUDED_FILE):
            if os.path.exists(path(name) + ".partial"):
                os.remove(path(name) + ".partial")
        raise
    finally:
        conn.close()
    for name in (ML_READY_FILE, ML_EXCLUDED_FILE):
        os.replace(path(name) + ".partial", path(name))
    pct = lambda n: round(n * 100 / total, 2) if total else 0.0           # noqa: E731
    result = {
        "created_utc": utc_now().isoformat(),
        "policy": {"exclude_rules": sorted(config.ML_EXCLUDE_RULES),
                   "drop_later_duplicates": config.ML_DROP_LATER_DUPLICATES,
                   "require_author": config.ML_REQUIRE_AUTHOR,
                   "quarantined_articles_excluded": True,
                   "kept_warnings": ["URL_002 (source label differs from URL host)",
                                     "TEXT_001 (short title)", "FIELD_002 (author missing)",
                                     "first copy of every duplicate group"]},
        "validated_articles": total, "ml_ready": kept, "excluded": excluded,
        "ml_ready_pct": pct(kept), "excluded_pct": pct(excluded),
        "excluded_by_reason_articles": dict(sorted(by_reason.items())),
        "articles_by_source": {s: {"kept": by_source["kept"][s],
                                   "excluded": by_source["excluded"][s]}
                               for s in sorted(set(by_source["kept"]) | set(by_source["excluded"]))},
        "reconciliation": {"kept_plus_excluded_equals_validated": kept + excluded == total},
    }
    with open(path(ML_SUMMARY_FILE), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"ML-ready export (database not changed)\n"
          f"  validated articles   {total:>9,}\n  ML-ready (kept)      {kept:>9,}  ({pct(kept)}%)\n"
          f"  excluded             {excluded:>9,}  ({pct(excluded)}%)\n\n"
          f"Articles per exclusion reason (one article can have several):")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<24}{n:>8,}")
    print(f"\nFiles in {output_dir}:\n  {ML_READY_FILE}  ({kept:,} rows, with article text)\n"
          f"  {ML_EXCLUDED_FILE}  ({excluded:,} rows, with reasons)\n  {ML_SUMMARY_FILE}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate the cleaned articles (read-only).")
    parser.add_argument("--schema-check", action="store_true",
                        help="inspect table layout against the data contract")
    parser.add_argument("--dry-run", action="store_true", help="validate, write no files")
    parser.add_argument("--run", action="store_true", help="validate and write the reports")
    parser.add_argument("--export-ml-ready", action="store_true",
                        help="write the curated ML-ready dataset as a separate file")
    parser.add_argument("--verify", action="store_true",
                        help="re-check the written reports against the database")
    parser.add_argument("--limit", type=int, help="only the first N articles (by id)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.schema_check:
        sys.exit(schema_check())
    if args.export_ml_ready:
        sys.exit(export_ml_ready(args.output_dir))
    if args.verify:
        sys.exit(verify(args.output_dir))
    if args.dry_run or args.run:
        sys.exit(run(args.limit, args.dry_run, args.output_dir))
    parser.print_help()


if __name__ == "__main__":
    main()
    