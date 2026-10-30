"""Step 1: inspect the existing project and identify the correct datasets.

Read-only. It opens files for reading, runs read-only SQL, and writes only one small report
(analysis_output/step1_inspection.json). No original dataset is changed.

Checks: which folders exist, the real columns and row counts of the validation files and
data_ml_ready.csv, how the ML-ready file was selected, and whether everything reconciles
with public.articles_cleaned.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import config

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))     # article bodies are long


def mb(path):
    return round(os.path.getsize(path) / 1024 / 1024, 1)


def is_blank(v):
    return v is None or str(v).strip() == ""


def scan_csv(path, id_col):
    """Stream a CSV once: columns, row count, ids, blank counts per column."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        blanks, ids, rows = Counter(), [], 0
        for row in reader:
            rows += 1
            for c in cols:
                if is_blank(row.get(c)):
                    blanks[c] += 1
            if id_col in row:
                ids.append(row[id_col])
    return {"columns": cols, "rows": rows, "blank_by_column": {c: blanks[c] for c in cols}, "ids": ids}


def folder_listing(path, depth=1):
    if not os.path.isdir(path):
        return None
    names = []
    for n in sorted(os.listdir(path)):
        if n.startswith(".") or n in {"__pycache__", "node_modules"}:
            continue
        names.append(n + ("/" if os.path.isdir(os.path.join(path, n)) else ""))
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 1: inspect datasets (read-only).")
    ap.add_argument("--validation-dir", default=config.VALIDATION_DIR)
    ap.add_argument("--no-db", action="store_true", help="skip the database checks")
    args = ap.parse_args(argv)
    vdir = args.validation_dir
    report, problems = {}, []

    print("== 1. Project folders ==")
    report["repo_root_listing"] = folder_listing(config.REPO_ROOT)
    for n in report["repo_root_listing"] or []:
        print("  ", n)

    print(f"\n== 2. Validation output folder ==\n   {vdir}")
    if not os.path.isdir(vdir):
        print("   NOT FOUND. Run the validation project first (python validate.py --run, then --export-ml-ready).")
        return 1
    expected = [config.VALIDATION_ALL_FILE, config.ML_READY_FILE, config.ML_EXCLUDED_FILE, config.ML_SUMMARY_FILE]
    for name in expected:
        p = os.path.join(vdir, name)
        print(f"   {'OK     ' if os.path.exists(p) else 'MISSING'} {name}" + (f"  ({mb(p)} MB)" if os.path.exists(p) else ""))
        if not os.path.exists(p):
            problems.append(f"missing file: {name}")
    if problems:
        return 1

    print("\n== 3. data_ml_ready.csv (primary candidate for classification) ==")
    ml = scan_csv(os.path.join(vdir, config.ML_READY_FILE), "id")
    print(f"   rows: {ml['rows']:,}    unique ids: {len(set(ml['ids'])):,}")
    print(f"   columns ({len(ml['columns'])}): {', '.join(ml['columns'])}")
    print("   blank values per column:")
    for c, n in ml["blank_by_column"].items():
        print(f"      {c:<18}{n:>8,}  ({100 * n / max(ml['rows'], 1):.2f}%)")
    absent = [c for c in ("description", "tags", "language", "published_time") if c not in ml["columns"]]
    print(f"   NOT in this file (only in the database): {', '.join(absent) or 'none'}")
    if len(set(ml["ids"])) != ml["rows"]:
        problems.append("duplicate ids inside data_ml_ready.csv")

    print("\n== 4. data_validation_all.csv (full validated dataset) ==")
    va = scan_csv(os.path.join(vdir, config.VALIDATION_ALL_FILE), "article_id")
    with open(os.path.join(vdir, config.VALIDATION_ALL_FILE), encoding="utf-8-sig", newline="") as fh:
        disp = Counter(r["validation_disposition"] for r in csv.DictReader(fh))
    print(f"   rows: {va['rows']:,}    unique ids: {len(set(va['ids'])):,}")
    print(f"   columns: {', '.join(va['columns'])}")
    print(f"   dispositions: {dict(disp)}")

    print("\n== 5. data_ml_excluded.csv (articles left out, with reasons) ==")
    ex = scan_csv(os.path.join(vdir, config.ML_EXCLUDED_FILE), "article_id")
    reasons = Counter()
    with open(os.path.join(vdir, config.ML_EXCLUDED_FILE), encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            reasons.update(x for x in r["reasons"].split(";") if x)
    print(f"   rows: {ex['rows']:,}")
    for k, v in reasons.most_common():
        print(f"      {k:<22}{v:>8,}")

    print("\n== 6. Reconciliation ==")
    ml_ids, ex_ids, va_ids = set(ml["ids"]), set(ex["ids"]), set(va["ids"])
    checks = {
        "ml_ready + excluded = validated rows": ml["rows"] + ex["rows"] == va["rows"],
        "ml_ready and excluded share no id": not (ml_ids & ex_ids),
        "ml_ready ids + excluded ids = validated ids": (ml_ids | ex_ids) == va_ids,
    }
    with open(os.path.join(vdir, config.ML_SUMMARY_FILE), encoding="utf-8") as fh:
        summ = json.load(fh)
    checks["summary json counts match the CSV files"] = (summ.get("ml_ready") == ml["rows"]
                                                         and summ.get("excluded") == ex["rows"])
    for k, ok in checks.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {k}")
        if not ok:
            problems.append(k)

    print("\n== 7. How data_ml_ready.csv was selected (from its summary file) ==")
    pol = summ.get("policy", {})
    print(f"   exclude rules: {', '.join(pol.get('exclude_rules', []))}")
    print(f"   quarantined excluded: {pol.get('quarantined_articles_excluded')}   "
          f"later duplicate copies dropped: {pol.get('drop_later_duplicates')}   "
          f"author required: {pol.get('require_author')}")
    print(f"   kept on purpose: {'; '.join(pol.get('kept_warnings', []))}")

    db = {}
    if not args.no_db:
        print("\n== 8. Database (read-only) ==")
        from db import get_readonly_connection
        conn = get_readonly_connection()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT count(*) FROM {config.INPUT_TABLE}")
            db["articles_cleaned_rows"] = cur.fetchone()[0]
            cur.execute(f"SELECT id FROM {config.INPUT_TABLE}")
            db_ids = {str(r[0]) for r in cur.fetchall()}
        finally:
            conn.close()
        print(f"   {config.INPUT_TABLE} rows: {db['articles_cleaned_rows']:,}")
        for k, ok in {"every validated id exists in the database": va_ids <= db_ids,
                      "every ML-ready id exists in the database": ml_ids <= db_ids,
                      "validated ids = database ids": va_ids == db_ids}.items():
            print(f"   {'PASS' if ok else 'FAIL'}  {k}")
            if not ok:
                problems.append(k)

    report.update({"validation_dir": vdir, "ml_ready": {k: v for k, v in ml.items() if k != "ids"},
                   "validated": {k: v for k, v in va.items() if k != "ids"},
                   "dispositions": dict(disp), "excluded_rows": ex["rows"],
                   "excluded_reasons": dict(reasons), "checks": checks, "database": db,
                   "problems": problems})
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "step1_inspection.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("\nRESULT:", "all checks passed" if not problems else f"PROBLEMS: {problems}")
    print(f"Report saved to {os.path.join(config.OUTPUT_DIR, 'step1_inspection.json')}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
