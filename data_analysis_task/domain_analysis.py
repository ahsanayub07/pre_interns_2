"""Step 2: domain and pattern analysis of public.articles_cleaned.

Read-only. Every number comes from the real table, aggregated inside PostgreSQL, so no
article text is downloaded. Results are written as CSV files to analysis_output/ and a
short summary is printed.

Volume is not quality: a domain with many articles is not more trustworthy or important.
"""
import argparse
import csv
import json
import os
import sys

import config
from db import get_readonly_connection

T = config.INPUT_TABLE

# Website name taken from the stored URL: lower case, no "www.", no port.
DOMAIN_SQL = (
    "regexp_replace(lower(substring(url from '^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+)')),"
    " '^www\\.|:[0-9]+$', '', 'g')"
)


def pct(part, whole):
    """Percentage rounded to 2 places; 0 when the whole is 0."""
    return round(100.0 * part / whole, 2) if whole else 0.0


def write_csv(name, header, rows):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, name)
    with open(path + ".partial", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(path + ".partial", path)          # file appears only when complete
    return path


def fetch(cur, sql, params=None):
    cur.execute(sql, params)          # params=None: a literal % (LIKE) is then left alone
    return cur.fetchall()


def run(conn):
    cur = conn.cursor()
    out = {}

    total = fetch(cur, f"SELECT count(*) FROM {T}")[0][0]
    out["total_articles"] = total

    # 1. articles per domain (from URL) with quality columns -------------------------------
    rows = fetch(cur, f"""
        SELECT {DOMAIN_SQL} AS domain,
               count(*) AS articles,
               round(avg(clean_word_count)::numeric, 1) AS avg_words,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY clean_word_count) AS median_words,
               count(*) FILTER (WHERE is_duplicate) AS duplicate_rows,
               count(*) FILTER (WHERE is_non_article) AS non_article_rows,
               count(*) FILTER (WHERE author_missing) AS author_missing,
               count(*) FILTER (WHERE published_date IS NULL) AS date_missing
        FROM {T} GROUP BY 1 ORDER BY articles DESC, domain""")
    cum, table = 0, []
    for d, n, avgw, medw, dup, nonart, am, dm in rows:
        cum += n
        table.append([d, n, pct(n, total), pct(cum, total), avgw,
                      None if medw is None else round(float(medw), 1),
                      pct(dup, n), pct(nonart, n), pct(am, n), pct(dm, n)])
    write_csv("articles_by_domain.csv",
              ["domain", "articles", "share_pct", "cumulative_share_pct", "avg_words",
               "median_words", "duplicate_pct", "non_article_pct", "author_missing_pct",
               "date_missing_pct"], table)
    out["distinct_domains"] = len(table)
    out["top5_share_pct"] = table[4][3] if len(table) >= 5 else None
    out["top10_share_pct"] = table[9][3] if len(table) >= 10 else None
    top = table[: config.TOP_N]

    # 2. stored source label vs URL domain -------------------------------------------------
    rows = fetch(cur, f"""
        SELECT source, {DOMAIN_SQL} AS domain, count(*)
        FROM {T}
        WHERE NOT ({DOMAIN_SQL} = lower(source) OR {DOMAIN_SQL} LIKE '%.' || lower(source))
        GROUP BY 1, 2 ORDER BY 3 DESC""")
    write_csv("source_domain_mismatch.csv", ["source", "domain_in_url", "articles"], rows)
    out["source_label_mismatch_articles"] = sum(r[2] for r in rows)

    # 3. publication dates -----------------------------------------------------------------
    lo = config.TREND_MIN_DATE
    out["date_missing"] = fetch(cur, f"SELECT count(*) FROM {T} WHERE published_date IS NULL")[0][0]
    out["date_out_of_range"] = fetch(
        cur, f"SELECT count(*) FROM {T} WHERE published_date < %s OR published_date > CURRENT_DATE", (lo,))[0][0]
    ok = "published_date >= %s AND published_date <= CURRENT_DATE"
    mn, mx = fetch(cur, f"SELECT min(published_date), max(published_date) FROM {T} WHERE {ok}", (lo,))[0]
    out["trend_date_range"] = [str(mn), str(mx)]
    for name, unit in (("year", "year"), ("month", "month"), ("day", "day")):
        rows = fetch(cur, f"""SELECT date_trunc('{unit}', published_date)::date AS period, count(*)
                              FROM {T} WHERE {ok} GROUP BY 1 ORDER BY 1""", (lo,))
        write_csv(f"articles_by_{name}.csv", [name, "articles"], rows)
    out["days_with_articles"] = len(rows)

    # 4. missing data ----------------------------------------------------------------------
    checks = [
        ("title", "title IS NULL OR btrim(title) = ''"),
        ("body", "body IS NULL OR btrim(body) = ''"),
        ("author", "author IS NULL OR btrim(author) = ''"),
        ("published_date", "published_date IS NULL"),
        ("published_time", "published_time IS NULL"),
        ("description", "description IS NULL OR btrim(description) = ''"),
        ("language", "language IS NULL OR btrim(language) = ''"),
        ("tags", "tags IS NULL OR cardinality(tags) = 0"),
    ]
    rows = []
    for field, cond in checks:
        n = fetch(cur, f"SELECT count(*) FROM {T} WHERE {cond}")[0][0]
        rows.append([field, n, pct(n, total)])
    write_csv("missing_fields.csv", ["field", "missing", "missing_pct"], rows)
    out["missing_pct"] = {r[0]: r[2] for r in rows}

    # 5. language --------------------------------------------------------------------------
    rows = fetch(cur, f"SELECT coalesce(nullif(btrim(language), ''), '(missing)'), count(*) "
                      f"FROM {T} GROUP BY 1 ORDER BY 2 DESC")
    write_csv("language_distribution.csv", ["language", "articles"], rows)
    out["languages_recorded"] = [r[0] for r in rows if r[0] != "(missing)"]

    # 6. duplicates and quality flags (reused from preprocessing) --------------------------
    d = fetch(cur, f"""
        SELECT count(*) FILTER (WHERE is_duplicate),
               count(*) FILTER (WHERE duplicate_group_size > 1),
               count(DISTINCT body_hash) FILTER (WHERE duplicate_group_size > 1),
               count(*) FILTER (WHERE is_non_article),
               count(*) FILTER (WHERE is_short),
               count(*) FILTER (WHERE has_html_tags),
               count(*) FILTER (WHERE has_boilerplate),
               count(*) FILTER (WHERE has_suspicious_encoding),
               count(*) FILTER (WHERE encoding_repaired)
        FROM {T}""")[0]
    names = ["is_duplicate", "in_duplicate_group", "duplicate_groups", "is_non_article",
             "is_short", "has_html_tags", "has_boilerplate", "has_suspicious_encoding",
             "encoding_repaired"]
    write_csv("duplicates_and_flags.csv", ["measure", "articles", "pct_of_all"],
              [[n, v, pct(v, total)] for n, v in zip(names, d)])
    out["duplicate_rows"] = d[0]

    # 7. text length -----------------------------------------------------------------------
    q = fetch(cur, f"""SELECT min(clean_word_count),
                              percentile_cont(ARRAY[0.05,0.25,0.5,0.75,0.95]) WITHIN GROUP (ORDER BY clean_word_count),
                              max(clean_word_count), round(avg(clean_word_count)::numeric,1)
                       FROM {T}""")[0]
    out["words"] = {"min": q[0], "p05": q[1][0], "p25": q[1][1], "median": q[1][2],
                    "p75": q[1][3], "p95": q[1][4], "max": q[2], "mean": float(q[3])}

    with open(os.path.join(config.OUTPUT_DIR, "overview.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out, top


def main(argv=None):
    argparse.ArgumentParser(description="Domain and pattern analysis (read-only).").parse_args(argv)
    conn = get_readonly_connection()
    try:
        out, top = run(conn)
    finally:
        conn.close()
    print(f"Articles analysed: {out['total_articles']:,}")
    print(f"Distinct domains:  {out['distinct_domains']}  "
          f"(top 5 = {out['top5_share_pct']}%, top 10 = {out['top10_share_pct']}%)")
    print(f"Valid publication dates: {out['trend_date_range'][0]} to {out['trend_date_range'][1]}; "
          f"missing {out['date_missing']:,}, out of range {out['date_out_of_range']:,}")
    print(f"Source label differs from URL domain: {out['source_label_mismatch_articles']:,} articles")
    print(f"Duplicate rows: {out['duplicate_rows']:,}   Languages recorded: {out['languages_recorded'] or 'none'}")
    print(f"Missing % by field: {out['missing_pct']}")
    print(f"\nTop {len(top)} domains (volume only, not quality):")
    for r in top:
        print(f"  {r[0]:<28}{r[1]:>7,}  {r[2]:>5}%")
    print(f"\nCSV files and overview.json written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
