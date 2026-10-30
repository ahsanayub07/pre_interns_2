"""Step 10: run the frozen classifier on ALL ML-ready articles and save the labels in a NEW file.

Reads   data_validation_quality_assessment/validation_output/data_ml_ready.csv  (never changed)
Writes  analysis_output/articles_classified.csv   (id, domain, published_date + the labels, no article text)
        analysis_output/step10_summary.txt
The file is written to a .partial name first and renamed only when everything is finished.
"""
import argparse
import csv
import os
import sys
import time

import pandas as pd

import classifier
import classifier_rules
import config
import text_profile

COLS = ["id", "domain", "published_date", "category", "sub_label", "page_type", "confidence",
        "needs_review", "secondary_topics", "reason", "rules_version"]


def safe_cell(v):
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v


def classify_chunk(chunk):
    rows = []
    for _id, url, date, title, body in zip(chunk["id"], chunk["url"], chunk["published_date"],
                                           chunk["title"], chunk["body"]):
        r = classifier.classify(title, body)
        rows.append([_id, text_profile.domain_of(url), date, r["category"], r["sub_label"], r["page_type"],
                     r["confidence"], r["needs_review"], r["secondary_topics"], safe_cell(r["reason"]),
                     r["rules_version"]])
    return rows


def run(src, out_dir, chunk_rows, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, "articles_classified.csv")
    part = final + ".partial"
    n, t0 = 0, time.time()
    with open(part, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for chunk in pd.read_csv(src, usecols=["id", "url", "title", "body", "published_date"], dtype=str,
                                 chunksize=chunk_rows, keep_default_na=False):
            if limit is not None:
                chunk = chunk.head(limit - n)
            w.writerows(classify_chunk(chunk))
            n += len(chunk)
            print(f"  {n:,} articles done ({time.time() - t0:.0f}s)", flush=True)
            if limit is not None and n >= limit:
                break
    os.replace(part, final)
    return final, n


def summarise(final):
    df = pd.read_csv(final, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    n = len(df)
    L = [f"STEP 10 - ALL ARTICLES CLASSIFIED ({classifier_rules.RULES_VERSION})", f"Articles: {n:,}", ""]
    L.append("Category distribution:")
    for c, k in df["category"].value_counts().items():
        L.append(f"  {c:<14}{k:>8,}  {100 * k / n:5.1f}%")
    L.append("")
    L.append("Page types: " + ", ".join(f"{a} {b:,}" for a, b in df["page_type"].value_counts().items()))
    L.append("Sub-labels: " + ", ".join(f"{a} {b:,}" for a, b in df[df["sub_label"] != ""]["sub_label"].value_counts().items()))
    flag = (df["needs_review"] == "True").sum()
    L.append(f"Flagged needs_review: {flag:,} ({100 * flag / n:.1f}%)")
    conf = pd.to_numeric(df["confidence"])
    L.append(f"Confidence: median {conf.median():.2f}, share >= 0.6: {100 * (conf >= 0.6).mean():.1f}%")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(config.VALIDATION_DIR, config.ML_READY_FILE))
    ap.add_argument("--out", default=config.OUTPUT_DIR)
    ap.add_argument("--chunk", type=int, default=config.CHUNK_ROWS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N articles (quick trial)")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.path.exists(a.src):
        raise SystemExit(f"Not found: {a.src}")
    final, n = run(a.src, a.out, a.chunk, a.limit)
    text = summarise(final)
    with open(os.path.join(a.out, "step10_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print("\n" + text + f"\n\nSaved: {final}")


if __name__ == "__main__":
    sys.exit(main())
    