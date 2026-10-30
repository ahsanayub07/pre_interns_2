"""Step 5: run the local classifier (rules v1) on the same 3,000-article sample as Step 4.

Read-only on the ML-ready file. Writes analysis_output/sample_classified.csv and step5_report.txt.
The labels are RULE-GENERATED, not verified ground truth. Step 6 measures accuracy by hand.
"""
import os
import sys

import pandas as pd

import classifier
import config
import text_profile


def safe_cell(v):
    """Stop spreadsheet programs from running text that starts with = + - @ as a formula."""
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v


def classify_frame(sample):
    rows = []
    for _id, domain, date, title, body in zip(sample["id"], sample["domain"], sample["published_date"],
                                              sample["title"], sample["body"]):
        r = classifier.classify(title, body)
        rows.append({"id": _id, "domain": domain, "published_date": date, "title": title,
                     "category": r["category"], "sub_label": r["sub_label"], "page_type": r["page_type"],
                     "confidence": r["confidence"],
                     "needs_review": r["needs_review"], "reason": r["reason"],
                     "secondary_topics": r["secondary_topics"],
                     **{f"score_{k}": v for k, v in r["scores"].items()}})
    return pd.DataFrame(rows)


def summarise(df):
    n = len(df)
    dist = df["category"].value_counts().reindex(classifier.CATEGORIES, fill_value=0)
    why = df[df["category"] == "uncertain"]["reason"].str.replace(r"^weak evidence:.*", "weak evidence", regex=True) \
        .str.replace(r"^ambiguous between .*", "ambiguous between two topics", regex=True).value_counts()
    return {"n": n, "dist": dist, "uncertain_reasons": why,
            "review": int(df["needs_review"].sum()),
            "conf_by_cat": df.groupby("category")["confidence"].agg(["median", "min", "max"]).round(2),
            "sub_labels": df["sub_label"].replace("", pd.NA).dropna().value_counts(),
            "page_types": df["page_type"].value_counts()}


def compare_with_previous(df, previous_path):
    """Rows = label in the previous run, columns = label now. Shows what the rule changes moved."""
    old = pd.read_csv(previous_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")[["id", "category"]]
    merged = old.merge(df[["id", "category"]], on="id", suffixes=("_before", "_now"))
    table = pd.crosstab(merged["category_before"], merged["category_now"]).reindex(
        index=classifier.CATEGORIES, columns=classifier.CATEGORIES, fill_value=0)
    return table, int((merged["category_before"] != merged["category_now"]).sum()), len(merged)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines = []

    def say(*parts):
        text = " ".join(str(x) for x in parts)
        print(text)
        lines.append(text)

    path = os.path.join(config.VALIDATION_DIR, config.ML_READY_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
    say("Reading the sample (about 30-60 seconds)...")
    sample, total = text_profile.sample_articles(path, config.SAMPLE_SIZE, config.SAMPLE_SEED,
                                                 config.CHUNK_ROWS, config.LEAD_CHARS)
    sample["domain"] = sample["url"].map(text_profile.domain_of)
    df = classify_frame(sample)
    s = summarise(df)
    n = s["n"]
    say(f"Rules version: {classifier.R.RULES_VERSION}   file rows: {total:,}   classified: {n:,}")
    say("\nCategory distribution (rule-generated, NOT verified):")
    for cat, v in s["dist"].items():
        say(f"  {cat:<15}{v:>6,}  {100 * v / n:5.1f}%")
    say(f"\nBlockchain sub-label (inside technology): {int(s['sub_labels'].sum()):,}")
    say("Page types: " + ", ".join(f"{k} {v:,}" for k, v in s["page_types"].items()))
    say(f"Needs manual review: {s['review']:,} ({100 * s['review'] / n:.1f}%)")
    say("\nWhy articles are 'uncertain':")
    for r, v in s["uncertain_reasons"].items():
        say(f"  {v:>5,}  {r}")
    say("\nConfidence by category (a heuristic from keyword evidence, not a probability):")
    say(s["conf_by_cat"].to_string())
    say("\nExamples per category (5 random titles each; check them by eye):")
    for cat in classifier.CATEGORIES:
        part = df[df["category"] == cat]
        say(f"\n[{cat}]")
        for _i, r in part.sample(min(5, len(part)), random_state=config.SAMPLE_SEED).iterrows():
            say(f"  - {r['title'][:90]}  ::  {r['reason'][:70]}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    current = os.path.join(config.OUTPUT_DIR, "sample_classified.csv")
    if os.path.exists(current):
        table, changed, both = compare_with_previous(df, current)
        say(f"\nChange against the previous run: {changed:,} of {both:,} articles got a different label "
            f"({100 * changed / max(both, 1):.1f}%). Rows = before, columns = now:")
        say(table.to_string())
        os.replace(current, os.path.join(config.OUTPUT_DIR, "sample_classified_previous.csv"))
    out = df.copy()
    out["title"] = out["title"].map(safe_cell)
    out.to_csv(os.path.join(config.OUTPUT_DIR, "sample_classified.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(config.OUTPUT_DIR, "step5_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved sample_classified.csv and step5_report.txt to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
