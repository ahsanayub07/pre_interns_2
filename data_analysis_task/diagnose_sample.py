"""Step 6: diagnose the rule-based labels on the development sample (no rule is changed here).

Reads analysis_output/sample_classified.csv (made by classify_sample.py) and prints where the
rules probably go wrong: template pages, missing vocabulary, domain patterns, weak-evidence cases.
Domain is used ONLY to look for blind spots, never as classifier input.
"""
import os
import re
import sys
from collections import Counter

import pandas as pd

import classifier
import classifier_rules as R
import config

TEMPLATES = ["company overview", "stock price", "discount code", "coupon", "promo code", "review",
             "best ", "deals", "how to", " live ", "newsletter", "podcast", "video", "photos"]
SKIP = R.STOPWORDS | {"how", "what", "why", "who", "new", "after", "about", "more", "your", "you", "can",
                      "its", "his", "her", "their", "out", "up", "one", "two", "over", "into", "than",
                      "when", "now", "us", "we", "our", "all", "but", "or", "if", "no", "do", "who's",
                      "could", "would", "may", "says", "say", "first", "year", "years", "just", "get",
                      "off", "more", "most", "back", "new", "been", "were", "had", "about", "while"}


def title_words(titles, top=40):
    c = Counter()
    for t in titles:
        c.update(w for w in re.findall(r"[a-z][a-z'-]{2,}", classifier.clean_title(str(t)).lower()) if w not in SKIP)
    return c.most_common(top)


def template_table(df):
    rows = []
    for phrase in TEMPLATES:
        part = df[df["title"].str.lower().str.contains(re.escape(phrase), na=False)]
        if len(part):
            dist = part["category"].value_counts()
            rows.append((phrase.strip(), len(part), ", ".join(f"{k} {v}" for k, v in dist.head(3).items())))
    return rows


def domain_table(df, min_articles=25, top=15):
    big = df["domain"].value_counts()
    keep = big[big >= min_articles].head(top).index
    ct = pd.crosstab(df["domain"], df["category"], normalize="index").mul(100).round(0).astype(int)
    ct = ct.reindex(columns=classifier.CATEGORIES, fill_value=0)
    ct.insert(0, "n", big)
    return ct.loc[keep]


def weak_evidence_topics(df):
    part = df[df["reason"].str.startswith("weak evidence")]
    topics = part["reason"].str.extract(r"weak evidence: (\w+):")[0].value_counts()
    words = Counter()
    for r in part["reason"]:
        m = re.match(r"weak evidence: \w+: (.*)", r)
        if m:
            words.update(w.replace(" (title)", "").strip() for w in m.group(1).split(","))
    return topics, words.most_common(15)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines = []

    def say(*parts):
        text = " ".join(str(x) for x in parts)
        print(text)
        lines.append(text)

    path = os.path.join(config.OUTPUT_DIR, "sample_classified.csv")
    if not os.path.exists(path):
        raise SystemExit("Run python classify_sample.py first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["confidence"] = df["confidence"].astype(float)
    n = len(df)
    say(f"Diagnosing {n:,} rule-labelled articles (development sample)")

    say("\n1. Category shares for the biggest domains in the sample (blind-spot check, % of that domain):")
    say(domain_table(df).to_string())

    say("\n2. Title phrases typical of template or non-news pages (count, top labels):")
    for phrase, cnt, dist in template_table(df):
        say(f"  {phrase:<18}{cnt:>5}   {dist}")

    say("\n3. Most common title words in articles labelled 'other' (missing vocabulary shows up here):")
    say("  " + ", ".join(f"{w} {c}" for w, c in title_words(df[df['category'] == 'other']['title'])))

    say("\n4. Most common title words in articles labelled 'uncertain':")
    say("  " + ", ".join(f"{w} {c}" for w, c in title_words(df[df['category'] == 'uncertain']['title'], 30)))

    topics, words = weak_evidence_topics(df)
    say("\n5. 'Weak evidence' uncertain cases: which topic was closest, and which words triggered it:")
    say("  topics: " + ", ".join(f"{k} {v}" for k, v in topics.items()))
    say("  words:  " + ", ".join(f"{w} {c}" for w, c in words))

    say("\n6. Titles 'not analysed' as probably not English (12 random):")
    ne = df[df["reason"].str.contains("probably not english")]
    for t in ne["title"].sample(min(12, len(ne)), random_state=config.SAMPLE_SEED):
        say("  -", t[:100])

    say("\n7. 25 random 'other' titles (do you agree they are none of the five?):")
    oth = df[df["category"] == "other"]
    for _i, r in oth.sample(min(25, len(oth)), random_state=1).iterrows():
        say("  -", r["title"][:100])

    say("\n8. 20 random 'uncertain' titles with the reason:")
    unc = df[(df["category"] == "uncertain") & ~df["reason"].str.startswith("not analysed")]
    for _i, r in unc.sample(min(20, len(unc)), random_state=1).iterrows():
        say(f"  - {r['title'][:80]}  ::  {r['reason'][:60]}")

    with open(os.path.join(config.OUTPUT_DIR, "step6_diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved step6_diagnostics.txt to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
