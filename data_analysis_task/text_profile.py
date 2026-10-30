"""Step 4: sample the real ML-ready articles and measure how topic keywords appear.

Read-only. Reads data_ml_ready.csv in chunks (it is ~450 MB), keeps a random sample, and
reports keyword evidence to guide how the category rules are written. Writes only to
analysis_output/ (text_profile.json, text_sample.csv).
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config
import topic_keywords

DOMAIN_RE = r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]+)"


def compile_topic(spec):
    """Two regexes for one topic: case-insensitive words, case-sensitive acronyms."""
    words = re.compile(r"\b(" + "|".join(re.escape(w) for w in spec["words"]) + r")(?:s|es)?\b",
                       re.IGNORECASE) if spec["words"] else None
    acr = re.compile(r"\b(" + "|".join(re.escape(a) for a in spec["acronyms"]) + r")\b") \
        if spec["acronyms"] else None
    return words, acr


def find_hits(text, compiled):
    """Distinct keywords found in the text (words in lower case, acronyms as written)."""
    words, acr = compiled
    hits = set()
    if words:
        hits.update(m.group(1).lower() for m in words.finditer(text))
    if acr:
        hits.update(m.group(1) for m in acr.finditer(text))
    return hits


def domain_of(url):
    m = re.match(DOMAIN_RE, str(url))
    return re.sub(r"^www\.|:[0-9]+$", "", m.group(1).lower()) if m else "(unknown)"


def sample_articles(path, k, seed, chunk_rows, lead_chars, exclude_ids=None):
    """Exact random sample of k rows from a big CSV, without loading it all.

    Every row gets a random key; only the k smallest keys are kept across chunks.
    exclude_ids: ids (as text) that may not be drawn, e.g. the development sample.
    """
    rng = np.random.default_rng(seed)
    keep, total = None, 0
    cols = ["id", "url", "source", "title", "body", "published_date", "clean_word_count"]
    for chunk in pd.read_csv(path, usecols=cols, dtype=str, chunksize=chunk_rows, keep_default_na=False):
        total += len(chunk)
        if exclude_ids:
            chunk = chunk[~chunk["id"].isin(exclude_ids)]
        chunk = chunk.assign(_key=rng.random(len(chunk))).nsmallest(k, "_key")
        keep = chunk if keep is None else pd.concat([keep, chunk]).nsmallest(k, "_key")
        keep["body"] = keep["body"].str.slice(0, lead_chars * 2)       # keep memory small
    keep = keep.sort_values("id", key=lambda s: s.astype("int64")).reset_index(drop=True)
    return keep.drop(columns="_key"), total


def profile(sample, lead_chars):
    compiled = {n: compile_topic(s) for n, s in {**topic_keywords.TOPICS, **topic_keywords.WATCH}.items()}
    topics = list(topic_keywords.TOPICS)
    title_hits = {t: [] for t in compiled}
    lead_hits = {t: [] for t in compiled}
    for title, body in zip(sample["title"], sample["body"]):
        lead = body[:lead_chars]
        for t, c in compiled.items():
            title_hits[t].append(find_hits(title, c))
            lead_hits[t].append(find_hits(lead, c))
    n = len(sample)
    rows, counters = {}, {}
    for t in compiled:
        any_hit = [bool(a or b) for a, b in zip(title_hits[t], lead_hits[t])]
        rows[t] = {
            "any_pct": round(100 * sum(any_hit) / n, 1),
            "title_pct": round(100 * sum(bool(h) for h in title_hits[t]) / n, 1),
            "strong_pct": round(100 * sum(len(a | b) >= 3 for a, b in zip(title_hits[t], lead_hits[t])) / n, 1),
        }
        c = Counter()
        for a, b in zip(title_hits[t], lead_hits[t]):
            c.update(a | b)
        counters[t] = c.most_common(8)
    # crypto belongs to technology (sub-label blockchain_technology): report the combined share too
    union = [bool(title_hits["technology"][i] or lead_hits["technology"][i]
                  or title_hits["blockchain_technology"][i] or lead_hits["blockchain_technology"][i])
             for i in range(n)]
    rows["technology_incl_blockchain"] = {"any_pct": round(100 * sum(union) / n, 1),
                                          "title_pct": None, "strong_pct": None}
    n_topics = Counter(sum(bool(title_hits[t][i] or lead_hits[t][i]) for t in topics) for i in range(n))
    return rows, counters, {int(k): v for k, v in sorted(n_topics.items())}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # titles contain non-English text
    lines = []

    def say(*parts):
        text = " ".join(str(x) for x in parts)
        print(text)
        lines.append(text)

    path = os.path.join(config.VALIDATION_DIR, config.ML_READY_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"Not found: {path}")
    say("Reading the ML-ready file in chunks (about 30-60 seconds)...")
    sample, total = sample_articles(path, config.SAMPLE_SIZE, config.SAMPLE_SEED,
                                    config.CHUNK_ROWS, config.LEAD_CHARS)
    n = len(sample)
    say(f"File rows: {total:,}   sample: {n:,} (seed {config.SAMPLE_SEED})")
    sample["domain"] = sample["url"].map(domain_of)
    rows, counters, n_topics = profile(sample, config.LEAD_CHARS)

    say(f"\nShare of sampled articles with a keyword hit (title + first {config.LEAD_CHARS} body characters):")
    say(f"  {'topic':<27}{'any hit':>7}{'in title':>10}{'3+ words':>10}")
    for t, r in rows.items():
        tt = "" if r["title_pct"] is None else f"{r['title_pct']}%"
        st = "" if r["strong_pct"] is None else f"{r['strong_pct']}%"
        say(f"  {t:<27}{r['any_pct']:>6}%{tt:>10}{st:>10}")
    say("\nHow many of the 5 topics each article touches (0 = no keyword at all):")
    for k, v in n_topics.items():
        say(f"  {k} topics: {v:>5,}  ({100 * v / n:.1f}%)")
    say("\nMost common matched words per topic (look for noisy ones):")
    for t, c in counters.items():
        say(f"  {t:<15}" + ", ".join(f"{w} {v}" for w, v in c))
    say("\nTop domains in the sample:", ", ".join(f"{d} {c}" for d, c in sample['domain'].value_counts().head(8).items()))
    say("\n20 random titles (to see the variety of real news):")
    for t in sample["title"].sample(20, random_state=config.SAMPLE_SEED):
        say("  -", t[:110])

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = sample[["id", "domain", "published_date", "title"]].assign(lead=sample["body"].str.slice(0, 500))
    out.to_csv(os.path.join(config.OUTPUT_DIR, "text_sample.csv"), index=False, encoding="utf-8")
    with open(os.path.join(config.OUTPUT_DIR, "text_profile.json"), "w", encoding="utf-8") as fh:
        json.dump({"file_rows": total, "sample_rows": n, "seed": config.SAMPLE_SEED,
                   "lead_chars": config.LEAD_CHARS, "topic_hits": rows,
                   "topics_per_article": n_topics, "top_words": counters}, fh, indent=2)
    with open(os.path.join(config.OUTPUT_DIR, "step4_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nSaved text_profile.json, text_sample.csv and step4_report.txt to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
