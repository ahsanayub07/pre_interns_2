"""Step 9: score the frozen classifier against YOUR hand labels (the blind test).

Reads   analysis_output/my_labels.csv      (exported from review_tool.html)
        analysis_output/review_sample.csv  (the same 200 articles)
Runs    the classifier (rules-1.2, unchanged) on those articles
Writes  analysis_output/step9_report.txt, score_details.csv, score_confusion.csv, charts/09_confusion.png
It never changes the rules and never touches the original datasets.
"""
import argparse
import math
import os
import sys

import pandas as pd

import classifier
import classifier_rules
import config

CATS = ["market_stocks", "technology", "health", "energy", "politics", "other", "uncertain"]
VALID = CATS + ["not_news"]


def safe_cell(v):
    v = "" if v is None else str(v)
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v


def wilson(k, n, z=1.96):
    """95% range for a share k/n (honest about small samples)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_inputs(labels_path, sample_path):
    lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    smp = pd.read_csv(sample_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    need = {"review_id", "article_id", "your_label"}
    if not need.issubset(lab.columns):
        raise SystemExit(f"my_labels.csv must have columns {sorted(need)}; found {list(lab.columns)}")
    lab["your_label"] = lab["your_label"].str.strip().str.lower()
    bad = lab[~lab["your_label"].isin(VALID)]
    if len(bad):
        raise SystemExit(f"Unknown labels in my_labels.csv: {sorted(set(bad['your_label']))}")
    if lab["review_id"].duplicated().any():
        raise SystemExit("my_labels.csv has the same review_id twice")
    df = lab.merge(smp, on="review_id", how="left", suffixes=("", "_s"))
    if df["id"].isna().any():
        raise SystemExit("Some review_ids in my_labels.csv are not in review_sample.csv "
                         "(did you run make_review_tool.py again after labelling?)")
    if (df["article_id"] != df["id"]).any():
        raise SystemExit("article_id in my_labels.csv does not match review_sample.csv. "
                         "The sample was rebuilt after you labelled; the labels belong to an older sample.")
    return df


def score(df):
    rows = []
    for _, r in df.iterrows():
        c = classifier.classify(r["title"], r["lead"])
        rows.append({"category": c["category"], "sub_label": c["sub_label"], "confidence": c["confidence"],
                     "needs_review": c["needs_review"], "reason": c["reason"], "page_type": c["page_type"]})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["correct"] = out["your_label"] == out["category"]
    return out


def per_category(ev):
    res = []
    for c in CATS:
        tp = int(((ev["your_label"] == c) & (ev["category"] == c)).sum())
        pred = int((ev["category"] == c).sum())
        true = int((ev["your_label"] == c).sum())
        p = tp / pred if pred else float("nan")
        r = tp / true if true else float("nan")
        f = 2 * p * r / (p + r) if pred and true and (p + r) else float("nan")
        res.append({"category": c, "you_labelled": true, "rules_predicted": pred, "correct": tp,
                    "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(res)


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def build_report(out):
    L = []
    n_all = len(out)
    nn = out[out["your_label"] == "not_news"]
    ev = out[out["your_label"] != "not_news"].copy()
    n = len(ev)
    L.append("STEP 9 - BLIND TEST OF THE RULE-BASED CLASSIFIER")
    L.append(f"Rules version: {classifier_rules.RULES_VERSION}")
    L.append(f"Articles you labelled: {n_all}   (not_news: {len(nn)}, scored: {n})")
    L.append("Your labels are the ground truth. The rules were frozen before this test.\n")

    k = int(ev["correct"].sum())
    lo, hi = wilson(k, n)
    L.append("1. OVERALL ACCURACY (strict: 'uncertain' counts as wrong unless you also said uncertain)")
    L.append(f"   {k} of {n} correct = {pct(k / n)}   (95% range: {pct(lo)} to {pct(hi)})")

    committed = ev[(ev["category"] != "uncertain") & (ev["your_label"] != "uncertain")]
    kc = int(committed["correct"].sum())
    if len(committed):
        lo, hi = wilson(kc, len(committed))
        L.append(f"   When the rules commit to a category: {kc} of {len(committed)} correct = "
                 f"{pct(kc / len(committed))}   (95% range: {pct(lo)} to {pct(hi)})")
    unc_pred = int((ev["category"] == "uncertain").sum())
    L.append(f"   The rules said 'uncertain' for {unc_pred} articles ({pct(unc_pred / n)}); "
             f"you said 'uncertain' for {int((ev['your_label'] == 'uncertain').sum())}.\n")

    L.append("2. PER CATEGORY (precision = when rules say X, how often you agree; recall = of your X, how many rules found)")
    pc = per_category(ev)
    L.append(f"   {'category':<14}{'you':>5}{'rules':>7}{'right':>7}{'precision':>11}{'recall':>9}{'f1':>8}")
    for _, r in pc.iterrows():
        L.append(f"   {r['category']:<14}{r['you_labelled']:>5}{r['rules_predicted']:>7}{r['correct']:>7}"
                 f"{pct(r['precision']):>11}{pct(r['recall']):>9}{pct(r['f1']):>8}")
    small = [r["category"] for _, r in pc.iterrows() if 0 < r["you_labelled"] < 10]
    if small:
        L.append(f"   Careful: fewer than 10 of your labels in {', '.join(small)}; those percentages are shaky.")
    L.append("")

    L.append("3. CONFUSION TABLE (rows = your label, columns = rules)")
    cm = pd.crosstab(ev["your_label"], ev["category"]).reindex(index=CATS, columns=CATS, fill_value=0)
    short = {c: c[:6] for c in CATS}
    L.append("   " + f"{'':<14}" + "".join(f"{short[c]:>8}" for c in CATS))
    for c in CATS:
        L.append("   " + f"{c:<14}" + "".join(f"{cm.loc[c, d]:>8}" for d in CATS))
    L.append("")

    L.append("4. MOST COMMON MISTAKES (your label -> rules label)")
    wrong = ev[~ev["correct"]]
    pairs = wrong.groupby(["your_label", "category"]).size().sort_values(ascending=False).head(8)
    for (a, b), cnt in pairs.items():
        L.append(f"   {cnt:>3}  {a} -> {b}")
    L.append("")

    L.append("5. IS THE CONFIDENCE / REVIEW FLAG USEFUL?")
    for flag, name in ((False, "needs_review = no "), (True, "needs_review = yes")):
        part = ev[ev["needs_review"] == flag]
        if len(part):
            L.append(f"   {name}: {len(part):>4} articles, {pct(part['correct'].mean())} correct")
    L.append(f"   Share of articles flagged for review: {pct(ev['needs_review'].mean())}\n")

    if len(nn):
        L.append("6. NOT_NEWS PAGES (shop, company profile...) - excluded from accuracy")
        L.append("   Rules put them in: " + ", ".join(f"{a} {b}" for a, b in nn["category"].value_counts().items()))
        L.append("")

    L.append("7. WRONG ANSWERS, ONE PER LINE (first 40; full list in score_details.csv)")
    for _, r in wrong.head(40).iterrows():
        L.append(f"   #{r['review_id']:>3} you={r['your_label']:<13} rules={r['category']:<13} {str(r['title'])[:80]}")
        L.append(f"        why: {str(r['reason'])[:110]}")
    return "\n".join(L), cm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(config.OUTPUT_DIR, "my_labels.csv"))
    ap.add_argument("--sample", default=os.path.join(config.OUTPUT_DIR, "review_sample.csv"))
    ap.add_argument("--out", default=config.OUTPUT_DIR)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for p in (a.labels, a.sample):
        if not os.path.exists(p):
            raise SystemExit(f"Not found: {p}")
    out = score(load_inputs(a.labels, a.sample))
    text, cm = build_report(out)
    os.makedirs(os.path.join(a.out, "charts"), exist_ok=True)
    with open(os.path.join(a.out, "step9_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    det = out[["review_id", "id", "domain", "title", "your_label", "category", "sub_label", "correct",
               "confidence", "needs_review", "reason", "page_type"]].copy()
    for col in ("title", "reason"):
        det[col] = det[col].map(safe_cell)
    det.to_csv(os.path.join(a.out, "score_details.csv"), index=False, encoding="utf-8-sig")
    cm.to_csv(os.path.join(a.out, "score_confusion.csv"), encoding="utf-8-sig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(CATS)), CATS, rotation=40, ha="right")
        ax.set_yticks(range(len(CATS)), CATS)
        for i in range(len(CATS)):
            for j in range(len(CATS)):
                v = int(cm.values[i, j])
                if v:
                    ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.values.max() / 2 else "black")
        ax.set_xlabel("Rules said")
        ax.set_ylabel("You said")
        ax.set_title("Blind test: your labels vs the rules")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "charts", "09_confusion.png"), dpi=130)
        plt.close(fig)
    except Exception as exc:                      # the chart is a bonus; the report is what matters
        print("Chart skipped:", exc)
    print(text)
    print(f"\nSaved: {os.path.join(a.out, 'step9_report.txt')}")


if __name__ == "__main__":
    sys.exit(main())
