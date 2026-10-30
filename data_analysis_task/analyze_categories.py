"""Step 11: what the labels tell us (domain, time, confidence) and how ready the data is for ML.

Reads   analysis_output/articles_classified.csv       (from classify_all.py; no article text)
        data_ml_ready.csv, only the id and title columns, to look for duplicate titles
Writes  analysis_output/category_by_domain.csv, category_by_year.csv, category_by_month_2024plus.csv,
        confidence_by_category.csv, uncertain_reasons.csv, ml_training_candidates.csv,
        step11_report.txt, charts/10_...14_*.png
Never changes the original datasets.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import config

CATS = ["market_stocks", "technology", "health", "energy", "politics", "other", "uncertain"]
TOPICS = CATS[:5]
# one fixed colour per category (validated categorical palette); other / uncertain are neutral greys
COLOR = {"market_stocks": "#2a78d6", "technology": "#eb6834", "health": "#1baf7a", "energy": "#eda100",
         "politics": "#e87ba4", "other": "#898781", "uncertain": "#c3c2b7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MIN_MONTH_N = 300          # months with fewer articles are not drawn (shares would be noise)
USABLE_CONF = 0.6


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["needs_review"] = df["needs_review"].eq("True")
    d = pd.to_datetime(df["published_date"], errors="coerce", utc=True, format="mixed")
    df["date"] = d.dt.tz_localize(None)
    return df


def by_domain(df, min_n=100):
    t = pd.crosstab(df["domain"], df["category"]).reindex(columns=CATS, fill_value=0)
    t["total"] = t.sum(axis=1)
    t = t[t["total"] >= min_n].sort_values("total", ascending=False)
    t["top_category"] = t[CATS].idxmax(axis=1)
    t["top_share_pct"] = (100 * t[CATS].max(axis=1) / t["total"]).round(1)
    return t


def by_period(df, freq, start):
    d = df[df["date"].notna() & (df["date"] >= start)].copy()
    d["period"] = d["date"].dt.to_period(freq).astype(str)
    t = pd.crosstab(d["period"], d["category"]).reindex(columns=CATS, fill_value=0)
    t["total"] = t.sum(axis=1)
    return t


def confidence_table(df):
    g = df.groupby("category")
    t = pd.DataFrame({"articles": g.size(),
                      "median_conf": g["confidence"].median().round(2),
                      "p25": g["confidence"].quantile(0.25).round(2),
                      "p75": g["confidence"].quantile(0.75).round(2),
                      "share_conf_ge_0.6_pct": (100 * g["confidence"].apply(lambda s: (s >= USABLE_CONF).mean())).round(1),
                      "needs_review_pct": (100 * g["needs_review"].mean()).round(1)})
    return t.reindex(CATS)


def uncertain_reasons(df):
    r = df[df["category"] == "uncertain"]["reason"]
    r = (r.str.replace(r"^weak evidence:.*", "weak evidence (one topic, too little)", regex=True)
          .str.replace(r"^ambiguous between .*", "ambiguous between two topics", regex=True)
          .str.replace(r"^not analysed:.*", "not analysed (not English)", regex=True))
    return r.value_counts().rename_axis("reason").reset_index(name="articles")


def usable_mask(df):
    """Safe-to-train rows: a real topic or `other`, news page, confident, not flagged."""
    return ((df["category"] != "uncertain") & (df["page_type"] == "news_article")
            & (df["confidence"] >= USABLE_CONF) & (~df["needs_review"]))


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def read_titles(path, ids):
    out = []
    for ch in pd.read_csv(path, usecols=["id", "title"], dtype=str, chunksize=20000, keep_default_na=False):
        out.append(ch[ch["id"].isin(ids)])
    t = pd.concat(out, ignore_index=True)
    t["tkey"] = t["title"].map(norm_title)
    return t[["id", "tkey"]]


def domain_baseline(df, seed=1):
    """How well could a model that only knows the domain do? (fit on one half, tested on the other)"""
    rng = np.random.default_rng(seed)
    m = rng.random(len(df)) < 0.5
    tr, te = df[m], df[~m]
    maj = tr.groupby("domain")["category"].agg(lambda s: s.value_counts().idxmax())
    seen = te["domain"].isin(maj.index)
    acc_seen = (te[seen]["domain"].map(maj) == te[seen]["category"]).mean() if seen.any() else float("nan")
    overall = tr["category"].value_counts().idxmax()
    pred = te["domain"].map(maj).fillna(overall)
    return {"domain_only_accuracy": float((pred == te["category"]).mean()),
            "accuracy_on_seen_domains": float(acc_seen),
            "always_largest_class_accuracy": float((te["category"] == overall).mean()),
            "largest_class": overall}


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def make_charts(df, dom, month, charts_dir, plt):
    os.makedirs(charts_dir, exist_ok=True)
    n = len(df)
    # 10: distribution
    vc = df["category"].value_counts().reindex(CATS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(CATS))[::-1]
    ax.barh(y, vc.values, color=[COLOR[c] for c in CATS], height=0.62)
    for yi, v in zip(y, vc.values):
        ax.text(v + vc.max() * 0.01, yi, f"{v:,}  ({100 * v / n:.1f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, CATS)
    ax.set_xlim(0, vc.max() * 1.25)
    ax.set_title(f"Articles per category ({n:,} ML-ready articles, rules-1.3)", loc="left", fontsize=11, color=INK)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "10_category_distribution.png"), dpi=130)
    plt.close(fig)
    # 11: category mix of the biggest domains (100% bars)
    top = dom.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    share = top[CATS].div(top["total"], axis=0) * 100
    for c in CATS:
        ax.barh(y, share[c].values, left=left, color=COLOR[c], height=0.7, label=c, edgecolor="#fcfcfb", linewidth=1.5)
        left += share[c].values
    ax.set_yticks(y, [f"{d}  ({t:,})" for d, t in zip(top.index, top["total"])], fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the domain's articles (%)", color=MUTED, fontsize=9)
    ax.set_title("What the 20 biggest domains publish (article count in brackets)", loc="left", fontsize=11, color=INK)
    ax.legend(ncol=7, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "11_category_by_domain.png"), dpi=130)
    plt.close(fig)
    # 12: monthly share from 2024
    m = month[month["total"] >= MIN_MONTH_N]
    if len(m):
        sh = m[CATS].div(m["total"], axis=0) * 100
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        x = np.arange(len(m))
        for c in TOPICS + ["other"]:
            ax.plot(x, sh[c].values, color=COLOR[c], linewidth=2, marker="o", markersize=3.5, label=c)
        step = max(1, len(m) // 12)
        ax.set_xticks(x[::step], list(m.index)[::step], rotation=40, ha="right")
        ax.set_xlim(-0.3, len(m) - 0.7)
        ax.set_ylabel("share of the month's articles (%)", color=MUTED, fontsize=9)
        ax.set_title(f"Category share by month since 2024 (months with at least {MIN_MONTH_N} articles)", loc="left", fontsize=11, color=INK)
        ax.legend(ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        style_ax(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        fig.tight_layout()
        fig.savefig(os.path.join(charts_dir, "12_category_trend_2024plus.png"), dpi=130)
        plt.close(fig)
    # 13: confidence
    ct = confidence_table(df)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(CATS))[::-1]
    ax.barh(y, ct["share_conf_ge_0.6_pct"].fillna(0).values, color=[COLOR[c] for c in CATS], height=0.62)
    for yi, v, med in zip(y, ct["share_conf_ge_0.6_pct"].fillna(0).values, ct["median_conf"].fillna(0).values):
        ax.text(v + 1, yi, f"{v:.0f}%   (median {med:.2f})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, CATS)
    ax.set_xlim(0, 125)
    ax.set_xlabel("share of the category with confidence >= 0.6 (%)", color=MUTED, fontsize=9)
    ax.set_title("How sure the rules are, per category (a heuristic, not a probability)", loc="left", fontsize=11, color=INK)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "13_confidence_by_category.png"), dpi=130)
    plt.close(fig)
    # 14: usable for ML
    um = usable_mask(df)
    tot = df["category"].value_counts().reindex(CATS, fill_value=0)
    use = df[um]["category"].value_counts().reindex(CATS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(CATS))[::-1]
    ax.barh(y, tot.values, color="#e1e0d9", height=0.62, label="all articles")
    ax.barh(y, use.values, color=[COLOR[c] for c in CATS], height=0.62, label="safe to train on")
    for yi, a, b in zip(y, tot.values, use.values):
        ax.text(a + tot.max() * 0.01, yi, f"{b:,} of {a:,}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, CATS)
    ax.set_xlim(0, tot.max() * 1.3)
    ax.set_title("Articles safe to train on: not uncertain, confidence >= 0.6, not flagged", loc="left", fontsize=11, color=INK)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "14_ml_readiness.png"), dpi=130)
    plt.close(fig)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def build_report(df, dom, year, month, ct, unc, ml):
    n = len(df)
    L = ["STEP 11 - WHAT THE LABELS SHOW, AND ML READINESS", f"Articles: {n:,} (rules-1.3)", ""]
    L.append("1. CATEGORY DISTRIBUTION")
    for c in CATS:
        k = int((df["category"] == c).sum())
        L.append(f"   {c:<14}{k:>8,}  {pct(k, n)}")
    L.append("")
    L.append("2. DOMAINS (domains with at least 100 articles)")
    L.append(f"   {len(dom)} domains cover {pct(int(dom['total'].sum()), n)} of all articles.")
    L.append("   Biggest 15: domain, articles, its most common category and that category's share")
    for d, r in dom.head(15).iterrows():
        L.append(f"   {d:<28}{int(r['total']):>7,}   {r['top_category']:<14}{r['top_share_pct']:>5.1f}%")
    focused = dom[(dom["top_share_pct"] >= 60) & (dom["top_category"].isin(TOPICS))]
    L.append(f"   {len(focused)} of {len(dom)} domains have 60%+ of their articles in ONE topic (not other/uncertain).")
    L.append("")
    L.append("3. TIME")
    ys = year[year.index >= "2020"]
    L.append("   Articles per year (2020 onward): " + ", ".join(f"{i} {int(r['total']):,}" for i, r in ys.iterrows()))
    big = month[month["total"] >= MIN_MONTH_N]
    L.append(f"   Months since 2024 with at least {MIN_MONTH_N} articles: {len(big)} of {len(month)}.")
    if len(big) >= 2:
        sh = big[CATS].div(big["total"], axis=0) * 100
        for c in TOPICS:
            L.append(f"   {c:<14} share first month {sh[c].iloc[0]:.1f}%  ->  last month {sh[c].iloc[-1]:.1f}%  "
                     f"(range {sh[c].min():.1f}% to {sh[c].max():.1f}%)")
    L.append("   Careful: the mix over time also reflects WHICH sites were scraped in which months, not only the news.")
    L.append("")
    L.append("4. CONFIDENCE")
    L.append("   category        articles  median  >=0.6   needs_review")
    for c, r in ct.iterrows():
        L.append(f"   {c:<14}{int(r['articles']):>9,}   {r['median_conf']:.2f}   {r['share_conf_ge_0.6_pct']:>5.1f}%   {r['needs_review_pct']:>5.1f}%")
    L.append("   Why 'uncertain':")
    for _, r in unc.iterrows():
        L.append(f"      {int(r['articles']):>7,}  {r['reason']}")
    L.append("")
    L.append("5. ML READINESS")
    L.append(f"   Safe to train on (real label, news page, confidence >= {USABLE_CONF}, not flagged): "
             f"{ml['usable']:,} of {n:,} ({pct(ml['usable'], n)})")
    L.append("   Per category: " + ", ".join(f"{c} {v:,}" for c, v in ml["usable_by_cat"].items()))
    big_c, small_c = max(ml["usable_by_cat"].values()), min(v for v in ml["usable_by_cat"].values() if v > 0)
    L.append(f"   Class balance: largest class is {big_c / small_c:.0f}x the smallest. Use class weights or sampling.")
    if "duplicate_titles" in ml:
        d = ml["duplicate_titles"]
        L.append(f"   Duplicate titles: {d['rows_in_dup_groups']:,} articles share a title with another article "
                 f"({d['groups']:,} groups); {d['groups_with_conflict']:,} groups got different labels.")
        L.append(f"   After keeping one article per title: {ml['candidates']:,} training candidates "
                 "(saved in ml_training_candidates.csv).")
    b = ml["baseline"]
    L.append(f"   Shortcut check: a model that only knows the DOMAIN gets {100 * b['domain_only_accuracy']:.1f}% right "
             f"(always guessing '{b['largest_class']}' gets {100 * b['always_largest_class_accuracy']:.1f}%).")
    L.append("   A text model is only impressive if it clearly beats the domain-only number.")
    L.append("   Split train/test by domain or by time, so the model cannot just recognise a site.")
    L.append("")
    L.append("6. HONEST WARNINGS FOR THE ML STAGE")
    L.append("   a) The labels come from keyword rules. A model trained on them mostly relearns those rules.")
    L.append("   b) Judge any model on the hand-labelled articles (300 in total), never on rule labels.")
    L.append("   c) Rule accuracy was 68.5% (v1.2, 200 articles) and 73.0% (v1.3, 100 articles); expect label noise.")
    L.append("   d) Use only the 'safe to train on' rows, or give the flagged ones lower weight.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default=os.path.join(config.OUTPUT_DIR, "articles_classified.csv"))
    ap.add_argument("--ml-ready", default=os.path.join(config.VALIDATION_DIR, config.ML_READY_FILE))
    ap.add_argument("--out", default=config.OUTPUT_DIR)
    ap.add_argument("--skip-titles", action="store_true", help="skip the duplicate-title check")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not os.path.exists(a.classified):
        raise SystemExit(f"Not found: {a.classified}  (run classify_all.py first)")
    os.makedirs(a.out, exist_ok=True)
    df = load(a.classified)
    dom = by_domain(df)
    year = by_period(df[df["date"] >= pd.Timestamp(config.TREND_MIN_DATE)], "Y", pd.Timestamp(config.TREND_MIN_DATE))
    month = by_period(df, "M", pd.Timestamp("2024-01-01"))
    ct, unc = confidence_table(df), uncertain_reasons(df)
    dom.to_csv(os.path.join(a.out, "category_by_domain.csv"), encoding="utf-8-sig")
    year.to_csv(os.path.join(a.out, "category_by_year.csv"), encoding="utf-8-sig")
    month.to_csv(os.path.join(a.out, "category_by_month_2024plus.csv"), encoding="utf-8-sig")
    ct.to_csv(os.path.join(a.out, "confidence_by_category.csv"), encoding="utf-8-sig")
    unc.to_csv(os.path.join(a.out, "uncertain_reasons.csv"), index=False, encoding="utf-8-sig")

    um = usable_mask(df)
    cand = df[um].copy()
    ml = {"usable": int(um.sum()), "usable_by_cat": cand["category"].value_counts().reindex(
        [c for c in CATS if c != "uncertain"], fill_value=0).to_dict()}
    if not a.skip_titles and os.path.exists(a.ml_ready):
        print("Reading titles to find duplicates (about 1 minute)...")
        t = read_titles(a.ml_ready, set(cand["id"]))
        cand = cand.merge(t, on="id", how="left")
        cand["tkey"] = cand["tkey"].fillna("")
        has = cand["tkey"].str.len() >= 8
        grp = cand[has].groupby("tkey")["category"].agg(["size", "nunique"])
        dup = grp[grp["size"] > 1]
        ml["duplicate_titles"] = {"rows_in_dup_groups": int(dup["size"].sum()), "groups": int(len(dup)),
                                  "groups_with_conflict": int((dup["nunique"] > 1).sum())}
        cand = cand[~(has & cand.duplicated("tkey", keep="first"))]
    ml["candidates"] = int(len(cand))
    ml["baseline"] = domain_baseline(df[um])
    cand[["id", "domain", "published_date", "category", "sub_label", "confidence"]].to_csv(
        os.path.join(a.out, "ml_training_candidates.csv"), index=False, encoding="utf-8-sig")
    text = build_report(df, dom, year, month, ct, unc, ml)
    with open(os.path.join(a.out, "step11_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        make_charts(df, dom, month, os.path.join(a.out, "charts"), plt)
    except Exception as exc:
        print("Charts skipped:", exc)
    print(text)
    print(f"\nSaved: {os.path.join(a.out, 'step11_report.txt')} and charts 10 to 14 in {os.path.join(a.out, 'charts')}")


if __name__ == "__main__":
    sys.exit(main())
    