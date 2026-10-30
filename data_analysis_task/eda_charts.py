"""Step 3: charts and findings from the Step 2 analysis files (no database needed).

Reads CSV files from analysis_output/ and writes PNG charts to analysis_output/charts/.
Every chart title is built from the real numbers. Domain charts show article VOLUME only:
more articles does not mean more trustworthy or more important.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

import config

OUT = config.OUTPUT_DIR
CHARTS = os.path.join(OUT, "charts")
ACCENT, GREY = "#2f6fb5", "#b9b9b4"


def need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        raise SystemExit(f"Missing {name}. Run python domain_analysis.py first.")
    return pd.read_csv(path)


def style(ax, title, subtitle=None):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    os.makedirs(CHARTS, exist_ok=True)
    path = os.path.join(CHARTS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def top_domains(dom):
    n = len(dom)
    top = dom.head(20).iloc[::-1]
    share = dom.head(20)["articles"].sum() / dom["articles"].sum() * 100
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["domain"], top["articles"], color=ACCENT)
    for y, v in enumerate(top["articles"]):
        ax.text(v + 15, y, f"{v:,}", va="center", fontsize=8)
    style(ax, f"The {min(n, 20)} biggest of {n} domains hold {share:.1f}% of articles",
          "Volume only. It reflects how much was scraped per site, not quality or importance.")
    ax.set_xlim(0, top["articles"].max() * 1.12)
    ax.set_xlabel("Articles")
    return save(fig, "01_top_domains.png")


def concentration(dom):
    n = len(dom)
    cum = dom["cumulative_share_pct"].to_numpy()
    half = int((cum >= 50).argmax()) + 1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1, n + 1), cum, color=ACCENT, lw=2)
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.axvline(half, color=GREY, ls="--", lw=1)
    ax.annotate(f"{half} domains give 50% of articles", (half, 50), xytext=(half + 8, 38), fontsize=9)
    style(ax, f"Articles are spread across many domains: {half} of {n} reach 50%",
          "A straight diagonal would mean perfectly equal domains.")
    ax.set_xlabel("Domains, largest first")
    ax.set_ylabel("Cumulative share of articles (%)")
    ax.set_ylim(0, 100)
    return save(fig, "02_domain_concentration.png"), half


def by_year(year):
    year = year.copy()
    year["y"] = pd.to_datetime(year["year"]).dt.year
    peak = year.loc[year["articles"].idxmax()]
    recent = year[year["y"] >= 2024]["articles"].sum() / year["articles"].sum() * 100
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(year["y"], year["articles"], color=[ACCENT if y == peak["y"] else GREY for y in year["y"]])
    style(ax, f"{recent:.1f}% of articles were published in 2024 or later; peak year {int(peak['y'])}",
          "Publication year of each article. Dates before 1990 or in the future are excluded.")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Articles")
    return save(fig, "03_articles_by_year.png"), recent, int(peak["y"])


def by_month(month):
    month = month.copy()
    month["m"] = pd.to_datetime(month["month"])
    last = month.tail(36)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(last["m"], last["articles"], color=ACCENT, lw=2, marker="o", ms=3)
    style(ax, "Articles per publication month (latest 36 months with data)",
          "The last month may be incomplete. Scraper schedule can create apparent trends.")
    ax.set_ylabel("Articles")
    return save(fig, "04_articles_by_month.png")


def missing(miss):
    miss = miss.sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(miss["field"], miss["missing_pct"], color=[ACCENT if v >= 5 else GREY for v in miss["missing_pct"]])
    for y, v in enumerate(miss["missing_pct"]):
        ax.text(v + 1, y, f"{v:.2f}%", va="center", fontsize=9)
    full = ", ".join(miss[miss["missing_pct"] >= 99.9]["field"])
    style(ax, "Missing data per field" + (f": {full} are completely empty" if full else ""),
          "Share of the 100,026 cleaned articles with no usable value.")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Missing (%)")
    return save(fig, "05_missing_fields.png")


def domain_quality(dom):
    big = dom[dom["articles"] >= 200]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, col, label in ((axes[0], "author_missing_pct", "Author missing (%)"),
                           (axes[1], "non_article_pct", "Flagged non-article (%)")):
        d = big.nlargest(12, col).iloc[::-1]
        ax.barh(d["domain"], d[col], color=ACCENT)
        for y, v in enumerate(d[col]):
            ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=8)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(d[col].max() * 1.15, 1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_title("Highest missing-author rates", loc="left", fontsize=11, fontweight="bold")
    axes[1].set_title("Highest non-article rates", loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Data quality differs a lot between domains (domains with 200+ articles)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    return save(fig, "06_domain_quality.png")


def flags(fl):
    fl = fl.sort_values("articles")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(fl["measure"], fl["articles"], color=ACCENT)
    for y, v in enumerate(fl["articles"]):
        ax.text(v + 20, y, f"{v:,}", va="center", fontsize=9)
    style(ax, "Duplicate and quality flags from preprocessing",
          "Articles carrying each flag. One article can carry several.")
    ax.set_xlim(0, fl["articles"].max() * 1.12)
    ax.set_xlabel("Articles")
    return save(fig, "07_duplicates_and_flags.png")


def main():
    dom, year = need("articles_by_domain.csv"), need("articles_by_year.csv")
    month, miss = need("articles_by_month.csv"), need("missing_fields.csv")
    fl, mism = need("duplicates_and_flags.csv"), need("source_domain_mismatch.csv")
    with open(os.path.join(OUT, "overview.json"), encoding="utf-8") as fh:
        ov = json.load(fh)
    paths = [top_domains(dom)]
    p, half = concentration(dom)
    paths.append(p)
    p, recent, peak = by_year(year)
    paths += [p, by_month(month), missing(miss), domain_quality(dom), flags(fl)]

    print("Charts saved to", CHARTS)
    for p in paths:
        print("  ", os.path.basename(p))
    top = dom.iloc[0]
    print("\nKey findings (evidence from the real data):")
    print(f"  1. {ov['distinct_domains']} domains; largest is {top['domain']} with {int(top['articles']):,} articles "
          f"({top['share_pct']}%). {half} domains give half of all articles.")
    print(f"  2. {recent:.1f}% of articles were published in 2024 or later; the busiest year is {peak}.")
    print(f"  3. Completely empty fields: {', '.join(miss[miss['missing_pct'] >= 99.9]['field']) or 'none'}.")
    print(f"  4. {ov['source_label_mismatch_articles']:,} articles have a source label that differs from the URL domain "
          f"({len(mism)} source/domain pairs, see source_domain_mismatch.csv).")
    print(f"  5. {ov['duplicate_rows']:,} duplicate rows (later copies of repeated articles).")
    print("  Caution: domain volume shows scraping coverage, not reliability or importance.")


if __name__ == "__main__":
    sys.exit(main())
