# News Data Analysis and Rule-Based Classification

Part of `pre_interns_2` (Limeox Sargodha AI course). This stage analyses the news articles that were scraped,
cleaned and validated in the earlier tasks, and labels every article with a topic **without any external API,
API key or paid service**. Everything is Python, pandas and matplotlib.

## What the data is
- `data_ml_ready.csv`: 95,094 validated articles (from `data_validation_quality_assessment`, which holds the full
  validated set of 100,026). The original files and the database are never modified. Every output here is a new file.
- The data is heavily recent: 65,322 of 95,094 articles are dated 2026 and about 80% are from 2024 or later.

## Categories
`market_stocks`, `technology`, `health`, `energy`, `politics`, `other`, `uncertain`.

Decisions taken for the edge cases:
- Cryptocurrency price, trading or investment stories go to `market_stocks` (sub-label `crypto_markets`).
  Blockchain technology developments go to `technology` (sub-label `blockchain_technology`).
- A tech-company story is `market_stocks` when it is about stock prices, deals, ETFs, revenue or earnings, and
  `technology` when it is about products, software or technical developments.
- Company profile pages with no real story are `uncertain` with page type `company_profile`.
- Consumer gadgets are `technology`. Video games, sports, entertainment, crime and science are `other`.
  War and government news is `politics`.
- `uncertain` means the rules could not decide: too little text, not English, one weak hint, or two topics tied.

## How the classifier works (`classifier.py`, `classifier_rules.py`, version rules-1.3)
- Input: the title plus the first 1,500 characters of the body. **The domain is never used**, so the labels do not
  just copy the site.
- Weighted keyword lists per topic (strong words 2.0, weak words 0.75, title matches count 2.5 times).
  Evidence for `other` (sports, entertainment, crime...) is scored too.
- A category wins only if it clearly beats the second one; otherwise the article is `uncertain` and flagged.
- Each article gets a confidence (a heuristic, not a probability) and a `needs_review` flag.

## How accurate is it?
Two blind tests on articles that were not used to write the rules. **The rules were frozen before each test.**

| Round | Rules | Articles | Labelled by | Accuracy | Accuracy when it commits |
|---|---|---|---|---|---|
| 1 | v1.2 | 200 (3 non-news removed) | the author, by hand | 68.5% (135 of 197) | 74.4% |
| 2 | v1.3 | 100 | Claude (an AI assistant), not by the author | 73.0% | 81.8% |

The round 2 labels have not yet been spot-checked by the author. Treat the 73.0% as an estimate, and the human-labelled round 1 (68.5%) as the firmer number.

Read these numbers with care:
- The two rounds use different articles and different labellers, and the 95% ranges overlap
  (round 1: 62% to 75%; round 2: 64% to 81%). v1.3 is not proven to be better overall.
  The one clear gain is politics (recall 52% to 95%).
- Round 1 showed that finance news was missed (market_stocks recall 18%). v1.3 added finance rules, but round 2
  contained only 5 market articles, so that fix is **not yet tested**. `market_stocks` is probably undercounted.
- Rules cannot recognise names or topics without keywords, and they do not analyse non-English text.
- The `needs_review` flag works: in round 2, flagged articles were 32% correct and unflagged ones 87%.
- Round 1's confusion chart and per-category tables are in `results/`.

## Results on all 95,094 articles
| Category | Articles | Share |
|---|---|---|
| other | 36,103 | 38.0% |
| technology | 19,142 | 20.1% |
| politics | 15,342 | 16.1% |
| uncertain | 10,716 | 11.3% |
| market_stocks | 7,072 | 7.4% |
| health | 5,495 | 5.8% |
| energy | 1,224 | 1.3% |

- 23.2% of articles are flagged `needs_review`. `uncertain` is made of 4,040 ambiguous, 3,696 weak-evidence,
  1,746 non-English and 1,234 company-profile pages.
- **Domains are very different.** Out of 102 domains with 100+ articles, 25 put 60% or more of their articles in
  one topic. Examples: theverge.com 63% technology, techradar.com 60% technology, metro.co.uk 74% `other`.
- **Time:** monthly shares move a lot because the scrape was not even over time (23 of 34 months since 2024 have
  300+ articles). Treat the trend chart as "what was scraped", not as a news trend.
- Charts: `results/charts/10_category_distribution.png`, `11_category_by_domain.png`,
  `12_category_trend_2024plus.png`, `13_confidence_by_category.png`, `14_ml_readiness.png`
  (earlier EDA charts 01 to 07 are in the same folder).

## Is the data ready for machine learning?
Yes, with conditions:
- 71,674 articles are safe to train on (real label, news page, confidence 0.6 or more, not flagged); after dropping
  duplicate titles, 71,319 remain.
- Classes are very unbalanced: `other` 30,696 against energy 724 (about 42 times). Use class weights or sampling.
- **Shortcut risk:** a model that only knows the domain already gets 67.6% right (always guessing `other` gets
  42.7%). A text model is only impressive if it clearly beats 67.6%. Split train and test by domain or by time.
- The labels come from keyword rules, so a model trained on them mostly relearns those rules. Judge any model on the
  hand-labelled articles, not on rule labels.

## Known data issues
- About 1,800 articles are dated 1996 to 1999, which may be wrong dates. They are left out of the trend charts.
- Some titles contain mojibake (broken characters). This has not been counted yet.

## Files
| File | What it does |
|---|---|
| `inspect_datasets.py` | checks the input files and database, read-only |
| `domain_analysis.py`, `eda_charts.py` | domain, date, language, missing-value and duplicate analysis plus charts |
| `text_profile.py`, `topic_keywords.py` | text sample used to write the first rules |
| `classifier_rules.py`, `classifier.py` | the rules and the classifier |
| `classify_sample.py`, `diagnose_sample.py` | tuning on the 3,000-article development sample |
| `make_review_tool.py` | builds the blind hand-labelling page (no predictions shown) |
| `score_review.py` | scores the classifier against hand labels |
| `classify_all.py` | labels all articles and writes a new file |
| `analyze_categories.py` | category by domain and time, confidence, ML readiness, charts |
| `tests/` | automated tests (made-up examples, not real articles) |
| `results/` | charts and small summary tables (no article text) |

## Reproduce
```powershell
cd data_analysis_task
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q tests
python classify_all.py             # about 10 minutes, writes analysis_output\articles_classified.csv
python analyze_categories.py       # report and charts
```
Needs the validated data from `data_validation_quality_assessment\validation_output`. `domain_analysis.py` also needs a
`.env` with read-only database settings (see `.env.example`). **Never commit `.env`.**
`analysis_output/` is not committed because it contains article text.
