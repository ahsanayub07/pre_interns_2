# data_validation_quality_assessment — Article Data Validation

## 1. Project Overview

This project validates the cleaned article dataset produced by the preprocessing stage. It reads `public.articles_cleaned` (100,026 articles) and labels every article as one of three outcomes:

- **ACCEPT**: no issue found.
- **WARNING**: at least one issue found; the article stays in the dataset but is listed for review.
- **QUARANTINE**: a fatal issue (missing or blank `url`, `title` or `body`, or an invalid URL).

Validation is **read-only**. It never adds, changes or deletes a row in PostgreSQL. It only writes report files to `validation_output/`.

It also produces a separate curated file, `data_ml_ready.csv`, which contains the articles that are suitable for machine learning. This file is created outside the database; the database tables are not touched.

## 2. Technologies Used

- Python
- PostgreSQL (read-only connection, `REPEATABLE READ` snapshot)
- psycopg2, python-dotenv
- pytest (59 tests)
- Plain `csv` and `json` output, streamed in batches of 2,000 rows (no pandas)

## 3. Project Structure

```text
data_validation_quality_assessment/
├── .env.example           placeholder settings (copy to .env)
├── .gitignore             keeps .env and generated reports out of Git
├── config.py              data contract, thresholds, severity and ML-export policy
├── db.py                  read-only database connection
├── validation_rules.py    the rules (pure functions, no database)
├── report_generator.py    CSV and JSON report writers
├── validate.py            command-line program
├── Validation_Quality_Assessment_Results.ipynb   results notebook (rules, exercises A-E, reports)
├── requirements.txt
├── tests/
│   └── test_validation_rules.py
└── validation_output/     generated reports (not committed)
```

## 4. Setup

From the project folder, with the virtual environment active:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env        # put your local PostgreSQL password in DB_PASSWORD
```

`.env` must never be committed or shared.

## 5. How to Run

```powershell
python -m pytest -q tests          # 59 tests, no database needed
python validate.py --schema-check  # checks the table layout against the data contract
python validate.py --dry-run       # validates everything, writes no files
python validate.py --run           # validates and writes the reports
python validate.py --verify        # checks the reports and prints raw -> cleaned -> validated
python validate.py --export-ml-ready   # writes the ML-ready dataset
```

`--limit N` validates only the first N articles (by id), for quick tests. `--output-dir` changes where reports are written.

### Results notebook

`Validation_Quality_Assessment_Results.ipynb` explains every rule, demonstrates the rules and edge cases on sample records, reads the reports in `validation_output/`, and answers Exercises A to E (placeholder titles, per-domain monitoring, duplicate policy with an audit log, edge-case tests, acceptance criteria for two tasks). It never connects to the database. Run `python validate.py --run` first, then start Jupyter from this folder (`python -m pip install pandas notebook`).

## 6. Validation Rules

| Rule | Severity | Meaning |
|---|---|---|
| SCHEMA_001–004 | ERROR | Table layout does not match the data contract (run stops) |
| FIELD_001 | ERROR (fatal) | `url`, `title` or `body` missing, blank or only invisible characters |
| FIELD_002 | INFO | Author missing (never guessed, never fatal) |
| TYPE_001 | WARNING | A text field holds a non-text value |
| URL_001 | ERROR (fatal) | Not a plausible http(s) URL |
| URL_002 | WARNING | `source` does not match the URL hostname |
| TIME_001 / TIME_005 | INFO | `imported_at` / `published_date` missing |
| TIME_002 | WARNING | Timestamp present but unusable |
| TIME_003 | WARNING | Published later than imported |
| TIME_004 | WARNING | Publication date before 1990 |
| TEXT_001 | WARNING | Title too short |
| TEXT_002 | WARNING | Body too short |
| TEXT_003 | WARNING | Block page, CAPTCHA or access-denied text |
| TEXT_004 | WARNING | Placeholder title (for example "Home") |
| TEXT_005 | WARNING | Body is mostly punctuation |
| FLAG_001–005 | WARNING | Flags reused from preprocessing: short, non-article, HTML tags, cookie or newsletter text, broken encoding |
| DUP_001–003 | WARNING | Duplicate candidates: repeated normalized URL, identical title and body, identical body with a different title |

Duplicates are only candidates for review. Nothing is merged or deleted. All thresholds and policies are in `config.py`.

## 7. Output Files

In `validation_output/`:

| File | Content |
|---|---|
| `data_validation_all.csv` | Every article with its disposition |
| `data_accepted.csv` | ACCEPT articles |
| `data_warning_review.csv` | WARNING articles |
| `data_quarantine.csv` | QUARANTINE articles |
| `data_validation_issues.csv` | One row per finding (rule, field, article id) |
| `data_field_completeness.csv` | Completeness per field |
| `domain_quality_summary.csv` | Quality rates per source |
| `data_validation_summary.json` | Counts and settings of the run |
| `data_validation_reconciliation.json` | Reconciliation of raw, cleaned and validated counts |
| `data_ml_ready.csv` | Articles kept for machine learning (full text) |
| `data_ml_excluded.csv` | Excluded articles with their reasons |
| `data_ml_ready_summary.json` | Counts per exclusion reason and per source |

Report files are large and are excluded from Git.

## 8. Results

### Raw, cleaned and validated

| Stage | Articles |
|---|---:|
| Raw (`public.articles`) | 100,026 |
| Cleaned (`public.articles_cleaned`) | 100,026 |
| Validated | 100,026 |

No article is missing between raw and cleaned (0 only in raw, 0 only in cleaned). The preprocessing stage changed the text of some articles (body 15,552, title 4, author 5). **The validation stage changed nothing.**

### Dispositions

| Disposition | Articles |
|---|---:|
| ACCEPT | 93,311 |
| WARNING | 6,715 |
| QUARANTINE | 0 |
| **Total** | **100,026** |

Of the 6,715 warnings, 1,751 have at least one finding that is new in validation (for example URL_002, TEXT_001, TEXT_003, TEXT_005, TIME_004). The other 4,964 carry only findings that preprocessing had already flagged (FLAG_* and duplicates).

### Verification

`python validate.py --verify` passed all five checks:

- every row is in exactly one partition (no overlap, no gaps),
- partition sizes match the summary file,
- reported ids are exactly the ids in `articles_cleaned`,
- every article id in the issue log exists in the input,
- the issue log row count matches the summary.

### ML-ready dataset

| | Articles | Share |
|---|---:|---:|
| Kept (`data_ml_ready.csv`) | 95,094 | 95.07% |
| Excluded (`data_ml_excluded.csv`) | 4,932 | 4.93% |
| **Total** | **100,026** | |

Exclusion reasons (one article can have several reasons):

| Reason | Articles |
|---|---:|
| FLAG_002 non-article page | 2,584 |
| FLAG_001 short body | 908 |
| DUPLICATE_LATER_COPY | 779 |
| FLAG_003 leftover HTML | 708 |
| FLAG_004 cookie or newsletter text | 39 |
| TEXT_003 block page text | 17 |
| FLAG_005 broken encoding | 9 |
| TIME_004 implausibly old date | 2 |
| TEXT_005 mostly punctuation | 1 |

The policy is defined in `config.py` (`ML_EXCLUDE_RULES`, `ML_DROP_LATER_DUPLICATES`, `ML_REQUIRE_AUTHOR`). Articles with a missing author, a short title, or a source label that differs from the URL host are kept.

## 9. Important Limitations

- URL checks are syntax checks only. No website is contacted, so a valid-looking URL may still be dead.
- `imported_at` is the time the article was loaded into the database, not the original scrape time.
- Duplicate findings are candidates for review. The ML export keeps the first copy (lowest id) of each duplicate group; if that first copy has another problem, the whole group is excluded.
- Rules are pattern and keyword checks. A warning is a hint for review, not proof that an article is wrong.
- `data_ml_ready.csv` contains unaltered article text. Open it with pandas or a similar tool, not Excel.
- Passing validation does not prove that the content is factually correct.

## 10. Data Safety

- The database connection is read-only: any INSERT, UPDATE, DELETE or DDL statement is refused.
- `.env` holds the database password and is excluded from Git.
- Generated reports and the ML-ready dataset are excluded from Git.
- No record is deleted or modified in `public.articles` or `public.articles_cleaned`.
