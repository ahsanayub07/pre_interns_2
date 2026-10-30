"""Settings for the News Intelligence module (step 2: domain and pattern analysis)."""
import os

INPUT_TABLE = "public.articles_cleaned"      # read-only input, never modified
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_output")

# Publication dates outside this window are counted separately and left out of trend charts.
# (The data contains a few implausible dates such as 1965-01-01.)
TREND_MIN_DATE = "1990-01-01"
TOP_N = 25                                    # rows shown in the printed summary

# ---- Step 1: where the validated datasets live (relative to the repository root) ----------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(REPO_ROOT, "data_validation_quality_assessment", "validation_output")
ML_READY_FILE = "data_ml_ready.csv"
ML_EXCLUDED_FILE = "data_ml_excluded.csv"
ML_SUMMARY_FILE = "data_ml_ready_summary.json"
VALIDATION_ALL_FILE = "data_validation_all.csv"

# ---- Step 4: text profile ------------------------------------------------------------------
SAMPLE_SIZE = 3000          # random articles examined
SAMPLE_SEED = 42            # same seed = same sample every run
LEAD_CHARS = 1500           # how much of the body is searched (the "lead")
CHUNK_ROWS = 5000           # rows read at a time from the 450 MB file

# ---- Step 8: blind test sample ---------------------------------------------------------------
REVIEW_N = 200              # articles you label by hand
REVIEW_SEED = 2026          # different from SAMPLE_SEED, so the articles are new
