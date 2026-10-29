"""Settings and the data contract for the validation stage.

Everything that can be tuned lives here, so rules stay free of magic numbers.
"""
RULESET_VERSION = "1.0.0"

INPUT_TABLE = "public.articles_cleaned"   # read-only input
BATCH_SIZE = 2000                         # rows streamed per batch

# ---- Data contract: column name -> accepted PostgreSQL data types -------------------
TEXT = {"text", "character varying", "character"}

REQUIRED_COLUMNS = {                      # missing column = schema failure, run stops
    "id":    {"bigint", "integer"},
    "url":   TEXT,
    "title": TEXT,
    "body":  TEXT,
}

OPTIONAL_COLUMNS = {                      # missing column = INFO, dependent rules are skipped
    "source":                  TEXT,                       # compared with the URL hostname
    "author":                  TEXT,
    "published_date":          {"date"},
    "published_time":          {"time with time zone", "time without time zone"},
    "imported_at":             {"timestamp with time zone", "timestamp without time zone"},
    "language":                TEXT,
    "description":             TEXT,
    "method":                  TEXT,
    "tags":                    {"ARRAY"},
    # flags and fingerprints produced by preprocessing (reused, never recomputed)
    "body_hash":               TEXT,
    "duplicate_group_size":    {"integer", "bigint"},
    "clean_word_count":        {"integer", "bigint"},
    "is_duplicate":            {"boolean"},
    "encoding_repaired":       {"boolean"},
    "has_suspicious_encoding": {"boolean"},
    "is_non_article":          {"boolean"},
    "author_missing":          {"boolean"},
    "is_short":                {"boolean"},
    "has_html_tags":           {"boolean"},
    "has_boilerplate":         {"boolean"},
}

# ---- Thresholds (lecture defaults) ---------------------------------------------------
MIN_TITLE_CHARS_WARNING = 8
MIN_CONTENT_CHARS_WARNING = 200
EARLIEST_PLAUSIBLE_DATE = "1990-01-01"
# A DATE with no time/zone may be one day "ahead" of UTC, so allow this much slack
# before calling a publication date later than the import time.
TIME_FUTURE_TOLERANCE_DAYS = 1

# ---- Content-quality settings --------------------------------------------------------
# Extraction-failure text (block pages, CAPTCHA...). Only bodies up to ERROR_PATTERN_MAX_CHARS
# are checked: a real block page is short, while a long article may merely mention these words.
ERROR_PATTERNS = [
    r"access denied", r"enable javascript", r"captcha", r"robot check", r"403 forbidden",
    r"page not found", r"verify you are (a )?human", r"are you a robot",
    r"checking your browser", r"unusual traffic",
]
ERROR_PATTERN_MAX_CHARS = 1500
# Titles that are not real headlines (compared in lower case, without punctuation).
PLACEHOLDER_TITLES = {
    "home", "homepage", "untitled", "no title", "title", "null", "none", "na", "n a",
    "error", "404", "403", "not found", "page not found", "access denied", "just a moment",
    "loading", "index", "default", "undefined", "news", "article",
}
# A body where fewer than this share of non-space characters are letters/digits is
# "mostly punctuation".
MIN_ALNUM_RATIO = 0.5

# ---- Duplicate-detection settings ----------------------------------------------------
# Query parameters that never change which article a URL points to.
TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAMS = {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src"}

# ---- Disposition policy --------------------------------------------------------------
# Rules in this set send a row to QUARANTINE. Everything else with an issue is WARNING.
# Missing author, short content and duplicate candidates are deliberately NOT fatal.
FATAL_RULES = {"FIELD_001", "URL_001"}
# INFO findings are logged and counted, but they do not turn an ACCEPT row into WARNING.
INFO_AFFECTS_DISPOSITION = False

# ---- ML-ready export policy (applied to the validation reports, never to the database) --
# An article is EXCLUDED from data_ml_ready.csv if it was QUARANTINED, carries any of these
# rules, or is a later copy of a duplicate. Everything else is kept, including a source label
# that differs from the URL host (URL_002), a short title (TEXT_001) and a missing author.
ML_EXCLUDE_RULES = {
    "FLAG_001", "TEXT_002",             # body too short
    "FLAG_002",                         # not an article page
    "FLAG_003", "FLAG_004",             # leftover HTML tags / cookie or newsletter text
    "FLAG_005",                         # text still shows broken encoding
    "TEXT_003", "TEXT_004", "TEXT_005", # block page, placeholder title, mostly punctuation
    "TIME_003", "TIME_004", "TYPE_001", # impossible or implausible dates, non-text value
}
ML_DROP_LATER_DUPLICATES = True         # keep the first copy (lowest id) of each duplicate
ML_REQUIRE_AUTHOR = False               # True also drops articles without an author
