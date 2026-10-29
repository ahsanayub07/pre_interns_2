"""Pure validation rules (no database), so each one is easy to test.

An issue is a dict: rule_id, severity (ERROR/WARNING/INFO), scope (dataset/row),
article_id, field, message.

Rule IDs
  SCHEMA_001..004  table layout          FIELD_001  required field missing/blank (fatal)
  FIELD_002        author missing (INFO) TYPE_001   text field holds a non-text value
  URL_001          invalid URL (fatal)   URL_002    source does not match URL hostname
  TIME_001         imported_at missing   TIME_002   timestamp present but unusable
  TIME_003         published after import TIME_004  implausibly old publication date
  TIME_005         publication date missing
  TEXT_001         title too short       TEXT_002   body too short
  TEXT_003         block / CAPTCHA text  TEXT_004   placeholder title
  TEXT_005         body mostly punctuation
  FLAG_001..005    reused preprocessing flags (short, non-article, HTML, boilerplate, encoding)
  DUP_001          normalised URL repeats
  DUP_002          identical title + body (strong duplicate candidate)
  DUP_003          identical body, different title (duplicate candidate)
Duplicates are only ever candidates for review. Nothing is merged or deleted.
"""
import hashlib
import re
from collections import Counter
from urllib.parse import parse_qsl
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import urlparse

import config

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)
_HOST_RE = re.compile(r"^[^\s/\\?#@:.]+(\.[^\s/\\?#@:.]+)+$")   # labels never contain dots


def make_issue(rule_id, severity, scope, field, message, article_id=None, group=None):
    """`group` ties together the articles of one duplicate group (None otherwise)."""
    return {"rule_id": rule_id, "severity": severity, "scope": scope,
            "article_id": article_id, "field": field, "message": message, "group": group}


# ---------------------------------------------------------------- schema ------------
def validate_schema(columns, required=None, optional=None):
    """Check the table layout against the data contract.

    `columns` is a list of (column_name, data_type) pairs, as in information_schema.
    Missing columns are reported, never silently created.
    """
    required = config.REQUIRED_COLUMNS if required is None else required
    optional = config.OPTIONAL_COLUMNS if optional is None else optional
    issues = []
    names = [name for name, _ in columns]
    types = dict(columns)

    for name, count in Counter(names).items():
        if count > 1:
            issues.append(make_issue("SCHEMA_001", "ERROR", "dataset", name,
                                     f"Duplicate column name: {name}"))
    for name, accepted in required.items():
        if name not in types:
            issues.append(make_issue("SCHEMA_002", "ERROR", "dataset", name,
                                     f"Missing required column: {name}"))
        elif types[name] not in accepted:
            issues.append(make_issue("SCHEMA_004", "ERROR", "dataset", name,
                                     f"Column {name} has type '{types[name]}', "
                                     f"expected one of {sorted(accepted)}"))
    for name, accepted in optional.items():
        if name not in types:
            issues.append(make_issue("SCHEMA_003", "INFO", "dataset", name,
                                     f"Optional column not present: {name}; "
                                     f"rules that need it are skipped"))
        elif types[name] not in accepted:
            issues.append(make_issue("SCHEMA_004", "WARNING", "dataset", name,
                                     f"Column {name} has type '{types[name]}', "
                                     f"expected one of {sorted(accepted)}; "
                                     f"rules that need it are skipped"))
    return issues


def has_errors(issues):
    return any(i["severity"] == "ERROR" for i in issues)


# ---------------------------------------------------------------- fields ------------
def is_missing(value):
    """None, NaN, or text that is empty after removing spaces and zero-width characters."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:      # NaN
        return True
    if isinstance(value, str):
        return not value.translate(_ZERO_WIDTH).strip()
    return False


def check_fields(row, available, article_id):
    issues = []
    for field in ("url", "title", "body"):
        if field in available and is_missing(row.get(field)):
            issues.append(make_issue("FIELD_001", "ERROR", "row", field,
                                     f"Required field '{field}' is missing or blank.",
                                     article_id))
    for field in ("url", "title", "body", "author"):
        value = row.get(field)
        if field in available and value is not None and not isinstance(value, str) \
                and not is_missing(value):
            issues.append(make_issue("TYPE_001", "WARNING", "row", field,
                                     f"Expected text value, found {type(value).__name__}.",
                                     article_id))
    if "author" in available and is_missing(row.get("author")):
        issues.append(make_issue("FIELD_002", "INFO", "row", "author",
                                 "Author is missing in the source data; not guessed.",
                                 article_id))
    return issues


# ---------------------------------------------------------------- url ---------------
def parse_http_url(value):
    """Return the parsed URL if it is a plausible HTTP(S) URL, else None.

    Syntax only: this does not prove the page exists or is reachable.
    """
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return None
    if re.search(r"\s", value):
        return None
    try:
        parsed = urlparse(value)
        host = parsed.hostname
        parsed.port                                      # raises ValueError if out of range
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https") or not host:
        return None
    if not _HOST_RE.match(host) or host.endswith("."):
        return None
    return parsed


def is_valid_http_url(value):
    return parse_http_url(value) is not None


def normalize_host(value):
    """Policy for comparing a stored source with a URL hostname:
    lower case, no surrounding dots/spaces, no leading 'www.'."""
    host = value.strip().strip(".").lower()
    return host[4:] if host.startswith("www.") else host


def domain_matches(source, hostname):
    """True if source equals the hostname or the hostname is a subdomain of it."""
    src, host = normalize_host(source), normalize_host(hostname)
    return host == src or host.endswith("." + src)


def check_url(row, available, article_id):
    if "url" not in available or is_missing(row.get("url")):
        return []                                        # FIELD_001 already covers this
    parsed = parse_http_url(row["url"])
    if parsed is None:
        return [make_issue("URL_001", "ERROR", "row", "url",
                           "URL is not a plausible HTTP(S) URL with a hostname.", article_id)]
    source = row.get("source")
    if "source" in available and not is_missing(source) \
            and not domain_matches(source, parsed.hostname):
        return [make_issue("URL_002", "WARNING", "row", "source",
                           f"Source '{source}' does not match URL hostname "
                           f"'{parsed.hostname}'.", article_id)]
    return []


# ---------------------------------------------------------------- time --------------
def parse_date(value):
    """date / datetime / ISO text -> date, or None if unusable."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def parse_timestamp(value):
    """datetime / ISO text -> datetime (aware or naive as given), or None."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def published_after_import(pub_date, pub_time, imported_at,
                           tolerance_days=None):
    """True if the publication moment is later than imported_at.

    - date + time WITH offset: exact comparison in UTC.
    - date only, or a time without offset: compare dates only, with a tolerance, because
      a naive time is never assumed to be UTC.
    """
    tolerance_days = config.TIME_FUTURE_TOLERANCE_DAYS if tolerance_days is None \
        else tolerance_days
    if imported_at.tzinfo is None:
        imported_utc_date = imported_at.date()
    else:
        imported_utc_date = imported_at.astimezone(timezone.utc).date()
    if isinstance(pub_time, time) and pub_time.tzinfo is not None \
            and pub_time.utcoffset() is not None and imported_at.tzinfo is not None:
        moment = datetime.combine(pub_date, pub_time.replace(tzinfo=None),
                                  tzinfo=timezone(pub_time.utcoffset()))
        return moment > imported_at
    return pub_date > imported_utc_date + timedelta(days=tolerance_days)


def check_times(row, available, article_id):
    issues = []
    earliest = date.fromisoformat(config.EARLIEST_PLAUSIBLE_DATE)
    pub_date = imported_at = None

    if "published_date" in available:
        raw = row.get("published_date")
        if is_missing(raw):
            issues.append(make_issue("TIME_005", "INFO", "row", "published_date",
                                     "Publication date is missing; not guessed.", article_id))
        else:
            pub_date = parse_date(raw)
            if pub_date is None:
                issues.append(make_issue("TIME_002", "WARNING", "row", "published_date",
                                         "Publication date is present but cannot be parsed.",
                                         article_id))
            elif pub_date < earliest:
                issues.append(make_issue("TIME_004", "WARNING", "row", "published_date",
                                         f"Publication date {pub_date} is before "
                                         f"{earliest}; verify the source.", article_id))
    if "imported_at" in available:
        raw = row.get("imported_at")
        if is_missing(raw):
            issues.append(make_issue("TIME_001", "INFO", "row", "imported_at",
                                     "Import timestamp missing; traceability is reduced.",
                                     article_id))
        else:
            imported_at = parse_timestamp(raw)
            if imported_at is None:
                issues.append(make_issue("TIME_002", "WARNING", "row", "imported_at",
                                         "Import timestamp is present but cannot be parsed.",
                                         article_id))
    if pub_date is not None and imported_at is not None:
        pub_time = row.get("published_time") if "published_time" in available else None
        if published_after_import(pub_date, pub_time, imported_at):
            issues.append(make_issue("TIME_003", "WARNING", "row", "published_date",
                                     "Publication is later than the import time; verify "
                                     "timezone and source metadata.", article_id))
    return issues


# ---------------------------------------------------------------- content ----------
_ERROR_RE = re.compile("|".join(config.ERROR_PATTERNS), re.IGNORECASE)


def is_placeholder_title(title):
    """True for generic titles ('Home', 'Untitled', 'Page not found') or titles with no
    letters or digits at all."""
    plain = re.sub(r"[^\w\s]", " ", title.casefold())
    plain = re.sub(r"\s+", " ", plain).strip()
    return not any(ch.isalnum() for ch in plain) or plain in config.PLACEHOLDER_TITLES


_NON_ALNUM = re.compile(r"[\W_]+")


def alnum_ratio(text):
    """Share of non-space characters that are letters or digits (any alphabet).
    Done with C-speed string operations because it runs on every article body."""
    compact = len("".join(text.split()))
    return len(_NON_ALNUM.sub("", text)) / compact if compact else 0.0


def check_content(row, available, article_id):
    issues = []
    title, body = row.get("title"), row.get("body")
    if "title" in available and isinstance(title, str) and not is_missing(title):
        title = title.strip()
        if len(title) < config.MIN_TITLE_CHARS_WARNING:
            issues.append(make_issue("TEXT_001", "WARNING", "row", "title",
                                     f"Title is shorter than {config.MIN_TITLE_CHARS_WARNING} "
                                     f"characters.", article_id))
        if is_placeholder_title(title):
            issues.append(make_issue("TEXT_004", "WARNING", "row", "title",
                                     "Title looks like a placeholder or error page, not a "
                                     "headline.", article_id))
    if "body" in available and isinstance(body, str) and not is_missing(body):
        body = body.strip()
        if len(body) < config.MIN_CONTENT_CHARS_WARNING:
            issues.append(make_issue("TEXT_002", "WARNING", "row", "body",
                                     f"Body is shorter than {config.MIN_CONTENT_CHARS_WARNING} "
                                     f"characters.", article_id))
        if len(body) <= config.ERROR_PATTERN_MAX_CHARS and _ERROR_RE.search(body):
            issues.append(make_issue("TEXT_003", "WARNING", "row", "body",
                                     "Body may be a block page, CAPTCHA or access-error "
                                     "response.", article_id))
        if alnum_ratio(body) < config.MIN_ALNUM_RATIO:
            issues.append(make_issue("TEXT_005", "WARNING", "row", "body",
                                     "Body is mostly punctuation or symbols.", article_id))
    return issues


# Flags already computed by the preprocessing stage are reused, never recomputed.
FLAG_RULES = (
    ("is_short", "FLAG_001", "Preprocessing flagged the body as short (under 50 words)."),
    ("is_non_article", "FLAG_002", "Preprocessing flagged the page as a non-article page."),
    ("has_html_tags", "FLAG_003", "Preprocessing found HTML-like tags in the body."),
    ("has_boilerplate", "FLAG_004", "Preprocessing found cookie or newsletter text."),
    ("has_suspicious_encoding", "FLAG_005",
     "Text still shows wrong-encoding patterns after repair."),
)


def check_flags(row, available, article_id):
    return [make_issue(rule_id, "WARNING", "row", flag, message, article_id)
            for flag, rule_id, message in FLAG_RULES
            if flag in available and row.get(flag) is True]


# ---------------------------------------------------------------- duplicates --------
def normalize_url(url):
    """Key for 'is this the same address?': http/https and www ignored, host lower-cased,
    fragment and tracking parameters dropped, query sorted, trailing slash dropped.
    Returns None if the URL is not valid."""
    parsed = parse_http_url(url) if isinstance(url, str) else None
    if parsed is None:
        return None
    port = parsed.port
    port_text = f":{port}" if port not in (None, 80, 443) else ""
    path = parsed.path.rstrip("/")
    query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                   if k.lower() not in config.TRACKING_PARAMS
                   and not k.lower().startswith(config.TRACKING_PARAM_PREFIXES))
    query_text = "&".join(f"{k}={v}" for k, v in query)
    return f"{normalize_host(parsed.hostname)}{port_text}{path}?{query_text}"


def signature_key(title, body):
    """Key for 'same title and same body': case-folded, whitespace collapsed."""
    if is_missing(title) and is_missing(body):
        return None
    norm = lambda t: re.sub(r"\s+", " ", (t or "").casefold()).strip()  # noqa: E731
    return norm(title) + "\x1f" + norm(body)


def _digest(key):
    return hashlib.blake2b(key.encode("utf-8"), digest_size=6).hexdigest()


class DuplicateIndex:
    """Two-pass duplicate detection that stays small in memory.

    Pass 1: call add(row) for every row (only 12-character digests are counted).
    Pass 2: call check(row) for every row to get the DUP_* issues.
    """

    def __init__(self):
        self.url_counts = Counter()
        self.sig_counts = Counter()

    def add(self, row):
        key = normalize_url(row.get("url"))
        if key:
            self.url_counts[_digest(key)] += 1
        sig = signature_key(row.get("title"), row.get("body"))
        if sig:
            self.sig_counts[_digest(sig)] += 1

    def check(self, row):
        issues, article_id = [], row.get("id")
        key = normalize_url(row.get("url"))
        if key:
            group = _digest(key)
            n = self.url_counts[group]
            if n > 1:
                issues.append(make_issue(
                    "DUP_001", "WARNING", "row", "url",
                    f"Normalised URL is shared by {n} articles; it may be a duplicate or an "
                    f"updated version of the same article.", article_id, group))
        sig = signature_key(row.get("title"), row.get("body"))
        strong = False
        if sig:
            group = _digest(sig)
            n = self.sig_counts[group]
            if n > 1:
                strong = True
                issues.append(make_issue(
                    "DUP_002", "WARNING", "row", "title/body",
                    f"Identical title and body shared by {n} articles: strong duplicate "
                    f"candidate (not confirmed, not removed).", article_id, group))
        size = row.get("duplicate_group_size")
        if not strong and isinstance(size, int) and size > 1:
            later = " This is a later copy." if row.get("is_duplicate") is True else ""
            body_hash = row.get("body_hash")
            issues.append(make_issue(
                "DUP_003", "WARNING", "row", "body",
                f"Identical body (preprocessing body_hash) shared by {size} articles, with "
                f"different titles: duplicate candidate.{later}", article_id,
                body_hash[:12] if isinstance(body_hash, str) else None))
        return issues


# ---------------------------------------------------------------- one row -----------
def validate_row(row, available):
    """Run every row-level rule that the available columns allow."""
    article_id = row.get("id")
    issues = []
    issues += check_fields(row, available, article_id)
    issues += check_url(row, available, article_id)
    issues += check_times(row, available, article_id)
    issues += check_content(row, available, article_id)
    issues += check_flags(row, available, article_id)
    return issues


# ---------------------------------------------------------------- disposition -------
ACCEPT, WARNING, QUARANTINE = "ACCEPT", "WARNING", "QUARANTINE"


def disposition(issues):
    """QUARANTINE if any fatal rule fired; else WARNING if any ERROR/WARNING issue;
    else ACCEPT. INFO findings are logged but never change the outcome."""
    if any(i["rule_id"] in config.FATAL_RULES for i in issues):
        return QUARANTINE
    if any(i["severity"] in ("ERROR", "WARNING") for i in issues):
        return WARNING
    return ACCEPT
