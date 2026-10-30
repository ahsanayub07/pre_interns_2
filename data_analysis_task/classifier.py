"""Local rule-based article classifier (no API, no network). Input: title and body text only.

classify(title, body) -> dict with category, sub_label, confidence, needs_review, reason,
secondary_topics, scores. The confidence is a heuristic from keyword evidence, NOT a probability.
"""
import re
from collections import Counter

import classifier_rules as R

CATEGORIES = ["market_stocks", "technology", "health", "energy", "politics", "other", "uncertain"]


def _alternation(terms):
    return "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))


def _build():
    compiled = {}
    for topic, spec in R.TOPICS.items():
        strong, weak = spec["terms"]
        weights = {t: R.STRONG for t in strong}
        weights.update({t: R.WEAK for t in weak if t not in weights})
        words = re.compile(r"\b(" + _alternation(weights) + r")(?:s|es)?\b", re.IGNORECASE)
        acr = re.compile(r"\b(" + _alternation(spec["acronyms"]) + r")s?\b") if spec["acronyms"] else None
        compiled[topic] = (words, acr, weights, set(spec["acronyms"]))
    other = re.compile(r"\b(" + _alternation(R.OTHER_HINTS) + r")(?:s|es)?\b", re.IGNORECASE)
    return compiled, other


_COMPILED, _OTHER = _build()
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def clean_title(title):
    """Remove a trailing site name such as ' - ZDNET' or ' | The National'."""
    title = (title or "").strip()
    for sep in (" | ", " - ", " – ", " — "):
        if sep in title:
            head, tail = title.rsplit(sep, 1)
            if len(tail.split()) <= 4 and len(head.split()) >= 3:
                return head.strip()
    return title


def language_check(text):
    """Return (ok, reason). Flags other scripts and text that is mostly another Latin-script language.

    English text with few common words (score lists, short items) is NOT flagged.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False, "no_letters"
    latin = sum(1 for c in letters if c.isascii() or "\u00c0" <= c <= "\u024f")
    if latin / len(letters) < R.MIN_LATIN_RATIO:
        return False, "non_latin_script"
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if len(words) >= R.FOREIGN_MIN_WORDS:
        english = sum(w in R.STOPWORDS for w in words)
        foreign = sum(w in R.FOREIGN_WORDS for w in words)
        if foreign / len(words) >= R.FOREIGN_MIN_RATIO and foreign > english:
            return False, "probably_not_english"
    return True, ""


def page_type(title):
    """'company_profile', 'deals_page' or 'news_article', decided from the ORIGINAL title."""
    if re.search(R.COMPANY_PROFILE_RE, title or "", re.IGNORECASE):
        return "company_profile"
    if re.search(R.DEALS_PAGE_RE, title or "", re.IGNORECASE):
        return "deals_page"
    return "news_article"


def normalise(text):
    """Straighten curly quotes so that gov\u2019t matches gov't."""
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def _terms(compiled, text):
    words, acr, _w, _a = compiled
    c = Counter(m.group(1).lower() for m in words.finditer(text))
    if acr:
        c.update(m.group(1) for m in acr.finditer(text))
    return c


def score_topic(topic, title, lead):
    """Score one topic. Each distinct keyword counts once for the title and once for the lead."""
    compiled = _COMPILED[topic]
    weights, acronyms = compiled[2], compiled[3]
    in_title, in_lead = _terms(compiled, title), _terms(compiled, lead)
    total, evidence = 0.0, []
    for term in set(in_title) | set(in_lead):
        w = R.STRONG if term in acronyms else weights.get(term, R.WEAK)
        part = 0.0
        if in_title[term]:
            part += w * R.TITLE_MULT
        if in_lead[term]:
            part += w * (1 + 0.25 * (min(in_lead[term], 4) - 1))
        total += part
        evidence.append((part, term, bool(in_title[term])))
    evidence.sort(reverse=True)
    return total, evidence


_MASK = re.compile("|".join(re.escape(p) for p in R.MASK_PHRASES), re.IGNORECASE)


def mask_phrases(text):
    """Blank out phrases like 'solar system' so their topic word does not count."""
    return _MASK.sub(" ", text)


_VIOLENCE = re.compile(r"\b(" + _alternation(R.VIOLENCE_HINTS) + r")(?:s|es)?\b", re.IGNORECASE)
_FOCUS = {k: re.compile(getattr(R, k), re.IGNORECASE) for k in
          ("FOCUS_TITLE_RE", "FOCUS_MONEY_RE", "FOCUS_DEAL_RE", "FOCUS_ASSET_RE", "FOCUS_MOVE_RE")}


def other_score(title, lead):
    """Evidence for `other`: 2.0 per distinct sports/entertainment/lifestyle/crime word, 1.0 per violence word."""
    text = title + " " + lead
    hits = Counter(m.group(1).lower() for m in _OTHER.finditer(text))
    vio = {m.group(1).lower() for m in _VIOLENCE.finditer(text)}
    return R.STRONG * len(hits) + 0.25 * (sum(hits.values()) - len(hits)) + R.VIOLENCE_WEIGHT * len(vio)


def finance_focus(title):
    """True when the TITLE is about money: a stock/ETF/earnings word, a priced deal, or an asset that moves."""
    f = _FOCUS
    return bool(f["FOCUS_TITLE_RE"].search(title)
                or (f["FOCUS_MONEY_RE"].search(title) and f["FOCUS_DEAL_RE"].search(title))
                or (f["FOCUS_ASSET_RE"].search(title) and f["FOCUS_MOVE_RE"].search(title)))


def _result(category, confidence, reason, scores, sub_label="", secondary=(), review=None, ptype="news_article"):
    confidence = round(max(0.0, min(0.99, confidence)), 2)
    return {"category": category, "sub_label": sub_label, "confidence": confidence,
            "needs_review": (confidence < R.REVIEW_BELOW or bool(secondary)) if review is None else review,
            "reason": reason, "secondary_topics": ";".join(secondary),
            "scores": {k: round(v, 2) for k, v in scores.items()}, "page_type": ptype,
            "rules_version": R.RULES_VERSION}


def classify(title, body):
    title_n = normalise(title)
    title_c = clean_title(title_n)
    lead = normalise(body)[: R.LEAD_CHARS]
    text = f"{title_c} {lead}"
    n_words = len(text.split())
    zero = {t: 0.0 for t in R.TOPICS}
    ptype = page_type(title_n)        # judged on the original title, before a site-name suffix is cut
    if ptype == "company_profile":
        return _result("uncertain", 0.8, "company profile page: template text, not a news article", zero,
                       review=False, ptype=ptype)
    if ptype == "deals_page":
        return _result("other", 0.8, "shopping or deals page, not a news article", zero, review=False, ptype=ptype)
    if n_words < R.MIN_WORDS:
        return _result("uncertain", 0.2, "too little text to judge", zero, review=True)
    ok, why = language_check(text)
    if not ok:
        return _result("uncertain", 0.2, f"not analysed: {why.replace('_', ' ')}", zero, review=True)

    scored = {t: score_topic(t, mask_phrases(title_c), mask_phrases(lead)) for t in R.TOPICS}
    if finance_focus(title_c):                      # v1.3: the title is about money
        s_m, ev_m = scored["market_stocks"]
        scored["market_stocks"] = (s_m + R.FOCUS_BONUS, sorted(ev_m + [(R.FOCUS_BONUS, "money focus in title", True)], reverse=True))
    scores = {t: s for t, (s, _e) in scored.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, s1 = ranked[0], scores[ranked[0]]
    second, s2 = ranked[1], scores[ranked[1]]
    secondary = tuple(t for t in ranked[1:] if scores[t] >= R.MIN_SCORE and scores[t] >= R.SECONDARY_RATIO * s1)
    o = other_score(title_c, lead)

    def why_text(topic):
        ev = scored[topic][1][:3]
        return f"{topic}: " + ", ".join(t + (" (title)" if in_t else "") for _p, t, in_t in ev)

    if o >= max(R.OTHER_DOMINATES_MIN, 1.5 * s1):
        return _result("other", 0.6 + min(0.3, 0.03 * o), "mostly wording from outside the five categories (sports, entertainment, science, crime...)", scores)
    if s1 < R.OTHER_BELOW:
        conf = 0.6 + (0.1 if o >= 2 else 0) - (0.15 if s1 >= 1.0 else 0)
        return _result("other", conf, "no clear topic evidence for the five categories", scores)
    if s1 < R.MIN_SCORE:
        return _result("uncertain", 0.35, "weak evidence: " + why_text(best), scores, review=True)
    if s1 - s2 < R.MIN_MARGIN or s1 < R.MIN_RATIO * s2:
        return _result("uncertain", 0.35, f"ambiguous between {best} and {second}", scores,
                       secondary=(best, second), review=True)
    share = s1 / (sum(scores.values()) or 1)
    conf = 0.5 * share + 0.5 * min(1.0, s1 / 8.0)
    sub = ""
    if best == "market_stocks":                     # crypto price/trading stories (decision: market_stocks)
        crypto = sum(p for p, t, _i in scored["technology"][1] if t in R.BLOCKCHAIN_TERMS)
        if crypto >= R.BLOCKCHAIN_MIN:
            sub = "crypto_markets"
    if best == "technology":
        crypto = sum(p for p, t, _i in scored["technology"][1] if t in R.BLOCKCHAIN_TERMS)
        if crypto >= R.BLOCKCHAIN_MIN:
            sub = "blockchain_technology"
    return _result(best, conf, why_text(best), scores, sub_label=sub, secondary=secondary)
