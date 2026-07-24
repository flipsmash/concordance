"""Archive full-text metadata (§ archive metadata) -- word-count/vocabulary-
richness stats computed locally from each book's own text, plus a best-
effort publication date pulled from Project Gutenberg's per-book catalog
metadata.

Every text in archive/ is a Project Gutenberg plain-text release (see
cli.py's `ingest` command, which moves incoming/ files here after
processing), wrapped in Gutenberg's own standard boilerplate (license
text, a "Release date" header) between a `*** START OF ... ***` and
`*** END OF ... ***` marker -- stripped here before counting so
word_count/distinct_nonstop_word_count describe the actual book, not the
~1-2k words of shared license text every file also carries.

Publication date: Gutenberg's RDF catalog metadata has no reliable
structured field for a book's ORIGINAL publication date -- only its own
`dcterms:issued` (when GUTENBERG digitized/released it, not when the work
was written). Its auto-generated `marc520` summary sometimes states an
exact year for well-documented classics, but at this corpus's real scale
a live random sample of 30 books came back 0/30 with an exact year --
almost every book only gets a hedge like "written in the early 20th
century." So this extracts BOTH: publication_year (rare, exact, only
when confidently parsed) and publication_era (the free-text hedge, far
more common) rather than forcing an exact date nobody actually has.
"""

from __future__ import annotations

import re

_START_RE = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.MULTILINE | re.IGNORECASE)
_END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*\s*$", re.MULTILINE | re.IGNORECASE)
_EBOOK_ID_RE = re.compile(r"\[eBook #(\d+)\]", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]+")

# marc520 states an exact year for a handful of globally-famous classics
# ("published in 1876"); everything else in this corpus (a live sample:
# 0/30 random books) only hedges with a century/decade phrase. Both
# patterns anchor on the same small set of reporting verbs so an unrelated
# year/era mentioned elsewhere in the summary (a historical event, a
# character's era) doesn't get mistaken for the book's own publication info.
_YEAR_RE = re.compile(
    r"\b(?:published|written|composed)\b(?:(?!\.).){0,60}?\b(1[4-9]\d{2}|20[0-2]\d)\b",
    re.IGNORECASE,
)
_ERA_RE = re.compile(
    r"\b(?:published|written|composed)\b(?:(?!\.).){0,60}?"
    r"\b((?:early|mid|late)[\s-](?:1[4-9]|20)th\s+century"
    r"|(?:early|mid|late)\s+\d{4}s"
    r"|\d{4}s"
    r"|\d{3,4}\s*BC)\b",
    re.IGNORECASE,
)

_stopword_set: frozenset[str] | None = None


def _stopwords() -> frozenset[str]:
    """NLTK's standard English stopword list (lazy, cached, auto-downloaded
    on first use) -- mirrors tokenize.py's own lazy-corpus-load pattern for
    its NLTK word list. Deliberately not spaCy's `is_stop` (tokenize.py's
    own filter): this module is the "fast, no spaCy" path chosen for
    processing ~11.5k archive files without loading an NLP model per file."""
    global _stopword_set
    if _stopword_set is None:
        try:
            from nltk.corpus import stopwords
            _stopword_set = frozenset(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _stopword_set = frozenset(stopwords.words("english"))
    return _stopword_set


def strip_gutenberg_boilerplate(text: str) -> str:
    """Everything between the START/END markers -- the actual book, not
    Gutenberg's license preamble/footer. Falls back to the full text
    unchanged if a marker is missing (confirmed live: true for well under
    1% of this corpus) so a malformed file still gets SOME stats rather
    than none."""
    start = _START_RE.search(text)
    end = _END_RE.search(text)
    if start and end and end.start() > start.end():
        return text[start.end():end.start()]
    if start:
        return text[start.end():]
    return text


def extract_gutenberg_id(text: str) -> int | None:
    """The [eBook #NNNN] id from the Release date header -- confirmed
    present in ~every file in this corpus (11519/11519 in a full scan)."""
    m = _EBOOK_ID_RE.search(text)
    return int(m.group(1)) if m else None


def word_stats(text: str) -> tuple[int, int]:
    """(word_count, distinct_nonstop_word_count) -- total alphabetic tokens,
    and the size of the vocabulary excluding NLTK's stopword list. Plain
    regex tokenization, not spaCy: no lemmatization (run/running/runs count
    as 3 distinct words), which is the deliberate fast-path tradeoff for
    processing this many files without loading an NLP model."""
    words = _WORD_RE.findall(text)
    stop = _stopwords()
    distinct_nonstop = {w.lower() for w in words if w.lower() not in stop}
    return len(words), len(distinct_nonstop)


def fetch_publication_info(gutenberg_id: int, timeout: float = 15.0) -> tuple[int | None, str | None]:
    """(publication_year, publication_era) from Gutenberg's per-book RDF
    catalog metadata -- see this module's own docstring for why both exist
    and why neither is guaranteed. Network errors/timeouts/missing fields
    all just come back as (None, None) -- a failed lookup for one book
    shouldn't abort a run processing thousands of them."""
    import requests

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.rdf"
    try:
        resp = requests.get(url, timeout=timeout)
        if not resp.ok:
            return None, None
    except requests.RequestException:
        return None, None

    m = re.search(r"<pgterms:marc520>(.*?)</pgterms:marc520>", resp.text, re.DOTALL)
    if not m:
        return None, None
    summary = m.group(1)

    year_match = _YEAR_RE.search(summary)
    year = int(year_match.group(1)) if year_match else None

    era_match = _ERA_RE.search(summary)
    era = era_match.group(1).strip() if era_match else None

    return year, era
