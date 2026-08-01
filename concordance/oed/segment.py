"""Headword candidate detection (§ entry boundaries).

Two filters, both pilot-validated against real Volume I / Volume 16 pages:
  1. size: a row-leading span at least `headword_size_mult` * the page's body
     size (font-size heuristic — real headwords ran larger than surrounding
     prose on both volumes, though the absolute sizes differ per volume/scan).
  2. left margin: the span must start within `left_margin_tolerance` of a
     detected column margin. Size alone caught real headwords but also
     small-caps cross-references mid-paragraph (e.g. "ABBOT" inside
     "abbotship"'s etymology bracket) at a ~7% rate in the pilot sample —
     those never start flush at the column margin the way a real entry does.

Not attempted here: run-on/compound sub-entry classification (entry_type
beyond the 'main' default). That needs typographic cues (italic/bold family)
the baked OCR text layer doesn't preserve — every detected headword is
currently written as entry_type='main'. Left as a known gap; revisit once
real output volume makes the run-on/compound share of entries visible.
"""

from __future__ import annotations

import re

import fitz

from .config import OedConfig
from .extract import body_size, flatten_spans, page_columns

WORD_RE = re.compile(r"^[a-z][a-zA-Z\-æœ'.]{1,29}$")

_stopword_set: frozenset[str] | None = None


def _stopwords() -> frozenset[str]:
    """Same lazy NLTK stopword load as archive_metadata.py::_stopwords() —
    duplicated rather than imported since that's a private helper in an
    unrelated module; kept identical in behavior."""
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


def find_headwords(page: fitz.Page, cfg: OedConfig, *, skip_stopwords: bool = True) -> list[dict]:
    """Headword candidates on one page, top-to-bottom, in reading order.
    Each: {text, bbox, size, page}. `bbox` is the headword span's own box
    (not a crop region — pronunciation.py builds the generous crop from
    this anchor).

    Convenience wrapper that does its own page.get_text("dict") — that call
    measured ~330ms on a dense Volume I page, so pipeline.py (which needs
    the same column/row structure for its own reading-order slicing) calls
    find_headwords_from_columns directly instead of paying for it twice."""
    spans = flatten_spans(page)
    if not spans:
        return []
    col_rows, margins = page_columns(spans, page.rect.width)
    return find_headwords_from_columns(col_rows, margins, page.number, cfg, skip_stopwords=skip_stopwords)


def find_headwords_from_columns(col_rows: list[list[dict]], margins: list[float], page_number: int,
                                 cfg: OedConfig, *, skip_stopwords: bool = True) -> list[dict]:
    """col_rows[i] is column i's rows (already y-ordered) — see
    extract.page_columns. margins[i] is that column's own left edge, so the
    left-margin filter checks a headword candidate against ITS OWN column's
    margin, not against any margin on the page (checking against "any"
    margin was the earlier version's bug: a row merged across columns by a
    column-blind group_rows could spuriously match a different column's
    margin)."""
    all_spans = [s for rows in col_rows for row in rows for s in row["spans"]]
    body = body_size(all_spans)
    if body is None:
        return []
    threshold = body * cfg.headword_size_mult
    stops = _stopwords() if skip_stopwords else frozenset()

    hits = []
    for rows, margin in zip(col_rows, margins):
        for row in rows:
            row_spans = sorted(row["spans"], key=lambda s: s["bbox"][0])
            first = row_spans[0]
            text = first["text"].strip().rstrip(".")
            if not WORD_RE.match(first["text"].strip()):
                continue
            if first["size"] < threshold:
                continue
            if abs(first["bbox"][0] - margin) > cfg.left_margin_tolerance:
                continue
            if text.lower() in stops:
                continue
            hits.append({"text": first["text"].strip(), "bbox": list(first["bbox"]),
                          "size": first["size"], "page": page_number})
    return hits
