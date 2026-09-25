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


# --- v2 detector (`oed-ingest --add-missing`) ------------------------------
#
# Measured 2026-09-24: find_headwords_from_columns above keeps ~65% of real
# definition entries on Volume I but only ~15-35% on the later volumes. The
# losses are almost all WORD_RE, not size: later volumes print a comma after
# nearly every headword ("slent,", "slerg,"), the obsolete dagger is OCR'd as a
# glued-on "f"/"t'" ("fsleng", "t'slenker"), and stress marks lead or sit
# inside the word ("'slenting", "anagra'mmatically"). Size is secondary: real
# headwords measure ~1.26-1.29x body on those scans, just under the 1.30 bar,
# while some body words reach 1.30 too -- so size can't separate them alone.
#
# v2 normalizes those prefixes/suffixes and replaces most of the size rule with
# the page's own running heads ("SLENDERLY ... SLEUTH"): entries are
# alphabetical, so a candidate must fall inside that range. That also tells a
# glued-on dagger from a real leading f/t (slep is in range, fslep isn't).
# Pages whose running heads can't be read fall back to v1's size bar.

_RUNNING_HEAD_RE = re.compile(r"[A-Z][A-Z'’\-]{1,}\.?")
_DAGGER_RE = re.compile(r"^(?:[†+‡]|[tf]['’])")
_STRESS_RE = re.compile(r"['’ˈˌ‘`]")
_V2_WORD_RE = re.compile(r"^[a-z][a-z\-æœ.]{1,29}$")
_KEY_RE = re.compile(r"[^a-z]")
# "snarring: see snar v.", "snogly, adv.: see snog", "skeery, variant of scary"
# -- a pointer straight after the headword (optionally a POS/label or two).
_SEE_POINTER_RE = re.compile(r"^[\s,]*(?:(?:[A-Za-z]+\.?|etc\.?)[\s,]*){0,3}:\s*see\b|^[\s,]*variants?\s+of\b"
                             r"|^\s*see\b|^[\s,]*irreg\.\s+var\.",   # colon glued to the headword token
                             re.IGNORECASE)
# A pronunciation bracket straight after the headword ("sneck (snek), v.").
_PRON_START_RE = re.compile(r"^[\s,]*\(['\u2019\u02c8a-zA-Z]")
# A margin row that is the middle of an entry, not its start: an etymology
# bracket closing ("sheugh sb. ] 1.", "soldier sb. + -y", "slow a.] One who")
# or a forms list running on ("(6 -lie); 5-6", ", (7, 9 erron.", ", 7 shaftmont").
_CONTINUATION_RE = re.compile(r"^[\s,]*(?:(?:[a-z]+\.\s*){0,2}[\]+]|\(?\d)")
_V2_SIZE_MULT = 1.15              # with a running-head range to guard
_V2_LOOSE_SIZE_MULT = 1.22        # for the order-insensitive first-line check (see validate_v2_hit)
_LINE_POS_RE = re.compile(r"(?<![A-Za-z])(a|adv|adj|sb|v|vb|ppl|pa|prep|conj|int|pron|n)\.(?![A-Za-z])")
# One-line pointers, not definition entries: "var.", "ff.", "obs. form of",
# "obs. (Sc.) f. soar v.", "obs. pa. t. of shove", "obs. comp. of soon",
# "vbl. sb. 2 : see snort". (A bare ": see" is NOT one -- real entries' etymology
# brackets say "[f. L. ...: see -ize]".)
_XREF_RE = re.compile(
    r"\b(varr?\.|ff\.|obs\.\s+forms?\b|forms?\s+of\b"
    r"|obs\.\s+(?:(?:Sc|dial|north)\.\s+)*(?:f\.|pa\.|pl\.|comp\b|superl\b)"
    r"|vbl\.\s*sb\.\s*\d*\s*:\s*see\b)", re.IGNORECASE)


def headword_key(text: str) -> str:
    """Letters only, lowercase -- the comparison key for alphabetical range
    checks and for matching a v2 hit to an entry v1 already stored (whose
    headword may still carry a dagger/stress/comma: "t'wombclout")."""
    return _KEY_RE.sub("", normalize_headword(text)[0] or text.lower())


def running_head_range(spans: list[dict], page_height: float) -> tuple[str, str] | None:
    """(first, last) headword keys from the page's running heads, or None."""
    top = sorted((s for s in spans if s["bbox"][1] < page_height * 0.06), key=lambda s: s["bbox"][0])
    heads = [s for s in top if _RUNNING_HEAD_RE.fullmatch(s["text"].strip())]
    if len(heads) < 2:
        return None
    lo, hi = _KEY_RE.sub("", heads[0]["text"].lower()), _KEY_RE.sub("", heads[-1]["text"].lower())
    return (lo, hi) if lo and hi and lo <= hi else None


def in_range(key: str, rng: tuple[str, str]) -> bool:
    lo, hi = rng
    return key >= lo[: len(key)] and key[: len(hi)] <= hi


def normalize_headword(token: str, rng: tuple[str, str] | None = None) -> tuple[str | None, bool]:
    """(clean headword | None, obsolete) from a row's first span: strips the
    OCR'd dagger (recording it -- in OED a dagger marks the entry obsolete),
    stress marks, trailing comma/semicolon and a "(e" spelling tail. A bare
    leading f/t is only treated as a dagger when that's what brings the word
    into the page's range."""
    t = token.strip()
    obsolete = bool(_DAGGER_RE.match(t))
    t = _DAGGER_RE.sub("", t)
    t = _STRESS_RE.sub("", t)
    t = re.sub(r"[,;:]+$", "", t)
    t = re.sub(r"\(.*$", "", t)
    t = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", t.lower()).rstrip(".")   # OCR dot inside a word
    if not _V2_WORD_RE.match(t):
        return None, False
    if (rng and t[0] in "ft" and len(t) > 3 and not in_range(_KEY_RE.sub("", t), rng)
            and in_range(_KEY_RE.sub("", t[1:]), rng)):
        t, obsolete = t[1:], True
    return t, obsolete


def find_headwords_v2(col_rows: list[list[dict]], margins: list[float], page_number: int,
                       cfg: OedConfig, page_height: float) -> list[dict]:
    """v2 headword candidates (see the section comment). Each hit also
    carries `headword` (clean), `obsolete`, `first_line` and `loose_ok` for
    validate_v2_hit. `text` stays the raw span so pipeline slicing/matching
    works exactly as for v1 hits."""
    all_spans = [s for rows in col_rows for row in rows for s in row["spans"]]
    body = body_size(all_spans)
    if body is None:
        return []
    rng = running_head_range(all_spans, page_height)
    size_bar = body * (_V2_SIZE_MULT if rng else cfg.headword_size_mult)
    stops = _stopwords()
    hits = []
    for rows, margin in zip(col_rows, margins):
        for row in rows:
            row_spans = sorted(row["spans"], key=lambda s: s["bbox"][0])
            first = row_spans[0]
            if abs(first["bbox"][0] - margin) > cfg.left_margin_tolerance or first["size"] < size_bar:
                continue
            raw = first["text"].strip()
            if (raw.isupper() and len(raw) > 1) or raw.endswith(";"):
                continue                        # small-caps cross-ref in a quotation / mid-entry list item
            hw, obsolete = normalize_headword(raw, rng)
            if not hw or hw.endswith("-") or len(_KEY_RE.sub("", hw)) < 4 or hw in stops:
                continue
            if rng and not in_range(_KEY_RE.sub("", hw), rng):
                continue
            line = " ".join(s["text"] for s in row_spans)
            rest = line.split(first["text"].strip(), 1)[-1]
            has_pron = bool(_PRON_START_RE.match(rest))
            if (_XREF_RE.search(rest) or _SEE_POINTER_RE.match(rest)) and not has_pron:
                continue                        # one-line "var./ff./obs. form of/: see X" pointer
            if _CONTINUATION_RE.match(rest):
                continue                        # mid-entry line: etymology "... sb.] 1." / forms "(6 -lie); 5-6"
            loose_ok = bool(rng and first["size"] >= body * _V2_LOOSE_SIZE_MULT
                            and ("(" in rest or _LINE_POS_RE.search(rest)))
            hits.append({"text": first["text"].strip(), "headword": hw, "obsolete": obsolete,
                         "bbox": list(first["bbox"]), "size": first["size"], "page": page_number,
                         "loose_ok": loose_ok})
    return hits
