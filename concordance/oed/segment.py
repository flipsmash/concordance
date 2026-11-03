"""Headword detection (§ entry boundaries).

A headword is a flush-left row whose first word is printed bold, with extra
leading above it. Both signals come from the page itself rather than the OCR
text, which is why this beats the earlier size-and-pattern detectors (~70% ->
~96% of definition entries on hand-labelled pages):

  * bold: measured from the scan (extract.stroke_width). The text layer reports
    every span as the same font, so bold can't be read from it directly.
  * leading: the row sits more than `headword_gap_min` line pitches below the
    one above it. A bold row that opens with a pronunciation and a part of
    speech is exempt: OCR row tops are too noisy to refuse one that plain.

The text then only has to reject what slips through. Entries are alphabetical,
so the page's running heads ("SLENDERLY ... SLEUTH") bound what can appear; a
candidate outside that range, or too short to trust, must also look like an
entry (a comma, pronunciation bracket, part of speech or label after the word).
Pages whose running heads can't be read fall back on the range the page's own
candidates imply. Lists of bold combinations inside an entry are the main thing
that range check stops: each item reads like a headword with a comma after it.

The OCR spells a word's markers into or beside it, and normalize_headword undoes
that: OED's obsolete dagger is read as a glued or separate "f"/"t"/"+", the
foreign-word mark ‖ as "Il"/"ll"/"II", and stress marks lead or sit inside the
word. A glued marker is told from a real leading letter by the alphabetical
range (slep is in range, fslep isn't).

One-line cross-references ("sleuce, obs. form of sluice sb.") are real terms but
carry no definition of their own: classify_pointer marks them and names the word
they point at.

Not attempted: run-in sub-entries inside another entry's text ("Hence 'slenger",
"raised beach"). They start mid-row, so nothing here sees them.
"""

from __future__ import annotations

import collections
import re

from . import extract
from .config import OedConfig

_stopword_set: frozenset[str] | None = None


def _stopwords() -> frozenset[str]:
    """Same lazy NLTK stopword load as archive_metadata.py::_stopwords() --
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


_RUNNING_HEAD_RE = re.compile(r"[A-Z][A-Z'’\-]{1,}\.?")
_RUNNING_HEAD_ZONE = 0.06           # top fraction of the page where the running heads may sit
_RUNNING_HEAD_ROW_PT = 14           # their depth: a column's first row starts about 18pt below them
_DAGGER_RE = re.compile(r"^(?:[†+‡]|[tf]['’])")
_PARALLEL_RE = re.compile(r"^(?:\|\||[Il1|]{2})(?=[A-Za-z])")     # OED's ‖ as the OCR spells it
_STRESS_RE = re.compile(r"['’ˈˌ‘`]")
_WORD_RE = re.compile(r"^[a-z][a-z\-æœ.]{0,29}$")
_KEY_RE = re.compile(r"[^a-z]")
# A dagger or ‖ read as letters of the word, lowercased.
_GLUED_MARKS = ("il", "ll", "ij", "j-", "f", "t", "h", "i", "l", "u")
_MARKER_SPAN_RE = re.compile(r"[^A-Za-z]*[A-Za-z]?[^A-Za-z]*|\|\||[Il1]{2}")   # a lone dagger, ‖ or stress mark
_HEAD_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*[,.;:]?")
_HEAD_ENDED_RE = re.compile(r"[,;:.)]$|\(")
_MAX_HEAD_WORDS = 3

# A pronunciation bracket straight after the headword ("sneck (snek), v.").
_PRON_START_RE = re.compile(r"^[\s,]*\(['’ˈ,a-zA-Z]")
# A whole opening: word, pronunciation, then a part of speech or label. Tight
# enough that a bold row reading this way needs no extra leading above it;
# body text's asides "(see ...)", "(cf. ...)", "(Pliny)" don't get through.
_PRON_OPENING_RE = re.compile(
    r"^[^\s(]{2,}\s*\((?!(?i:see|cf|e\.g|esp|in|ibid|also|quot|ed)\b)(?:[^()]|\([^()]*\)){2,40}\)\s*[,.]?\s*"
    r"(?:(?:adv|sb|ppl|vbl|int|prep|conj|pron|a|v)\s?\.|\[|(?:Also|Forms?|Pl|Obs|Now|Chem|Bot|Zool)\b)")
_LINE_POS_RE = re.compile(r"(?<![A-Za-z])(a|adv|adj|sb|v|vb|ppl|pa|prep|conj|int|pron|n)\s?\.(?![A-Za-z])")
_LABEL_START_RE = re.compile(r"^[\s,.]*(?:\d{1,2}\s*)?(?:\[|Forms?\b|Also\b|Pl\b|[A-Z][a-z]{1,9}\.|rare\b|nonce\b|dial\b|"
                             r"colloq\b|combining form|comb\. form|[a-z]+\.\s*:?\s*see\b|:\s*see\b)")

# "snarring: see snar v.", "snogly, adv.: see snog", "skeery, variant of scary"
# -- a pointer straight after the headword (optionally a POS/label or two).
_SEE_POINTER_RE = re.compile(r"^[\s,]*(?:(?:[A-Za-z\-]+\.?|etc\.?)[\s,]*){0,3}:\s*see\b|^[\s,]*variants?\s+of\b"
                             r"|^\s*see\b|^[\s,]*irreg\.\s+var\.",   # colon glued to the headword token
                             re.IGNORECASE)
# One-line pointers, not definition entries: "var.", "ff.", "obs. form of",
# "obs. (Sc.) f. soar v.", "obs. pa. t. of shove", "obs. comp. of soon",
# "vbl. sb. 2 : see snort", "corruption of brand-new", "error for swyper".
# (A bare ": see" is NOT one -- real entries' etymology brackets say
# "[f. L. ...: see -ize]".)
_XREF_RE = re.compile(
    r"\b(varr?\.|ff\.|obs\.\s+forms?\b(?!\s*:)|(?<!combining )forms?\s+of\b|variants?\s+of\b|corruption\s+of\b|error\s+for\b"
    r"|obs\.\s+(?:(?:Sc|dial|north)\.\s+)*(?:f\.|pa\.|pl\.|comp\b|superl\b)"
    r"|(?:Sc\.\s+)?pa\.\s*(?:t\.|pples?\.)(?:\s+(?:and\s+)?pa\.\s*pples?\.)?\s+of\b"
    r"|(?:vbl\.\s*sb\.|sb\.|v\.)\s*\d*\s*:\s*see\b|:\s*see\b)", re.IGNORECASE)
_XREF_WINDOW = 60                   # a pointer says so right after the headword
_GLOSS_OPENER_RE = re.compile(r"\b(?:A|An|The|One|To|Of|That|Having)\s")
_TARGET_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]+")
_TARGET_SKIP = frozenset("""of form forms under also the see var varr variant variants obs pa pple pples pl
    comp superl erron error for dial sc me north south and or sb adv etc irreg corruption ppl vbl mod
    ff note to pref suff""".split())
_TARGET_RELATIVE = frozenset(("prec", "next", "above", "below"))    # "= next": no word to follow


def in_range(key: str, rng: tuple[str, str]) -> bool:
    lo, hi = rng
    return key >= lo[: len(key)] and key[: len(hi)] <= hi


def _key(text: str) -> str:
    return _KEY_RE.sub("", text.lower())


def _near_range(key: str, rng: tuple[str, str]) -> bool:
    """In range on its first two letters: the test for a word with a marker
    stuck to it, whose later letters the OCR may also have got wrong."""
    return rng[0][:2] <= key[:2] <= rng[1][:2]


def normalize_headword(token: str, rng: tuple[str, str] | None = None) -> tuple[str | None, bool]:
    """(clean headword | None, obsolete) from a headword token: strips the OCR'd
    dagger (recording it -- in OED a dagger marks the entry obsolete), the ‖
    mark, stress marks, trailing punctuation and a "(e" spelling tail. A bare
    leading f/t, or a leading Il/ll, only counts as a marker when that's what
    brings the word into the page's range."""
    t = token.strip()
    obsolete = bool(_DAGGER_RE.match(t))
    t = _DAGGER_RE.sub("", t)
    m = _PARALLEL_RE.match(t)
    if m:
        bare = t[m.end():]
        if rng:
            if in_range(_key(bare), rng) and not in_range(_key(t), rng):
                t = bare
        elif t[:2] in ("II", "||"):
            t = bare
    t = _STRESS_RE.sub("", t)
    t = re.sub(r"[,;:]+$", "", t)
    t = re.sub(r"\(([a-z]+)\)(?=[a-z])", r"\1", t)     # slep(e)ry: the optional letters stay
    t = re.sub(r"\(.*$", "", t)
    t = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", t.lower()).rstrip(".")   # OCR dot inside a word
    if not _WORD_RE.match(t):
        return None, False
    if rng and not in_range(_key(t), rng):
        for mark in _GLUED_MARKS:
            bare = t[len(mark):]
            if t.startswith(mark) and len(bare) >= 3 and _near_range(_key(bare), rng):
                return bare, obsolete or mark in "ft"
    return t, obsolete


def headword_key(text: str) -> str:
    """Letters only, lowercase -- the comparison key for alphabetical range
    checks and for matching a detected hit to a stored entry (whose headword
    may still carry a dagger/stress/comma: "t'wombclout")."""
    return _key(normalize_headword(text)[0] or text.lower())


def body_top(spans: list[dict], page_height: float) -> float:
    """Where the body starts: below the page's topmost row when that's in the
    running-head zone. Not the zone's own edge, which on some volumes' page
    sizes falls below a column's first row."""
    top = min((s["bbox"][1] for s in spans), default=page_height)
    return top + _RUNNING_HEAD_ROW_PT if top < page_height * _RUNNING_HEAD_ZONE else top - 1


def running_head_range(spans: list[dict], page_height: float) -> tuple[str, str] | None:
    """(first, last) headword keys from the page's running heads, or None.
    Heads sit at the page's outer edges: the folio between them can OCR as
    capitals ("491" -> "III"), so a head in the middle third is ignored. Only
    the topmost row counts: the zone also reaches a column's first row, whose
    "VIII" of a citation would otherwise end the range."""
    first_row = min((s["bbox"][1] for s in spans), default=page_height) + _RUNNING_HEAD_ROW_PT
    top = sorted((s for s in spans if s["bbox"][1] < min(page_height * _RUNNING_HEAD_ZONE, first_row)),
                 key=lambda s: s["bbox"][0])
    if not top:
        return None
    width = max(s["bbox"][2] for s in spans)
    heads = [s for s in top if _RUNNING_HEAD_RE.fullmatch(s["text"].strip())
             and not width / 3 < (s["bbox"][0] + s["bbox"][2]) / 2 < width * 2 / 3]
    if len(heads) < 2:
        return None
    lo, hi = _key(heads[0]["text"]), _key(heads[-1]["text"])
    return (lo, hi) if lo and hi and lo <= hi else None


def classify_pointer(rest: str) -> tuple[bool, str | None]:
    """Is this entry, given its text after the headword, a one-line pointer?
    -> (is_pointer, target word | None). The target is the first real word
    after the pointer phrase, lowercased; OED's relative "see next" has none."""
    head = rest[: rest.find("[")] if "[" in rest else rest
    if _PRON_START_RE.match(head):
        return False, None
    m = _SEE_POINTER_RE.match(head)
    if not m:
        m = _XREF_RE.search(head[:_XREF_WINDOW])
        if not m or _GLOSS_OPENER_RE.search(head[: m.start()]):
            return False, None
    after = head[m.end():]
    of = re.match(r"(?:[^A-Z]{0,24}?)\bof\b", after)
    if of:
        after = after[of.end():]
    for word in _TARGET_WORD_RE.finditer(after[:_XREF_WINDOW]):
        w = word.group(0).lower().replace("’", "'").strip("'-")
        if w in _TARGET_RELATIVE:
            return True, None
        if w and len(w) > 1 and w not in _TARGET_SKIP:
            return True, w
    return True, None


def _has_entry_opening(token: str, rest: str) -> bool:
    """Is this bold word followed by a pronunciation, part of speech or label,
    or a pointer phrase, rather than by prose that happens to be heavy?"""
    return bool("(" in token or _PRON_START_RE.match(rest) or _LABEL_START_RE.match(rest)
                or _LINE_POS_RE.search(rest[:24]) or classify_pointer(rest)[0])


def _inferred_range(tokens: list[str]) -> tuple[str, str] | None:
    """A stand-in range when the running heads are unreadable: the two-letter
    prefix most of the page's bold words share, if most of them share one."""
    prefixes = collections.Counter(k[:2] for k in map(headword_key, tokens) if len(k) >= 2)
    if not prefixes:
        return None
    prefix, count = prefixes.most_common(1)[0]
    return (prefix, prefix + "zzzz") if count >= max(2, len(tokens) / 2) else None


def _word_span(row_spans: list[dict]) -> tuple[dict | None, list[dict], bool]:
    """(the row's headword span, the spans after it, obsolete) -- stepping past
    any dagger, ‖ or stress mark the OCR left as spans of their own ("t", "'",
    "totted,")."""
    i = 0
    obsolete = False
    while i < len(row_spans) - 1 and _MARKER_SPAN_RE.fullmatch(t := row_spans[i]["text"].strip()):
        obsolete = obsolete or t[:1] in ("t", "f", "+", "†", "‡")
        i += 1
    word = row_spans[i]
    if i >= 2 and len(row_spans[i - 1]["text"].strip()) == 1 and row_spans[i - 1]["text"].strip().isalpha():
        # After a mark, a lone letter is the word's first, split off: "t", "u", "n'soilable,"
        first = row_spans[i - 1]
        word = {**word, "text": first["text"].strip() + word["text"].strip(),
                "bbox": [first["bbox"][0], *word["bbox"][1:]]}
    return word, row_spans[i + 1:], obsolete


def find_headwords(col_rows: list[list[dict]], margins: list[float], page_number: int, cfg: OedConfig,
                   page_height: float, gray) -> list[dict]:
    """Headwords on one page, in reading order. Each hit: `headword` (clean),
    `text` (the leading span text, which entry parsing strips), `bbox` (the
    headword span -- pronunciation.py builds its crop from this), `size`,
    `page`, `obsolete`, `row_id` (id() of the row it starts) and `also` (further
    bold spellings on the row: "wombill, womble, obs. ff. wimble")."""
    rows_all = [r for rows in col_rows for r in rows]
    spans_all = [s for r in rows_all for s in r["spans"]]
    median = extract.median_stroke(gray, spans_all, cfg.ink_zoom)
    if not median:
        return []

    def ink(span: dict) -> float:
        x0, y0, x1, y1 = span["bbox"]
        t = span["text"].strip()
        mark = _PARALLEL_RE.match(t)
        if mark:                            # the ‖'s two thin strokes would dilute a bold word
            x0 += (x1 - x0) * mark.end() / len(t)
        return (extract.stroke_width(gray, (x0, y0, x1, y1), cfg.ink_zoom) or 0.0) / median

    def bold(span: dict) -> bool:
        return cfg.headword_ink_min <= ink(span) <= cfg.headword_ink_max

    top = body_top(spans_all, page_height)
    candidates = []
    for rows, margin in zip(col_rows, margins):
        rows = [r for r in rows if extract.row_top(r) > top]
        tops = [extract.row_top(r) for r in rows]
        pitches = sorted(b - a for a, b in zip(tops, tops[1:]) if 4 < b - a < 30)
        pitch = pitches[len(pitches) // 2] if pitches else 10.0
        for i, row in enumerate(rows):
            row_spans = sorted(row["spans"], key=lambda s: s["bbox"][0])
            if abs(row_spans[0]["bbox"][0] - margin) > cfg.left_margin_tolerance:
                continue
            word, after, dagger = _word_span(row_spans)
            t = word["text"].strip()
            if t[:1] in "[(" or (t.isupper() and len(t) > 1) or t.endswith((";", "]")):
                continue                    # bracket text, a small-caps cross-reference, a list item
            opening = " ".join(s["text"].strip() for s in [word, *after])
            if (i and (tops[i] - tops[i - 1]) / pitch < cfg.headword_gap_min
                    and not _PRON_OPENING_RE.match(opening)):
                continue
            if bold(word):
                candidates.append((row, row_spans, word, after, dagger, t))

    rng = running_head_range(spans_all, page_height)
    tokens = [c[5] for c in candidates]
    if tokens and (rng is None or sum(in_range(headword_key(t), rng) for t in tokens) < len(tokens) * 0.3):
        rng = _inferred_range(tokens)       # unreadable, or misread into nonsense
    stops = _stopwords()

    hits = []
    for row, row_spans, word, after, dagger, t in candidates:
        hw, obsolete = normalize_headword(t, rng)
        if not hw:
            continue
        head_spans = [word]
        while (len(head_spans) < _MAX_HEAD_WORDS and after
               and not _HEAD_ENDED_RE.search(head_spans[-1]["text"].strip())
               and _HEAD_WORD_RE.fullmatch(after[0]["text"].strip()) and bold(after[0])):
            head_spans.append(after[0])
            after = after[1:]
        for extra in head_spans[1:]:
            part, _ = normalize_headword(extra["text"], None)
            if part:
                hw = f"{hw} {part}"
        last = head_spans[-1]["text"].strip()
        rest = " ".join(s["text"] for s in after)
        key = _key(hw)
        opening = _has_entry_opening(last, rest)
        if len(key) < 2:
            continue
        if rng and in_range(key, rng):
            if not opening and not last.endswith((",", ":")) and (len(key) < 4 or hw in stops):
                continue
        elif not opening or len(key) < 3 or hw in stops:
            # A comma isn't enough here: outside the range it's usually an item
            # in a combinations list ("resin, any of various"), which starting
            # an entry would cut out of the one it belongs to.
            continue
        also = []
        for s in after:
            word_t = s["text"].strip()
            if word_t.startswith("-") or not _HEAD_WORD_RE.fullmatch(word_t) or not bold(s):
                break
            extra_hw, _ = normalize_headword(word_t, rng)
            if extra_hw and len(_key(extra_hw)) >= 3 and extra_hw != hw:
                also.append(extra_hw)
            if not word_t.endswith(","):
                break
        lead_end = len(row_spans) - len(after)
        hits.append({"text": " ".join(s["text"] for s in row_spans[:lead_end]).strip(), "headword": hw,
                     "obsolete": dagger or obsolete, "bbox": list(word["bbox"]), "size": word["size"],
                     "page": page_number, "row_id": id(row), "also": also})
    return hits
