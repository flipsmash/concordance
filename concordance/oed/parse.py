"""Entry raw text -> structured fields (part_of_speech, etymology, senses,
quotations). Regex-based, deliberately simple, and the least-validated part
of this pipeline — unlike pronunciation.py, there's no pilot-measured
accuracy number here yet. Expect this to need real iteration once run
against broad output; sense/quotation splitting in particular is a first
pass, not a calibrated parser. author/source_title on quotations carry
whatever baked-OCR noise was in the source prose (see quotation table
comment in db.py) — not cleaned up here.
"""

from __future__ import annotations

import re

# Common OED part-of-speech / grammatical abbreviations, checked as a run of
# tokens right after the headword (and past any inline pronunciation-bracket
# noise from the baked OCR layer — pronunciation itself is never read from
# this text, see pronunciation.py).
_POS_TOKEN = re.compile(
    r"\b(a|adv|adj|n|v|vb|v\.i|v\.t|v\.t\.|v\.i\.|sb|ppl\.?\s*a|pa\.?\s*pple|"
    r"pa\.?\s*t|prep|conj|int|pron|comb\.?\s*form)\.?\b",
    re.IGNORECASE,
)

_ETYMOLOGY_RE = re.compile(r"\[(.+?)\]", re.DOTALL)

# Sense markers: "1.", "2.", "a.", "b." (etc.) at what was a paragraph start
# in the source. The OCR text has already lost paragraph breaks in many
# cases, so this matches inline too — a real source of over/under-splitting
# that needs checking against broader output.
_SENSE_RE = re.compile(r"(?:(?<=\s)|^)(\d{1,2}|[a-z])\.\s+(?=[A-Z(])")

_YEAR_RE = re.compile(r"\b(a|c)?(1[0-9]{3}|20[0-2][0-9])\b")


_POS_CHAIN_SEP = re.compile(r"^\s*(and|or)\s+", re.IGNORECASE)


def split_pos(text: str) -> tuple[str | None, str]:
    """Pull a leading run of POS abbreviations off `text`; returns
    (pos_string_or_None, remainder). Handles a chain like "adv. and prep."
    (common — many OED headwords list more than one POS) by stripping the
    separating punctuation/conjunction between matches, not just before the
    first one."""
    pos_tokens = []
    remainder = text
    while True:
        stripped = remainder.lstrip(" .,;:")
        m = _POS_TOKEN.match(stripped)
        if not m:
            break
        pos_tokens.append(m.group(0))
        remainder = stripped[m.end():]
        chain = _POS_CHAIN_SEP.match(remainder.lstrip(" .,;:"))
        if chain:
            remainder = remainder.lstrip(" .,;:")[chain.end():]
    pos = " ".join(pos_tokens) if pos_tokens else None
    return pos, remainder.strip()


def looks_like_definition_entry(headword: str, raw_text: str) -> bool:
    """True if the text right after `headword` opens with a pronunciation
    bracket or a POS abbreviation -- the two things every real OED entry
    has and a false-positive headword detection (an ordinary body-text word
    that happened to sit at a column's left margin at headword-ish size,
    e.g. "form", "one", "great", "whence") never does. This is the
    discriminator the user asked for ("they will always be in bold"): the
    PDF's PyMuPDF span metadata carries no usable bold/flags signal for
    this document (confirmed empirically -- every span reports
    font='Courier', flags=8 regardless of visual weight), so bold can't be
    read from the text layer directly. What CAN be read reliably is what
    immediately follows a real headword, which serves the same purpose.
    Measured against page_number>=213 of Volume I: 728/1798 (40%) of
    then-current candidates failed this check, and the failures were
    exactly the class of word the user flagged (form, one, common, due,
    woman, whence, ...) while passes looked like genuine entries (address,
    aggregate, album, ...).
    Known cost: bare cross-reference entries with no pronunciation of their
    own (e.g. "abbotess, variant of abbatess, Obs., abbess.") are dropped
    by this too. Accepted -- those aren't definition entries either.
    Also known: a handful (13/182 already-transcribed pronunciations,
    checked directly) of entries with a leading usage label ("Obs.",
    "rare.") before the bracket, or that already look mis-sliced/corrupted
    for unrelated reasons (headword doesn't match the transcribed
    pronunciation at all), still fail this and are dropped too. Not chased
    further -- small, and some of those source rows are already wrong
    regardless of this check.
    """
    body = raw_text.strip()
    hw = headword.strip()
    if body.lower().startswith(hw.lower()):
        body = body[len(hw):]
    body = body.lstrip()
    # A homograph superscript number (e.g. "accidence 2 ('aeksidans)...")
    # sits between the headword and the bracket for same-spelling,
    # different-origin entries -- the baked OCR renders it as a plain
    # digit with a space. Without stripping it first, entries like
    # accidence/adage/adder (all with an already-resolved, clearly correct
    # pronunciation) wrongly failed this check.
    m = re.match(r"^\d{1,2}\s+", body)
    if m:
        body = body[m.end():]
    if body.startswith("("):
        return True
    return bool(_POS_TOKEN.match(body))


def extract_etymology(text: str) -> tuple[str | None, str]:
    """First bracketed run -> etymology; returns (etymology_or_None, text
    with that bracket removed)."""
    m = _ETYMOLOGY_RE.search(text)
    if not m:
        return None, text
    etym = m.group(1).strip()
    remainder = text[: m.start()] + text[m.end():]
    return etym, remainder


def split_senses(text: str) -> list[dict]:
    """Split entry body text into senses. Falls back to a single
    sense_label=None block if no numbered/lettered markers are found."""
    matches = list(_SENSE_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        if not stripped:
            return []
        return [{"sense_label": None, "definition_text": stripped,
                  "quotations": extract_quotations(stripped)}]

    senses = []
    lead = text[: matches[0].start()].strip()
    if lead:
        senses.append({"sense_label": None, "definition_text": lead,
                        "quotations": extract_quotations(lead)})
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end(): end].strip()
        if not body:
            continue
        senses.append({"sense_label": m.group(1), "definition_text": body,
                        "quotations": extract_quotations(body)})
    return senses


def extract_quotations(sense_text: str) -> list[dict]:
    """Split a sense's body on year tokens into dated citations. Best-effort:
    author/source_title is a naive grab of capitalized words right after the
    year, not a real citation parser."""
    matches = list(_YEAR_RE.finditer(sense_text))
    if not matches:
        return []
    quotations = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sense_text)
        chunk = sense_text[m.end(): end].strip()
        if not chunk:
            continue
        year_raw = m.group(0)
        year = int(m.group(2))
        year_approx = m.group(1) is not None
        header_match = re.match(r"([A-Z][\w.]*(?:\s+[A-Z][\w.]*){0,4})\s+(.*)", chunk, re.DOTALL)
        if header_match:
            author = header_match.group(1).strip()
            quoted_text = header_match.group(2).strip()
        else:
            author = None
            quoted_text = chunk
        quotations.append({
            "year_raw": year_raw, "year": year, "year_approx": year_approx,
            "author": author, "source_title": None, "quoted_text": quoted_text,
        })
    return quotations


def parse_entry(headword: str, raw_text: str) -> dict:
    """raw_text is the full slice from this headword's row to the next
    headword's row (see pipeline.py). Returns {part_of_speech, etymology,
    senses}."""
    # Drop the headword itself (and, best-effort, a leading inline
    # pronunciation-bracket artifact from the baked OCR text) before parsing
    # POS/etymology/senses out of the body.
    body = raw_text
    if body.strip().lower().startswith(headword.strip().lower()):
        body = body.strip()[len(headword.strip()):]
    # Strip a leading inline pronunciation-bracket artifact, then any
    # leftover comma/period/whitespace it leaves behind (e.g. "abaft
    # (a'baift, -ae-), adv. and prep." -> ", adv. and prep." after the
    # bracket alone is stripped) — without this second strip, split_pos's
    # re.match anchors on the stray comma and finds zero POS tokens even
    # though "adv." is right there, which was silently zeroing out POS
    # extraction on a lot of entries.
    body = re.sub(r"^\s*\([^)]{0,40}\)", "", body.strip())
    body = body.lstrip(" ,.;:")

    pos, body = split_pos(body)
    etymology, body = extract_etymology(body)
    senses = split_senses(body)
    return {"part_of_speech": pos, "etymology": etymology, "senses": senses}
