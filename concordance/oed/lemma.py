"""Is an oed.entry headword the lemma (base/citation) form of the word?

OED gives many inflected forms their own headword entry, most visibly
participial adjectives ("abandoned"/"ppl. a", "alcoholized"/"ppl. a") and
historical past-tense/past-participle forms ("v Pa. t", "pa. pple") -- these
are real headwords with their own definitions, but for a future cross-
reference into concordance.word (§ concordance-vocab1-import-flag) they're
noise: the root word ("abandon", "alcoholize") is what belongs in a
vocabulary list, not every inflected spin-off OED happened to catalogue.

Two-tier approach, in order:

  1. part_of_speech present (46% of entries) -- OED's own raw tag, OCR'd
     into ~90 distinct abbreviation strings (a/sb/v/adv/prep/int/conj/pron,
     plus historical-form tags ppl. a / pa. pple / pa. t, often combined:
     "a sb", "v Pa. t pa. pple"). Normalized into a small set of categories
     (_normalize_pos); any historical-form category present forces VERB-style
     lemmatization (that's specifically what those tags mean: "this headword
     is an inflected form of a verb"), overriding any co-listed a/sb/adv
     tags. Otherwise every listed category is tried and the headword counts
     as a lemma only if NONE of them would reduce it -- multi-sense entries
     like "a sb" (adjective and noun both spelled the same) are citation-form
     under both readings anyway, so this rarely matters in practice.

  2. part_of_speech NULL (54% of entries) -- nothing to normalize, so fall
     back to spaCy's own context-free guess (nlp(word), single-word "sentence")
     and the same distrust-a-dubious-reduction heuristic tokenize.py already
     uses for book text (_resolve_lemma): only treat a changed lemma as real
     if it's itself an attested word and not implausibly rarer than the
     surface form. Keep-biased per design philosophy -- an unreliable single-
     word POS guess should not silently exclude a real headword.
"""

from __future__ import annotations

# Public: also reused by resolve.py's OED definition tier (_from_oed) to
# pick a homograph entry by the tagger's coarse POS, the same way
# localdict._pick_entry/mw.pick_entry already do for their own sources.
POS_TOKEN_MAP = {
    "sb": "NOUN", "n": "NOUN", "N": "NOUN",
    "v": "VERB", "V": "VERB", "vb": "VERB",
    "a": "ADJ", "A": "ADJ", "adj": "ADJ",
    "adv": "ADV",
    "prep": "ADP",
    "int": "INTJ",
    "conj": "CCONJ",
    "pron": "PRON",
}

# Historical-form bigrams (raw tag split on whitespace, "." stripped) that mean
# "this headword IS an inflected form of a verb" -- forces VERB lemmatization
# regardless of any other tag on the same entry.
_HISTORICAL_VERB_FORM_BIGRAMS = {
    ("ppl", "a"),   # participial adjective: abandoned, alcoholized
    ("pa", "pple"),  # past participle
    ("pa", "t"),     # past tense
}


def _normalize_pos(raw: str | None) -> tuple[set[str], bool]:
    """-> (categories, is_historical_verb_form). Tolerant of OCR noise (case
    variants, stray periods like 'ppL a', 'a.') -- an unrecognized leftover
    token is simply dropped, not fatal, matching the rest of the OED
    pipeline's stance on OCR garbage."""
    if not raw or not raw.strip():
        return set(), False
    tokens = [t.strip(".").lower() for t in raw.split()]
    categories: set[str] = set()
    historical = False
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) in _HISTORICAL_VERB_FORM_BIGRAMS:
            historical = True
            i += 2
            continue
        pos = POS_TOKEN_MAP.get(tokens[i]) or POS_TOKEN_MAP.get(tokens[i].upper())
        if pos:
            categories.add(pos)
        i += 1
    return categories, historical


def pos_categories(raw: str | None) -> set[str]:
    """Public wrapper around _normalize_pos for callers that only need the
    coarse spaCy-style categories (resolve.py's OED definition tier, picking
    a homograph entry by cand.pos) and don't care about the historical-verb-
    form flag that's specific to is_lemma's own use."""
    categories, _ = _normalize_pos(raw)
    return categories


def _force_lemma(nlp, word: str, pos: str) -> str:
    from spacy.tokens import Doc

    doc = Doc(nlp.vocab, words=[word], pos=[pos])
    doc = nlp.get_pipe("lemmatizer")(doc)
    return doc[0].lemma_


def is_lemma(nlp, headword_norm: str, part_of_speech: str | None) -> bool:
    """True if `headword_norm` is already its own base/citation form."""
    categories, historical = _normalize_pos(part_of_speech)

    if historical:
        return _force_lemma(nlp, headword_norm, "VERB") == headword_norm

    if categories:
        return all(_force_lemma(nlp, headword_norm, pos) == headword_norm for pos in categories)

    # No usable OED tag -- fall back to spaCy's own context-free guess, with
    # the same "don't trust an implausible reduction" heuristic tokenize.py
    # applies to book text.
    from .. import tokenize

    doc = nlp(headword_norm)
    if not len(doc):
        return True
    resolved = tokenize._resolve_lemma(doc[0])
    return resolved == headword_norm
