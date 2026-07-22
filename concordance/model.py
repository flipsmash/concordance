"""Core data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """Where a token ended up. Mirrors §05 of the spec."""

    KEEP = "keep"        # a real word, passed to the LLM judge / review
    DROP = "drop"        # ruled out with corroborated evidence (junk/misspelling/name)
    UNSURE = "unsure"    # failed validity but recurs in the book — routed to review, never silently cut


class RejectReason(str, Enum):
    FREQUENCY_FLOOR = "frequency_floor"   # too common — stop-word-like
    PROPER_NOUN = "proper_noun"
    MISSPELLING = "misspelling"           # dominant higher-frequency near-neighbor
    NOT_A_WORD = "not_a_word"             # no authority vouched + no corpus presence
    NUMERIC_OR_SYMBOL = "numeric_or_symbol"
    FOREIGN_LANGUAGE = "foreign_language"  # sits in a non-English quoted phrase/sentence
    NOT_INTERESTING = "not_interesting"   # real, but the LLM judged it unremarkable
    ALREADY_KNOWN = "already_known"       # user marked it known during review


@dataclass
class Occurrence:
    """One appearance of a token in the book."""

    sentence: str
    chapter: str
    surface: str          # the inflected form exactly as it appeared


@dataclass
class Candidate:
    """A distinct lemma under consideration, with everything a later stage needs."""

    lemma: str
    pos: str                                    # coarse part of speech from the tagger
    occurrences: list[Occurrence] = field(default_factory=list)
    zipf: float = 0.0                           # general-English frequency (wordfreq)
    cap_ratio: float = 0.0                      # share of mid-sentence appearances that were capitalized
    propn_ratio: float = 0.0                    # share of appearances the tagger called PROPN
    ent_ratio: float = 0.0                      # share of appearances inside a named entity

    # Filled in by later stages:
    verdict: Verdict | None = None
    reject_reason: RejectReason | None = None
    validity_sources: list[str] = field(default_factory=list)   # which authorities vouched
    interesting_reason: str = ""                # one-line rationale from the LLM judge

    # Enrichment (dictionary lookup + sense pick):
    definition: str = ""
    part_of_speech: str = ""
    ipa: str = ""
    synonyms: list[str] = field(default_factory=list)
    etymology: str = ""
    definition_source: str = ""

    # A human-review flag (never an auto-reject — see validity_score.
    # variant_reject_reason's docstring for why): set when a definition
    # source successfully defined the word, but it looks like a foreign
    # word or an archaic/OCR spelling of a common modern word.
    variant_flag_reason: str = ""
    variant_flag_note: str = ""

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def representative(self) -> Occurrence | None:
        """A sentence to show the user — the shortest that still has real context."""
        usable = [o for o in self.occurrences if len(o.sentence.split()) >= 4]
        pool = usable or self.occurrences
        return min(pool, key=lambda o: len(o.sentence)) if pool else None


# Canonical, spelled-out part-of-speech vocabulary. Every write site (dictionary
# enrichment, the spaCy-tag fallback, hand-edited CSVs, the webapp's rescue
# path) has its own source of truth for this label, so they've drifted into a
# mess of abbreviations (adj, adv, pron, adp, sconj, num), spaCy's raw
# universal-tag casing, and stray Title-Case typos. This is the one place that
# folds all of it down to one consistent set.
_POS_ALIASES = {
    "n": "noun", "noun": "noun", "nouns": "noun",
    "v": "verb", "verb": "verb", "verbs": "verb",
    "adj": "adjective", "adjective": "adjective", "adjectives": "adjective",
    "adv": "adverb", "adverb": "adverb", "adverbs": "adverb",
    "pron": "pronoun", "pronoun": "pronoun", "pronouns": "pronoun",
    "adp": "preposition", "prep": "preposition",
    "preposition": "preposition", "prepositions": "preposition",
    "conj": "conjunction", "cconj": "conjunction", "sconj": "conjunction",
    "conjunction": "conjunction", "conjunctions": "conjunction",
    "intj": "interjection", "interjection": "interjection", "interjections": "interjection",
    "det": "determiner", "determiner": "determiner", "determiners": "determiner",
    "num": "numeral", "numeral": "numeral", "numerals": "numeral", "number": "numeral",
    "aux": "auxiliary", "auxiliary": "auxiliary",
    "part": "particle", "particle": "particle",
    "punct": "punctuation", "punctuation": "punctuation",
    "propn": "proper noun", "proper noun": "proper noun",
    "sym": "symbol", "symbol": "symbol",
    "x": "other", "other": "other",
}


def normalize_pos(pos: str | None) -> str:
    """Fold any spelling/case/abbreviation variant to the canonical label.
    Blank/unrecognized input stays '' (this project's existing "no value"
    convention for text fields, not NULL — see quiz_definition etc.)."""
    if not pos:
        return ""
    key = pos.strip().lower()
    return _POS_ALIASES.get(key, key)


# Categories a dictionary lookup can resolve a candidate to that are never
# real vocabulary for this project's purposes, regardless of what earlier
# pipeline stages decided — a "symbol" hit (ISO language codes, roman-numeral
# alt-case-form pages) or "proper noun" hit (the dictionary itself confirming
# a leaked name) is grounds to cast the word out. This is the single choke
# point for that check: every code path that acts on an enrichment-resolved
# POS (initial ingest, the refill/deepen backfills, and the webapp's rescue
# endpoint) should call `junk_pos_reason`/`is_junk_pos` rather than
# reimplementing the category list.
JUNK_POS_REASON: dict[str, RejectReason] = {
    "symbol": RejectReason.NUMERIC_OR_SYMBOL,
    "proper noun": RejectReason.PROPER_NOUN,
}


def is_junk_pos(pos: str | None) -> bool:
    return normalize_pos(pos) in JUNK_POS_REASON


def junk_pos_reason(pos: str | None) -> RejectReason | None:
    """The RejectReason to use if `pos` (any spelling/case) is never real
    vocabulary here, else None."""
    return JUNK_POS_REASON.get(normalize_pos(pos))
