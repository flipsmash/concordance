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

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def representative(self) -> Occurrence | None:
        """A sentence to show the user — the shortest that still has real context."""
        usable = [o for o in self.occurrences if len(o.sentence.split()) >= 4]
        pool = usable or self.occurrences
        return min(pool, key=lambda o: len(o.sentence)) if pool else None
