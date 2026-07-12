"""Stage 5 — strip proper nouns (§04).

Layered, cheapest test first:
  1. the tagger — spaCy called it PROPN or put it inside a named entity;
  2. in-book capitalization ratio — catches invented names (Aragorn, Winterfell)
     that no tagger knows, by how consistently the word is capitalized in
     mid-sentence position across the whole book.
The model backstop (an explicit "reject names" instruction) lives in the judge.

The hard case — names that collide with real words (Baker, Rose, Mark, or
Ulysses's own Leopold Bloom) — is why the capitalization ratio matters: a
real name is capitalized virtually every time it's used, so cap_ratio alone
still separates "the baker" from "Mr. Baker" (or Bloom the character from
"bloom" the flower) better than any dictionary check could; a common word
that only sometimes gets capitalized by convention will show a lower ratio.

The tagger signal is weaker than that: spaCy's statistical PROPN/entity tag
demonstrably misfires on ordinary words in unusual syntax (a real run on
Ulysses called "tram", "beggar", "alderman", and "sacrament" proper nouns —
none of them ever capitalized in the actual text) without requiring ANY
capitalization evidence at all. So the tagger alone is corroborated further:
a well-established dictionary word doesn't get dropped on the tagger's
say-so — only on the capitalization ratio, which a genuine collision case
(Bloom) still trips independently.
"""

from __future__ import annotations

from importlib.resources import files

from .config import Config
from .model import Candidate, RejectReason, Verdict

_sym_words: frozenset[str] | None = None


def _common_dictionary_word(lemma: str) -> bool:
    """SymSpell's 82k general-vocabulary wordlist — the same curated authority
    validity.py trusts. A personal name is rarely a headword there even when
    it's a common surname (Byrne, Rochford); an ordinary word the tagger
    mistagged (tram, beggar, alderman, sacrament) always is."""
    global _sym_words
    if _sym_words is None:
        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=0, prefix_length=1)
        dict_path = files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
        _sym_words = frozenset(sym.words)
    return lemma in _sym_words


def strip_proper_nouns(candidates: dict[str, Candidate], cfg: Config) -> None:
    for cand in candidates.values():
        if cand.verdict is not None:
            continue
        # Real mid-sentence capitalization is a strong, standalone signal.
        capitalization_says_name = cand.cap_ratio >= cfg.cap_ratio_threshold
        # The tagger alone is unreliable on a lone sentence-initial token
        # (where every word is capitalized). A single-occurrence PROPN always
        # has ratio 1.0, so ratio can't corroborate — recurrence must. This
        # keeps one-off words like "Motes of dust…" out of the proper-noun
        # bucket; a true one-off invented name still gets dropped downstream by
        # the validity gate (unattested, no dictionary vouch).
        tagger = cand.propn_ratio >= 0.5 or cand.ent_ratio >= 0.5
        tagger_corroborated = (
            tagger and cand.count >= 2 and not _common_dictionary_word(cand.lemma)
        )
        if capitalization_says_name or tagger_corroborated:
            cand.verdict = Verdict.DROP
            cand.reject_reason = RejectReason.PROPER_NOUN
