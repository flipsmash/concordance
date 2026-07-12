"""Validity likelihood for words that stay undefined (§05 follow-on).

For a word no dictionary could define, estimate how likely it is a *real* word
(archaic/dialectal/nonce) versus an artifact (OCR error, old-spelling variant of
a modern word, foreign token, or nonsense). Output is a 0..1 score, a 3-way label
(likely-valid / uncertain / likely-artifact), a human-readable notes string, and a
suggested correction when the word looks like a misspelling of something commoner.

Every signal is deterministic and explainable — no LLM opinion, per the project
rule that an LLM must never be the authority on whether a word is real:

  + Google Books Ngram      appeared in published books at all? (nonsense = 0)
  + wordfreq zipf           present in a broad web corpus
  + WordNet / NLTK corpora   vouched by a curated wordlist
  + morphology              decomposes to a known root (un+geniture+d)
  - SymSpell near-neighbor   a much commoner word one/two edits away => OCR/variant
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests
from wordfreq import zipf_frequency

_NGRAM = "https://books.google.com/ngrams/json"

# Order-of-magnitude Zipf jump to a near neighbor that marks a misspelling/variant
# (matches validity.py's misspelling_zipf_gap).
_NEIGHBOR_GAP = 1.5

# Context-language check: an English rare word sits in an English sentence; a
# Latin/French token sits among other foreign tokens. If too few of the context
# sentence's words are common English, the whole line is another language and the
# target word is probably foreign too.
_FOREIGN_MIN_TOKENS = 5
_FOREIGN_ENGLISH_FRACTION = 0.5
_COMMON_ENGLISH_ZIPF = 3.0

_PREFIXES = ("un", "re", "over", "under", "dis", "mis", "out", "be", "en", "im", "in")
_SUFFIXES = ("edly", "ing", "ed", "es", "s", "er", "est", "ness", "less", "ly",
             "ure", "ment", "ish", "y", "d")


@dataclass
class ValidityEstimate:
    word: str
    score: float
    label: str
    notes: str
    suggestion: str = ""


# --- lazily-built shared resources ---------------------------------------

_sym = None
_words = None


def _symspell():
    global _sym
    if _sym is None:
        from importlib.resources import files
        from symspellpy import SymSpell
        s = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        s.load_dictionary(str(files("symspellpy") / "frequency_dictionary_en_82_765.txt"), 0, 1)
        _sym = s
    return _sym


def _wordset():
    global _words
    if _words is None:
        try:
            from nltk.corpus import words as nltk_words
            _words = set(w.lower() for w in nltk_words.words())
        except Exception:
            _words = set()
    return _words


def _in_wordnet(word: str) -> bool:
    try:
        from nltk.corpus import wordnet
        return bool(wordnet.synsets(word))
    except Exception:
        return False


# --- signals --------------------------------------------------------------

def _ngram_peak(word: str, session: requests.Session | None) -> float | None:
    """Peak relative frequency in Google Books (1500–2019). None if unavailable."""
    getter = session.get if session is not None else requests.get
    try:
        r = getter(_NGRAM, params={"content": word, "year_start": 1500, "year_end": 2019,
                                   "corpus": "en-2019", "smoothing": 3}, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return 0.0
        return max(data[0].get("timeseries", [0.0]) or [0.0])
    except (requests.RequestException, ValueError):
        return None


def _dominant_neighbor(word: str) -> tuple[str, float] | None:
    """A near neighbor (<=2 edits) that is much commoner than the word itself —
    evidence the word is a misspelling / OCR / old-spelling variant of it."""
    from symspellpy import Verbosity
    self_z = zipf_frequency(word, "en")
    best = None
    for s in _symspell().lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False):
        if s.term == word:
            continue
        z = zipf_frequency(s.term, "en")
        # Require a real common word, not a proper name (Carrie/Barrie pollute the
        # 82k list): in WordNet, or plainly frequent.
        if not (_in_wordnet(s.term) or z >= 4.0):
            continue
        if z - self_z >= _NEIGHBOR_GAP and (best is None or z > best[1]):
            best = (s.term, z)
    return best


def _morph_root(word: str) -> str | None:
    """A known root reachable by peeling one prefix and/or one suffix."""
    def known(w):
        return len(w) >= 3 and (zipf_frequency(w, "en") >= 2.0 or w in _wordset() or _in_wordnet(w))

    cands = {word}
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            cands.add(word[: -len(suf)])
    for base in list(cands):
        for pre in _PREFIXES:
            if base.startswith(pre) and len(base) - len(pre) >= 3:
                cands.add(base[len(pre):])
    for c in cands:
        if c != word and known(c):
            return c
    return None


def effective_zipf(word: str, zipf: float | None = None) -> float:
    """The Zipf frequency to use for rarity/frequency-floor decisions.

    wordfreq undercounts a transparent prefixed/suffixed form relative to its
    root even though the derivation adds no real difficulty (unbuttoned vs
    button, overplayed vs play, quickly vs quick) — so when `word` reduces to
    a known common root via `_morph_root`, use whichever of the two Zipfs is
    higher rather than the artificially low Zipf of the derived form itself."""
    word = word.strip().lower()
    z = zipf_frequency(word, "en") if zipf is None else zipf
    root = _morph_root(word)
    if root:
        z = max(z, zipf_frequency(root, "en"))
    return z


# --- scoring --------------------------------------------------------------

def _english_fraction(sentence: str) -> tuple[float, int]:
    """(share of sentence tokens that are common English words, token count)."""
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", sentence) if len(t) > 1]
    if not toks:
        return 1.0, 0
    n_eng = sum(1 for t in toks if zipf_frequency(t, "en") >= _COMMON_ENGLISH_ZIPF)
    return n_eng / len(toks), len(toks)


def estimate(word: str, zipf: float | None = None,
             session: requests.Session | None = None,
             sentence: str = "") -> ValidityEstimate:
    word = word.strip().lower()
    zf = zipf_frequency(word, "en") if zipf is None else zipf
    ng = _ngram_peak(word, session)
    in_corpus = (word in _wordset()) or _in_wordnet(word)
    neighbor = _dominant_neighbor(word)
    root = _morph_root(word)

    score = 0.35
    notes = []

    if ng is None:
        notes.append("ngram unavailable")
    elif ng > 0:
        score += 0.10
        notes.append("in Google Books")
    else:
        score -= 0.25
        notes.append("absent from Google Books")

    if zf > 0:
        score += 0.05
        notes.append(f"web zipf {zf:.1f}")
    if in_corpus:
        score += 0.40                       # curated wordlist — the strongest signal
        notes.append("in WordNet/NLTK wordlist")
    if root:
        score += 0.15
        notes.append(f"root '{root}'")
    suggestion = ""
    if neighbor:
        score -= 0.45
        suggestion = neighbor[0]
        notes.append(f"near '{neighbor[0]}' (zipf {neighbor[1]:.1f}) — likely variant/OCR")

    score = max(0.0, min(1.0, score))

    # A foreign-language context sentence is decisive: whatever corpus/ngram credit
    # the token earned, if it lives among non-English words it is very likely not an
    # English word. Cap it down into the artifact range.
    frac, ntok = _english_fraction(sentence)
    if ntok >= _FOREIGN_MIN_TOKENS and frac < _FOREIGN_ENGLISH_FRACTION:
        score = min(score, 0.25)
        notes.append(f"foreign-language context ({frac*100:.0f}% English)")
    label = "likely-valid" if score >= 0.6 else "likely-artifact" if score <= 0.35 else "uncertain"
    return ValidityEstimate(word=word, score=round(score, 2), label=label,
                            notes="; ".join(notes), suggestion=suggestion)
