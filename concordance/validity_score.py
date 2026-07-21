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

_PREFIXES = ("un", "re", "over", "under", "dis", "mis", "out", "be", "en", "im", "in",
             "pre", "ir", "il", "non", "post", "sub", "super", "inter", "anti", "fore")
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


# --- hard-reject gates for a word a source ALREADY defined ----------------
#
# _dominant_neighbor above is a soft SCORING signal only, used exclusively on
# words nothing could define at all -- safe there because a wrong guess just
# nudges a score, never silently drops a word. The two functions below are
# for the opposite, riskier situation: a definition-lookup source (Wordnik,
# yourdictionary, web-search, even Free Dictionary) DID successfully return
# something for the word, and the question is whether to accept it as real
# vocabulary at all. That needs a much higher precision bar than the scoring
# signal -- see unambiguous_dominant_neighbor's docstring for the concrete
# false positive (bogoak/book) that ruled out reusing _dominant_neighbor
# verbatim for this purpose.

# A hard reject needs the SAME threshold ingest's ValidityGate already uses
# for its own misspelling drop (config.misspelling_zipf_gap) -- this is a
# second, independent enforcement point for that identical judgment call
# (see model docstring on why re-enforcement is necessary at all: the
# cross-book verdict cache makes an ingest-time KEEP sticky forever, and
# refill/deepen/fill_definitions never re-run ValidityGate).
_HARD_REJECT_GAP = 2.0

# Languages checked for the foreign-word gate, via wordfreq's own per-language
# corpora -- picked as the non-English languages most likely to appear as
# loanwords/quotations in this project's English-language literary corpus.
# Latin isn't in wordfreq's language list at all (no living web corpus to
# derive frequencies from), so a Latin fragment (nolunt, insidiis, prosequi)
# simply isn't caught here -- it falls through to validity_score.estimate's
# ordinary artifact scoring instead, same as before this gate existed; not a
# regression, just an uncovered case.
_FOREIGN_LANGS = ("fr", "it", "es", "de", "nl", "pt", "fi")
_FOREIGN_MIN_ZIPF = 3.0   # the word must be genuinely common THERE, not a one-off hit
_FOREIGN_GAP = 1.0        # ...and meaningfully commoner there than in English


def foreign_language_hint(word: str) -> tuple[str, float] | None:
    """(language, zipf) if `word` is clearly a foreign-language word rather
    than English -- e.g. acte (French, zipf 4.79 vs English 1.99), bellissimo
    (Italian 4.66 vs 1.66), auxilio (Spanish 3.98 vs 1.27). None if nothing
    clears both bars (a genuine English rarity like armiger/cangue/bogoak
    scores near-zero in every language checked, or the foreign showing is too
    marginal to trust -- montfaucon's French zipf of 2.65 falls under
    _FOREIGN_MIN_ZIPF, correctly left uncaught since it's actually a proper
    noun, a different problem this check isn't meant to solve)."""
    word = word.strip().lower()
    en_z = zipf_frequency(word, "en")
    best: tuple[str, float] | None = None
    for lang in _FOREIGN_LANGS:
        z = zipf_frequency(word, lang)
        if z >= _FOREIGN_MIN_ZIPF and z - en_z >= _FOREIGN_GAP and (best is None or z > best[1]):
            best = (lang, z)
    return best


def _shares_stem(a: str, b: str, prefix_len: int = 5) -> bool:
    """True if `a`/`b` are plausibly the same root (plural, adverb, etc. of
    each other) rather than two unrelated words that happen to be SymSpell
    neighbors of the same target -- apparel/apparels, beneficial/beneficially,
    not book/bogota/bogor (bogoak's actual SymSpell ties)."""
    return a.startswith(b) or b.startswith(a) or a[:prefix_len] == b[:prefix_len]


def unambiguous_dominant_neighbor(word: str) -> str | None:
    """Like _dominant_neighbor, but tuned for a hard reject rather than a
    soft score nudge: the neighbor must be the UNIQUE (up to trivial
    morphological variants) best SymSpell match, not just A match within 2
    edits. Without this, bogoak -- a real word (preserved bog wood) -- would
    get "corrected" to book purely because both sit within edit-distance 2
    and book is vastly more common; the giveaway is that bogoak ALSO ties
    with bogota/bogor/boga/bogon at the same distance, none related to each
    other or to book, whereas a genuine archaic-spelling variant like
    apparrell only ties with apparel/apparels -- the same root twice.
    Verified against this project's own real corpus words: correctly
    accepts assunder/beneficiall/apparrell/allyance/adventrous, correctly
    rejects bogoak/armiger/cangue/befalne/aftersong (either genuinely
    ambiguous, or a real word this project explicitly wants to keep)."""
    from symspellpy import Verbosity
    word = word.strip().lower()
    cands = [s for s in _symspell().lookup(word, Verbosity.CLOSEST, max_edit_distance=2,
                                            include_unknown=False) if s.term != word]
    if not cands:
        return None
    best = max(cands, key=lambda c: c.count)
    if any(c.term != best.term and not _shares_stem(c.term, best.term) for c in cands):
        return None  # a genuinely distinct second candidate -- too ambiguous to act on
    if not (_in_wordnet(best.term) or zipf_frequency(best.term, "en") >= 4.0):
        return None
    if zipf_frequency(best.term, "en") - zipf_frequency(word, "en") < _HARD_REJECT_GAP:
        return None
    return best.term


def variant_reject_reason(word: str) -> tuple["RejectReason", str] | None:
    """The single choke point every definition-acceptance call site should
    check on a word a source just successfully defined: (RejectReason,
    human-readable note) if it's clearly a foreign-language word or an
    archaic/OCR spelling variant of a common modern word, else None. Foreign
    is checked first since a word can occasionally trip both (acte both
    scores as French AND has English SymSpell neighbors like "act") --
    foreign is the more specific, more confident signal of the two."""
    from .model import RejectReason
    word = word.strip().lower()
    foreign = foreign_language_hint(word)
    if foreign:
        lang, z = foreign
        return RejectReason.FOREIGN_LANGUAGE, f"looks {lang} (zipf {z:.1f} there vs English)"
    neighbor = unambiguous_dominant_neighbor(word)
    if neighbor:
        return RejectReason.MISSPELLING, f"archaic/OCR spelling of '{neighbor}'"
    return None


def _morph_root(word: str) -> str | None:
    """The most common known root reachable by peeling a SINGLE prefix or a
    SINGLE suffix off `word` — e.g. unbuttoned -> buttoned, bemused -> mused.

    Deliberately single-affix only. An earlier version also chained a second
    peel onto the first (unbutton -> button), which is needed for a handful of
    genuine double-derivatives but, empirically, generates far more damage
    than value: any second peel can land on a real word that is unrelated to
    `word` by pure coincidence (impaled -[peel 'd']-> impale -[peel 'im']->
    'pale', bemused -[peel 'be']-> mused -[peel 'ed']-> 'mus'), and because
    effective_zipf() takes the MAX of word's own zipf and the root's, a
    coincidental hit that outranks the true root silently INFLATES a genuine
    rarity's effective frequency — which can push it over the floor and drop
    it before it ever reaches the validity gate or judge, with no backstop.
    A single peel that finds nothing, by contrast, just leaves `word` at its
    own (low) zipf — it proceeds to the floor/validity/judge as before, which
    is the safe direction per this project's keep-bias rule: false drops are
    forbidden, false survivals are cheap (the judge alone rejects roughly
    half of what reaches it in practice).

    Silent-e restoration: naively slicing a suffix off a word whose root ends
    in a dropped 'e' produces a truncated non-word (hoping -[peel 'ing']->
    'hop', not 'hope'; mused -[peel 'ed']-> 'mus', not 'muse'), and 'hop'/
    'mus' both happen to be real, unrelated words that would otherwise win.
    So every suffix peel also tries appending 'e' back and keeps whichever
    the `known()` gate accepts."""
    def known(w):
        return len(w) >= 3 and (zipf_frequency(w, "en") >= 2.0 or w in _wordset() or _in_wordnet(w))

    cands = set()
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            stem = word[: -len(suf)]
            cands.add(stem)
            cands.add(stem + "e")
    for pre in _PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 3:
            cands.add(word[len(pre):])
    best = max(
        (c for c in cands if c != word and known(c)),
        key=lambda c: (zipf_frequency(c, "en"), -len(c), c),
        default=None,
    )
    return best


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
