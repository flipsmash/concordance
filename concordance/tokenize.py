"""Stage 3 — tokenize, lemmatize, tag, and collapse to distinct candidates.

Every alphabetic token is reduced to its lemma so that inflected forms are
judged as one word (``besmirching`` -> ``besmirch``). Along the way we record,
per lemma, the evidence the proper-noun stage (§04) will need: how often the
tagger called it a proper noun / entity, and how often it was capitalized in
mid-sentence position (where capitalization is a real signal, not just the
start-of-sentence convention).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .extract import Chapter
from .model import Candidate, Occurrence

_MAX_CHUNK = 90_000  # keep each spaCy call well under its default max_length

_word_corpus = None


def _wordset() -> frozenset[str]:
    """NLTK's 234k-word dictionary (lazy, cached) — broad archaic coverage used to
    tell a real lemma from a mangled one. Mirrors validity.py's loader."""
    global _word_corpus
    if _word_corpus is None:
        try:
            from nltk.corpus import words
            _word_corpus = frozenset(w.lower() for w in words.words())
        except Exception:
            try:
                import nltk
                nltk.download("words", quiet=True)
                from nltk.corpus import words
                _word_corpus = frozenset(w.lower() for w in words.words())
            except Exception:
                _word_corpus = frozenset()
    return _word_corpus


def _attested(word: str) -> bool:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en") > 0 or word in _wordset()


# How much rarer (in Zipf points) the lemma is allowed to look than the surface
# form before we distrust the reduction. Real inflections don't trip this: a
# base is normally at least as common as its inflected form (besmirch 2.00 vs
# besmirching 1.68; house 5.71 vs houses 4.72) — comfortably inside the margin.
_LEMMA_RARITY_MARGIN = 0.5


def _resolve_lemma(tok) -> str:
    """spaCy's rule lemmatizer sometimes strips a real word down to a non-word
    (afeared -> afeare, overscutched -> overscutche, windring -> windre). Trust a
    *changed* lemma only when it is itself a real word; otherwise keep the author's
    spelling (which for archaic vocabulary is usually the correct headword).

    Separately, spaCy sometimes mis-tags an uninflected word as a plural noun
    and "reduces" it to a rarer cousin that just happens to also be a real word
    (oftentimes, NNS-tagged -> oftentime, zipf 0.0 vs oftentimes' 3.09;
    necropolis -> necropoli under the same mistake). Don't trust a lemma that
    makes the word look dramatically rarer than the surface form the author
    actually used — a genuine inflection's base never has this problem."""
    from wordfreq import zipf_frequency

    lemma = tok.lemma_.lower().strip()
    surface = tok.text.lower().strip()
    if not lemma:
        return surface
    if lemma == surface:
        return lemma
    if _attested(lemma):
        if zipf_frequency(lemma, "en") >= zipf_frequency(surface, "en") - _LEMMA_RARITY_MARGIN:
            return lemma
        return surface if _attested(surface) else lemma
    return surface if _attested(surface) else lemma


@dataclass
class _Acc:
    """Mutable per-lemma accumulator; frozen into a Candidate at the end."""

    pos_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    occurrences: list[Occurrence] = field(default_factory=list)
    propn_hits: int = 0
    ent_hits: int = 0
    midsentence_total: int = 0
    midsentence_caps: int = 0

    @property
    def total(self) -> int:
        return sum(self.pos_counts.values())


def _load_nlp():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is missing. Install it with:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


def _chunks(text: str) -> list[str]:
    if len(text) <= _MAX_CHUNK:
        return [text]
    out, buf = [], []
    size = 0
    for para in text.split("\n\n"):
        if size + len(para) > _MAX_CHUNK and buf:
            out.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out


def tokenize(chapters: list[Chapter]) -> dict[str, Candidate]:
    nlp = _load_nlp()
    acc: dict[str, _Acc] = defaultdict(_Acc)

    for chapter in chapters:
        for chunk in _chunks(chapter.text):
            doc = nlp(chunk)
            for sent in doc.sents:
                sent_text = sent.text.strip().replace("\n", " ")
                sent_start = sent.start
                for tok in sent:
                    if not tok.is_alpha or tok.is_stop or len(tok) < 3:
                        continue
                    lemma = _resolve_lemma(tok)
                    if not lemma:
                        continue
                    a = acc[lemma]
                    a.pos_counts[tok.pos_] += 1
                    if tok.pos_ == "PROPN":
                        a.propn_hits += 1
                    if tok.ent_type_:
                        a.ent_hits += 1
                    # Capitalization only carries signal away from sentence start.
                    if tok.i != sent_start:
                        a.midsentence_total += 1
                        if tok.text[:1].isupper():
                            a.midsentence_caps += 1
                    if len(a.occurrences) < 12:  # keep a bounded sample per lemma
                        a.occurrences.append(
                            Occurrence(sentence=sent_text, chapter=chapter.title, surface=tok.text)
                        )

    candidates: dict[str, Candidate] = {}
    for lemma, a in acc.items():
        dominant_pos = max(a.pos_counts, key=a.pos_counts.get)
        candidates[lemma] = Candidate(
            lemma=lemma,
            pos=dominant_pos,
            occurrences=a.occurrences,
            propn_ratio=a.propn_hits / a.total if a.total else 0.0,
            ent_ratio=a.ent_hits / a.total if a.total else 0.0,
            cap_ratio=(a.midsentence_caps / a.midsentence_total) if a.midsentence_total else 0.0,
        )
    return candidates
