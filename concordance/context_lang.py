"""Is a book sentence modern English? (§ validity -- context language)

Used to find words whose every use in their source books sits inside Old
English, Middle English or foreign-language text -- a quotation, an edition
of a medieval poem, a French epigraph -- rather than in the book's own
modern-English prose.

Two signals, because neither alone is enough:
  * fastText's lid.176 language identifier (downloaded once, <1 MB) catches
    foreign languages and Old English, but calls Middle English "en" at
    ~0.97;
  * Middle English markers: words Wiktionary has ONLY as Middle/Old English
    (no modern English, Scots or Translingual entry -- wikt.historic_term,
    built by db.load_wiktionary_langs), plus the share of words unknown to
    modern English. Scots dialogue (sleekit, cowrin, breastie) has unknown
    words but no Middle English markers, so it stays English.
The target word itself is excluded from its sentence's counts.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_MIN_TOKENS = 5              # shorter sentences are "unknown", never evidence
_FOREIGN_MIN_PROB = 0.5
_ME_MIN_MARKERS, _ME_MIN_SHARE = 2, 0.15
_UNKNOWN_MIN, _UNKNOWN_MIN_SHARE = 3, 0.30
_KNOWN_EN_ZIPF = 2.5


def lid_model_path() -> Path:
    base = Path(os.environ.get("CONCORDANCE_CACHE_DIR", Path.home() / ".cache" / "concordance"))
    path = base / "lid.176.ftz"
    if not path.exists():
        base.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-s", "--fail", "-o", str(path), LID_URL], check=True)
    return path


class ContextClassifier:
    """classify(sentence, target_forms) -> (kind, detail), kind one of
    'english', 'foreign', 'middle_english', 'unknown'."""

    def __init__(self, historic_terms: set[str]):
        import fasttext
        fasttext.FastText.eprint = lambda *a, **k: None      # silence its load warning
        from wordfreq import zipf_frequency
        from .validity_score import _in_wordnet, _wordset
        self._ft = fasttext.load_model(str(lid_model_path()))
        self._historic = historic_terms
        self._zipf, self._wordnet, self._words = zipf_frequency, _in_wordnet, _wordset()
        self._known: dict[str, bool] = {}

    def _is_known(self, t: str) -> bool:
        if t not in self._known:
            self._known[t] = (self._zipf(t, "en") >= _KNOWN_EN_ZIPF or t in self._words
                              or self._wordnet(t))
        return self._known[t]

    def classify(self, sentence: str, target_forms: set[str]) -> tuple[str, str | None]:
        other = [t for t in _TOKEN_RE.findall(sentence) if t.lower() not in target_forms]
        if len(other) < _MIN_TOKENS:
            return "unknown", None
        (label,), (prob,) = self._ft.predict(" ".join(sentence.split()), k=1)
        lang = label.removeprefix("__label__")
        if lang != "en" and prob >= _FOREIGN_MIN_PROB:
            return "foreign", lang
        low = [t.lower() for t in other]
        markers = [t for t in low if t in self._historic]
        if len(markers) >= _ME_MIN_MARKERS and len(markers) / len(low) >= _ME_MIN_SHARE:
            return "middle_english", ",".join(markers[:4])
        lower_only = [t.lower() for t in other if not t[0].isupper()]
        unknown = [t for t in lower_only if not self._is_known(t)]
        if (markers and len(unknown) >= _UNKNOWN_MIN
                and len(unknown) / max(1, len(lower_only)) >= _UNKNOWN_MIN_SHARE):
            return "middle_english", ",".join(unknown[:4])
        return ("unknown", lang) if lang != "en" else ("english", None)


def target_forms(lemma: str, as_seen: str | None = None) -> set[str]:
    """The surface forms to look for in a book: lemma, the form actually
    seen at ingest, and regular inflections."""
    forms = {lemma.lower()} | ({as_seen.lower()} if as_seen else set())
    if lemma.isalpha():
        forms |= {lemma + "s", lemma + "es", lemma + "ed", lemma + "d", lemma + "ing"}
        if lemma.endswith("y"):
            forms.add(lemma[:-1] + "ies")
    return forms


_BOUND = re.compile(r"[.!?][\"'”’)\]]*\s|\n\s*\n")


def sentence_at(text: str, start: int, end: int, reach: int = 500) -> str:
    lo, hi = max(0, start - reach), min(len(text), end + reach)
    before = [m.end() for m in _BOUND.finditer(text, lo, start)]
    s = before[-1] if before else lo
    m = _BOUND.search(text, end, hi)
    return " ".join(text[s:(m.end() if m else hi)].split())


def occurrences(text: str, words: dict[int, set[str]], per_word: int = 30) -> dict[int, list[str]]:
    """word id -> up to `per_word` sentences using any of its forms in `text`.
    Case-insensitive; forms are matched via casefold() so an old long-s
    spelling (ſ) that the regex folds to 's' still maps back to its word."""
    by_form = {f.casefold(): wid for wid, forms in words.items() for f in forms}
    if not by_form:
        return {}
    rx = re.compile(r"\b(" + "|".join(sorted(map(re.escape, by_form), key=len, reverse=True)) + r")\b",
                    re.IGNORECASE)
    out: dict[int, list[str]] = {}
    for m in rx.finditer(text):
        wid = by_form.get(m.group(1).casefold())
        if wid is None or len(out.setdefault(wid, [])) >= per_word:
            continue
        out[wid].append(sentence_at(text, m.start(), m.end()))
    return out
