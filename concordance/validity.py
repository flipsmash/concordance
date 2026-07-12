"""Stage 6 — the validity calculus (§05).

Separate axes: *existence* (a string is used in the wild) versus *correctness*
(it is a real, well-formed word). Common misspellings have big footprints, so
existence alone never saves a word. The discriminators, in order:

  1. curated headword?          -> KEEP  (real word, done)
  2. dominant higher-freq twin? -> DROP  (misspelling — even if attested)
  3. attested in the corpus?    -> KEEP  (a rarity with no dominant twin)
  4. recurs in the book?        -> UNSURE (possible coinage/name -> review)
  5. otherwise                  -> DROP  (unattested, unformed -> junk)

Keep-biased throughout: any single solid vouch keeps the word, and step 4 sends
ambiguous-but-recurring tokens to human review rather than dropping them.

This skeleton uses two offline authorities — the SymSpell 82k wordlist (curated
membership + near-neighbor frequencies) and wordfreq (corpus presence). The spec
layers Wiktionary entry-type, WordNet, hunspell, and an LLM adjudicator on top;
those slot into `_authorities` / the misspelling check without changing callers.
"""

from __future__ import annotations

from importlib.resources import files

from wordfreq import zipf_frequency

from .config import Config
from .model import Candidate, RejectReason, Verdict

_CORPUS_PRESENT = 0.0  # wordfreq returns 0.0 for tokens it has never seen


class ValidityGate:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._sym = self._load_symspell()
        self._wn = self._load_wordnet()
        self._words = self._load_word_corpus()

    @staticmethod
    def _load_symspell():
        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
        return sym

    @staticmethod
    def _load_wordnet():
        """WordNet is a much broader curated authority than the 82k wordlist —
        it recognizes archaic/literary words (abash, armiger, cangue) that would
        otherwise be mistaken for misspellings. Degrade gracefully if absent."""
        try:
            from nltk.corpus import wordnet as wn

            wn.synsets("test")  # force the lazy load; raises if data is missing
            return wn
        except Exception:
            try:
                import nltk

                nltk.download("wordnet", quiet=True)
                from nltk.corpus import wordnet as wn

                wn.synsets("test")
                return wn
            except Exception:
                return None

    def _in_wordnet(self, word: str) -> bool:
        return bool(self._wn and self._wn.synsets(word))

    @staticmethod
    def _load_word_corpus() -> frozenset[str]:
        """NLTK's 234k-word dictionary corpus — broad archaic coverage (destrier,
        fiacre, rebec, bartizan) that WordNet misses. A few junk strings leak
        through, but the judge/review catch those; losing real words is worse."""
        try:
            from nltk.corpus import words

            return frozenset(w.lower() for w in words.words())
        except Exception:
            try:
                import nltk

                nltk.download("words", quiet=True)
                from nltk.corpus import words

                return frozenset(w.lower() for w in words.words())
            except Exception:
                return frozenset()

    def _dominant_neighbor(self, word: str) -> str | None:
        """A near-neighbor that is *far* more frequent than `word` marks it a
        misspelling of that neighbor. 'Far' is measured on the Zipf scale, so
        the test is relative, never absolute frequency — which is what protects
        genuine rarities (they have no dominant twin)."""
        from symspellpy import Verbosity

        suggestions = self._sym.lookup(
            word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=False
        )
        for s in suggestions:
            if s.term == word:
                continue
            gap = zipf_frequency(s.term, "en") - zipf_frequency(word, "en")
            if gap >= self.cfg.misspelling_zipf_gap:
                return s.term
            break  # CLOSEST is sorted best-first; if the top isn't dominant, none are
        return None

    def judge(self, cand: Candidate) -> None:
        if cand.verdict is not None:
            return
        word = cand.lemma

        # 1. Curated headword in ANY authority — checked before the misspelling
        #    verdict so a real archaic word (armiger, abash, cangue) is never
        #    branded a typo just for being absent from one small wordlist.
        if word in self._sym.words:
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("wordlist")
            return
        if self._in_wordnet(word):
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("wordnet")
            return
        if word in self._words:
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("dictionary")
            return

        # 2. Misspelling — a dominant higher-frequency near-neighbor, and not a
        #    curated headword above. Author-invented words (alzabo, asimi)
        #    legitimately fall out here or at step 5; per scope, fictitious
        #    coinages need not be captured. EXCEPTION: a real OCR/typo artifact
        #    is almost always a one-off, so a "misspelling" that recurs as often
        #    as a deliberate coinage would (necropoli, 12x in one book) goes to
        #    review instead of a silent auto-drop.
        neighbor = self._dominant_neighbor(word)
        if neighbor:
            if cand.count >= self.cfg.coinage_min_count:
                cand.verdict = Verdict.UNSURE
                cand.interesting_reason = f"recurs but resembles a misspelling of '{neighbor}'"
                return
            cand.verdict = Verdict.DROP
            cand.reject_reason = RejectReason.MISSPELLING
            cand.interesting_reason = f"likely misspelling of '{neighbor}'"
            return

        # 3. Attested in the corpus, no dominant twin -> a rarity worth keeping.
        if cand.zipf > _CORPUS_PRESENT:
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("corpus")
            return

        # 4. Unattested but recurs in the book -> ambiguous; send to review
        #    rather than silently cut.
        if cand.count >= self.cfg.coinage_min_count:
            cand.verdict = Verdict.UNSURE
            return

        # 5. Unattested, unformed, one-off -> junk.
        cand.verdict = Verdict.DROP
        cand.reject_reason = RejectReason.NOT_A_WORD


def apply_validity(candidates: dict[str, Candidate], cfg: Config) -> None:
    gate = ValidityGate(cfg)
    for cand in candidates.values():
        gate.judge(cand)
