"""Stage 6 — the validity calculus (§05).

Separate axes: *existence* (a string is used in the wild) versus *correctness*
(it is a real, well-formed word). Common misspellings have big footprints, so
existence alone never saves a word. The discriminators, in order:

  0. in the local Wiktionary dump? -> KEEP  (curated, no "Proper noun" POS
                                              at all in this dump — see below)
  1. foreign-language context?     -> DROP/UNSURE (quoted non-English phrase)
  2. curated headword?             -> KEEP  (real word, done)
  3. dominant higher-freq twin?    -> DROP  (misspelling — even if attested)
  4. attested in the corpus?       -> KEEP  (a rarity with no dominant twin)
  5. recurs in the book?           -> UNSURE (possible coinage/name -> review)
  6. otherwise                     -> DROP  (unattested, unformed -> junk)

Keep-biased throughout: any single solid vouch keeps the word, and step 5 sends
ambiguous-but-recurring tokens to human review rather than dropping them.

Step 0 (vocab.wiktionary, ~500k terms already loaded in Postgres) is checked
before everything else: it's cheap (no network, no per-word cost beyond one
bulk query), and — unlike every authority in step 2, which are all frequency-
derived from general web text and so get polluted by real proper nouns that
happen to have some web footprint (verified: ahasuerus/oisin/fecit are all in
the 82k wordlist) — this dump was built with no "Proper noun" POS category at
all, so membership alone is a clean, structural "this is not a name" signal.
"""

from __future__ import annotations

from importlib.resources import files

from wordfreq import zipf_frequency

from .config import Config
from .model import Candidate, RejectReason, Verdict
from .validity_score import _english_fraction, _FOREIGN_ENGLISH_FRACTION, _FOREIGN_MIN_TOKENS

_CORPUS_PRESENT = 0.0  # wordfreq returns 0.0 for tokens it has never seen


class ValidityGate:
    def __init__(self, cfg: Config, local_dict: dict | None = None):
        self.cfg = cfg
        self.local_dict = local_dict or {}
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
        """True only if WordNet has a GENERIC sense. A word whose ONLY
        synsets are "instances" (Rahab, Ahasuerus, Coventry: specific named
        individuals, not common-noun categories) shouldn't get to vouch for
        itself as real vocabulary — WordNet catalogues biblical/historical/
        mythological names as instance entries same as any city or person,
        which otherwise lets a genuine one-off proper noun sail through."""
        if not self._wn:
            return False
        synsets = self._wn.synsets(word)
        return any(not s.instance_hypernyms() for s in synsets)

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

        # 0. Local Wiktionary dump — cheap, curated, and structurally free of
        #    proper nouns (see module docstring). Checked before even the
        #    foreign-context step: a curated confirmation that this is a real,
        #    documented English word outweighs a crude sentence-language guess.
        if word in self.local_dict:
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("wiktionary-local")
            return

        # 1. A foreign-language context sentence is decisive on its own — whatever
        #    dictionary/corpus attestation the token has, if it sits among mostly
        #    non-English words (a quoted Latin/French/Italian phrase) it's a
        #    fragment of that quote, not a word the reader would look up. This is
        #    what actually defeats step 2's keep-bias for one-off Latin tokens
        #    (cuius, verbo, fecit) that have SOME wordfreq corpus presence.
        rep = cand.representative
        if rep:
            frac, ntok = _english_fraction(rep.sentence)
            if ntok >= _FOREIGN_MIN_TOKENS and frac < _FOREIGN_ENGLISH_FRACTION:
                if cand.count >= self.cfg.coinage_min_count:
                    cand.verdict = Verdict.UNSURE
                    cand.interesting_reason = "recurs but sits in a non-English context"
                else:
                    cand.verdict = Verdict.DROP
                    cand.reject_reason = RejectReason.FOREIGN_LANGUAGE
                return

        # 2. Curated headword in ANY authority — checked before the misspelling
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

        # 3. Misspelling — a dominant higher-frequency near-neighbor, and not a
        #    curated headword above. Author-invented words (alzabo, asimi)
        #    legitimately fall out here or at step 6; per scope, fictitious
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

        # 4. Attested in the corpus, no dominant twin -> a rarity worth keeping.
        if cand.zipf > _CORPUS_PRESENT:
            cand.verdict = Verdict.KEEP
            cand.validity_sources.append("corpus")
            return

        # 5. Unattested but recurs in the book -> ambiguous; send to review
        #    rather than silently cut.
        if cand.count >= self.cfg.coinage_min_count:
            cand.verdict = Verdict.UNSURE
            return

        # 6. Unattested, unformed, one-off -> junk.
        cand.verdict = Verdict.DROP
        cand.reject_reason = RejectReason.NOT_A_WORD


def apply_validity(candidates: dict[str, Candidate], cfg: Config, local_dict: dict | None = None,
                   gate: "ValidityGate | None" = None) -> None:
    """Run the validity gate over every candidate. A batch run can build one
    gate (which loads SymSpell + WordNet + the 234k-word corpus once) and pass
    it in for every book; only the per-book `local_dict` (that book's slice of
    vocab.wiktionary) changes, so it's swapped in per call."""
    if gate is None:
        gate = ValidityGate(cfg, local_dict=local_dict)
    else:
        gate.local_dict = local_dict or {}
    for cand in candidates.values():
        gate.judge(cand)
