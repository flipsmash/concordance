"""Regression tests for the deterministic filters — the parts most likely to
silently over- or under-drop. No spaCy or model needed."""

from wordfreq import zipf_frequency

from concordance.config import Config
from concordance.floor import apply_floor
from concordance.model import Candidate, Occurrence, RejectReason, Verdict
from concordance.propernouns import strip_proper_nouns
from concordance.validity import ValidityGate


def _cand(lemma, *, zipf=None, count=1, propn=0.0, ent=0.0, cap=0.0):
    occ = [Occurrence(sentence=f"a sentence with {lemma} in it", chapter="c", surface=lemma)
           for _ in range(count)]
    c = Candidate(lemma=lemma, pos="NOUN", occurrences=occ,
                  propn_ratio=propn, ent_ratio=ent, cap_ratio=cap)
    c.zipf = zipf if zipf is not None else zipf_frequency(lemma, "en")
    return c


# --- validity gate (§05) -------------------------------------------------

def test_common_headword_kept():
    c = _cand("house")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP and "wordlist" in c.validity_sources


def test_misspelling_dropped_even_though_attested():
    # 'definately' has a real web footprint, but a dominant near-twin.
    c = _cand("definately")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.MISSPELLING


def test_attested_rarity_kept():
    # Not in the 82k wordlist, no dominant twin, but present in the corpus.
    c = _cand("apophenia", zipf=1.6)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP and "corpus" in c.validity_sources


def test_archaic_real_word_not_branded_a_misspelling():
    # Regression: real archaic words absent from the 82k wordlist but present in
    # WordNet / the dictionary corpus must not be dropped as misspellings, even
    # when a common near-neighbor exists (destrier ~ 'destroyer').
    gate = ValidityGate(Config())
    for word in ("destrier", "bartizan", "armiger", "rebec"):
        c = _cand(word)
        gate.judge(c)
        assert c.verdict is Verdict.KEEP, f"{word} was wrongly dropped"


def test_recurring_unattested_is_unsure_not_dropped():
    c = _cand("zblargnum", zipf=0.0, count=4)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.UNSURE  # possible coinage -> review, never silent drop


def test_oneoff_gibberish_dropped():
    c = _cand("xqzptv", zipf=0.0, count=1)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP and c.reject_reason is RejectReason.NOT_A_WORD


def test_foreign_context_sentence_drops_oneoff_fragment():
    # Regression: a real Ulysses run kept "cuius"/"verbo" as words because
    # wordfreq gives them some nonzero corpus zipf even though they're just
    # fragments of a quoted Latin liturgical sentence — the validity gate
    # never looked at the SENTENCE they came from until now.
    latin_sentence = ("Deus cuius verbo sanctificantur omnia benedictionem "
                       "tuam effunde super creaturas istas")
    c = Candidate(lemma="cuius", pos="NOUN",
                  occurrences=[Occurrence(sentence=latin_sentence, chapter="c", surface="cuius")])
    c.zipf = zipf_frequency("cuius", "en")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.FOREIGN_LANGUAGE


def test_foreign_context_sentence_recurring_is_unsure_not_dropped():
    # Keep-biased: if the same "foreign-context" word recurs as often as a
    # deliberate coinage would, send it to review rather than silently drop —
    # same leniency already given to the misspelling and unattested paths.
    latin_sentence = "Deus cuius verbo sanctificantur omnia benedictionem tuam"
    c = Candidate(lemma="cuius", pos="NOUN",
                  occurrences=[Occurrence(sentence=latin_sentence, chapter="c", surface="cuius")
                               for _ in range(5)])
    c.zipf = zipf_frequency("cuius", "en")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.UNSURE


def test_english_sentence_context_unaffected():
    # A genuine rare English word in a normal English sentence must not trip
    # the foreign-context check just because it has enough token count.
    c = _cand("apophenia", zipf=1.6, count=1)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP


def test_wordnet_instance_only_word_dropped_when_nothing_else_vouches():
    # 'abdias' -- WordNet's only synset for it is an instance (a specific
    # named biblical figure), and it's absent from both the 82k wordlist
    # and the NLTK words corpus, so nothing else offers a competing
    # common-noun vouch. This is the clean case: safe to drop outright.
    c = _cand("abdias", zipf=0.0, count=1)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.PROPER_NOUN


def test_wordnet_instance_only_collision_flagged_not_dropped():
    # 'godel' -- WordNet only catalogues it as a specific named entity (the
    # mathematician), but it's also in the 82k wordlist (a real collision:
    # some other authority DOES think it's a common word). Real ambiguity
    # this signal alone can't resolve -- keep-biased means flag for human
    # review, not auto-drop, same as the foreign/misspelling-variant flag.
    c = _cand("godel", count=1)
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP
    assert c.variant_flag_reason == RejectReason.PROPER_NOUN.value
    assert "godel" in c.variant_flag_note


def test_gazetteer_word_dropped_when_nothing_else_vouches():
    # Isolates the gazetteer path from WordNet's own instance-only signal --
    # a nonsense string has zero WordNet synsets (not instance-only ones)
    # and no dictionary presence, so only the gazetteer membership check
    # can be responsible for a DROP here.
    gate = ValidityGate(Config(), gazetteer_names=frozenset({"zzqwortk"}))
    c = _cand("zzqwortk", zipf=0.0, count=1)
    gate.judge(c)
    assert c.verdict is Verdict.DROP
    assert c.reject_reason is RejectReason.PROPER_NOUN


def test_gazetteer_collision_flagged_not_dropped():
    # 'house' is unambiguously a real common word (SymSpell/NLTK words
    # corpus both vouch) -- forcing it into a fake gazetteer set is a
    # synthetic way to isolate the collision-handling branch: a name-list
    # hit must never auto-drop a word something else independently vouches
    # for, regardless of source.
    gate = ValidityGate(Config(), gazetteer_names=frozenset({"house"}))
    c = _cand("house")
    gate.judge(c)
    assert c.verdict is Verdict.KEEP
    assert c.variant_flag_reason == RejectReason.PROPER_NOUN.value


def test_gazetteer_absent_is_a_silent_no_op():
    # No gazetteer_names passed at all (the default) -- must behave exactly
    # like before the gazetteer existed, not raise or drop anything.
    c = _cand("house")
    ValidityGate(Config()).judge(c)
    assert c.verdict is Verdict.KEEP
    assert c.variant_flag_reason == ""


# --- frequency floor (§03.4) --------------------------------------------

def test_floor_drops_common_word():
    cands = {"the": _cand("the")}
    apply_floor(cands, Config())
    assert cands["the"].verdict is Verdict.DROP
    assert cands["the"].reject_reason is RejectReason.FREQUENCY_FLOOR


def test_floor_spares_rare_word():
    cands = {"lugubrious": _cand("lugubrious")}
    apply_floor(cands, Config())
    assert cands["lugubrious"].verdict is None  # survives the floor


# --- proper-noun stripping (§04) ----------------------------------------

def test_oneoff_sentence_initial_propn_is_not_dropped():
    # The 'mote' bug: single PROPN tag on a lone sentence-initial token.
    cands = {"mote": _cand("mote", propn=1.0, count=1, cap=0.0)}
    strip_proper_nouns(cands, Config())
    assert cands["mote"].verdict is None


def test_recurring_propn_is_dropped():
    cands = {"aragorn": _cand("aragorn", propn=1.0, count=3, cap=0.0)}
    strip_proper_nouns(cands, Config())
    assert cands["aragorn"].verdict is Verdict.DROP
    assert cands["aragorn"].reject_reason is RejectReason.PROPER_NOUN


def test_tagger_mistag_on_common_word_is_not_dropped():
    # Regression: a real Ulysses run had spaCy's tagger call "tram"/"beggar"/
    # "alderman"/"sacrament" proper nouns purely on statistical mistagging —
    # none of them were EVER capitalized in the actual text. The tagger alone
    # shouldn't be trusted for a well-established dictionary word.
    for word in ("tram", "beggar", "alderman", "sacrament"):
        cands = {word: _cand(word, propn=0.9, ent=0.2, count=12, cap=0.0)}
        strip_proper_nouns(cands, Config())
        assert cands[word].verdict is None, f"{word} was wrongly dropped as a proper noun"


def test_dictionary_word_still_dropped_when_capitalization_is_high():
    # The Bloom case: a common dictionary word ("bloom") that's ALSO a
    # character's name must still be caught — by the capitalization ratio,
    # which the dictionary-word exemption above deliberately doesn't touch.
    cands = {"bloom": _cand("bloom", propn=0.94, ent=0.24, count=12, cap=0.98)}
    strip_proper_nouns(cands, Config())
    assert cands["bloom"].verdict is Verdict.DROP
    assert cands["bloom"].reject_reason is RejectReason.PROPER_NOUN


def test_high_capitalization_ratio_is_dropped():
    cands = {"baker": _cand("baker", propn=0.0, count=5, cap=0.9)}
    strip_proper_nouns(cands, Config())
    assert cands["baker"].verdict is Verdict.DROP
