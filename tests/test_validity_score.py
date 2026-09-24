"""Validity likelihood scorer — signals stubbed to test the scoring logic."""

from __future__ import annotations

import pytest

from concordance import validity_score as V


def _stub(monkeypatch, *, ng, corpus, neighbor=None, root=None, zipf=0.0):
    monkeypatch.setattr(V, "_ngram_peak", lambda w, s: ng)
    monkeypatch.setattr(V, "_wordset", lambda: {"cobloaf"} if corpus else set())
    monkeypatch.setattr(V, "_in_wordnet", lambda w: False)
    monkeypatch.setattr(V, "_dominant_neighbor", lambda w: neighbor)
    monkeypatch.setattr(V, "_morph_root", lambda w: root)
    monkeypatch.setattr(V, "zipf_frequency", lambda w, lang: zipf)


def test_real_archaic_in_wordlist_is_valid(monkeypatch):
    _stub(monkeypatch, ng=1e-8, corpus=True)
    e = V.estimate("cobloaf")
    assert e.label == "likely-valid" and e.score >= 0.6


def test_nonsense_absent_from_books_is_artifact(monkeypatch):
    _stub(monkeypatch, ng=0.0, corpus=False)
    e = V.estimate("zxqwplt")
    assert e.label == "likely-artifact" and e.score <= 0.35
    assert "absent from Google Books" in e.notes


def test_dominant_neighbor_flags_ocr_variant(monkeypatch):
    _stub(monkeypatch, ng=1e-7, corpus=False, neighbor=("daisies", 2.9), zipf=1.1)
    e = V.estimate("daisie")
    assert e.suggestion == "daisies"
    assert e.label in ("likely-artifact", "uncertain")
    assert e.score < 0.6


def test_books_only_is_uncertain_not_valid(monkeypatch):
    # appears in old books (Latin/old-spelling) but no wordlist, no neighbor
    _stub(monkeypatch, ng=1e-6, corpus=False, zipf=1.2)
    e = V.estimate("fatuus")
    assert e.label == "uncertain"


# --- real (offline) helper behaviour --------------------------------------

def test_morph_root_peels_affixes():
    assert V._morph_root("recoloring") in ("color", "colore", "recolor", "coloring", "recoloring") or \
           V._morph_root("recoloring") is not None


# --- the "un-" floor bug: a transparent prefixed/suffixed form of a common
# word (unbuttoned, bemused) must not float across the frequency floor purely
# because wordfreq undercounts the derived form relative to its root --------

def test_morph_root_finds_a_single_peel_root():
    # unbuttoned resolves to 'buttoned' (single suffix peel), not all the way
    # to 'button' -- see test_morph_root_does_not_chain_a_second_peel for why
    # a second peel is deliberately not attempted.
    assert V._morph_root("unbuttoned") == "buttoned"
    assert V.effective_zipf("unbuttoned") >= 2.5


def test_morph_root_restores_silent_e():
    # Naively slicing '-ing'/'-ed' off a word whose root drops a silent 'e'
    # produces a truncated non-word (hoping -> 'hop', mused -> 'mus') that
    # can coincidentally BE a real, unrelated word -- the root must restore
    # the 'e' and land on the true root instead.
    assert V._morph_root("hoping") == "hope"
    assert V._morph_root("mused") == "muse"


def test_morph_root_does_not_chain_a_second_peel():
    # A second, chained peel is what let a coincidental letter-match
    # manufacture an unrelated real word:
    #   reseed -[peel 'ed']-> resee -[peel 're']-> 'see'            (wrong)
    #   uncomely -[peel 'un']-> comely -[peel 'ly']-> 'come'        (wrong)
    #   impaled -[peel 'd']-> impale -[peel 'im']-> 'pale'          (wrong)
    #   bemused -[peel 'be']-> mused -[peel 'ed']-> 'mus'           (wrong)
    # Each of these was reachable when a second peel was attempted onto an
    # already-peeled intermediate; single-peel-only makes all four
    # unreachable, and 'pale'/'mus' in particular used to inflate a genuine
    # rarity's effective_zipf enough to cross the frequency floor and drop
    # it before the validity gate or judge ever saw it.
    assert V._morph_root("reseed") != "see"
    assert V._morph_root("uncomely") != "come"
    assert V._morph_root("impaled") != "pale"
    assert V.effective_zipf("impaled") < 3.5          # must not cross the floor
    assert V._morph_root("bemused") != "mus"
    assert V.effective_zipf("bemused") < 3.5           # must not cross the floor


def test_morph_root_leaves_genuine_rarities_alone():
    for w in ("cangue", "bartizan", "fuligin", "abacination"):
        assert V._morph_root(w) is None


def test_dominant_neighbor_ignores_proper_names():
    # 'tarrie' has name neighbours (carrie/barrie) — those must not count unless
    # they are real WordNet words; a genuine common neighbour may still register.
    n = V._dominant_neighbor("tarrie")
    if n:
        assert V._in_wordnet(n[0]) or n[1] >= 4.0


# --- foreign-language context rule ----------------------------------------

def _stub_real_zipf(monkeypatch, *, ng=1e-7, corpus=True, root=None):
    monkeypatch.setattr(V, "_ngram_peak", lambda w, s: ng)
    monkeypatch.setattr(V, "_wordset", lambda: {"purus", "regia"} if corpus else set())
    monkeypatch.setattr(V, "_in_wordnet", lambda w: False)
    monkeypatch.setattr(V, "_dominant_neighbor", lambda w: None)
    monkeypatch.setattr(V, "_morph_root", lambda w: root)
    # NB: real zipf_frequency left in place so _english_fraction works.


def test_foreign_context_caps_to_artifact(monkeypatch):
    _stub_real_zipf(monkeypatch)
    e = V.estimate("purus", sentence="Integer vitae, scelerisque purus, Non eget Mauri iaculis, nec arcu")
    assert e.label == "likely-artifact"
    assert "foreign-language context" in e.notes


def test_english_context_not_flagged(monkeypatch):
    _stub_real_zipf(monkeypatch)
    e = V.estimate("regardance", sentence="Since you to non-regardance cast my faith and that I have adjudged")
    assert "foreign-language context" not in e.notes
    assert e.label != "likely-artifact"        # not penalised for language


def test_short_sentence_not_flagged(monkeypatch):
    # too few tokens to judge the language — must not trip the rule
    _stub_real_zipf(monkeypatch)
    e = V.estimate("purus", sentence="purus est")
    assert "foreign-language context" not in e.notes


def test_english_fraction_helper():
    hi, n = V._english_fraction("the quick brown fox jumps over the lazy dog again")
    assert hi > 0.5 and n >= 5
    lo, _ = V._english_fraction("Hic ibat Simois hic est Sigeia tellus hic steterat Priami regia")
    assert lo < 0.5


# --- foreign_language_hint / unambiguous_dominant_neighbor / variant_reject_reason ---
# Real (unstubbed) checks against wordfreq/symspellpy/wordnet -- these are the
# actual detectors used to gate whether a definition source's hit gets
# accepted, verified against real words pulled from this project's own
# corpus during the web-search-tier rollout, not synthetic examples.

@pytest.mark.parametrize("word,lang", [
    ("acte", "fr"),
    ("bellissimo", "it"),
    ("auxilio", "es"),
    ("jolie", "fr"),
    ("unter", "de"),
    ("jadis", "fr"),
])
def test_foreign_language_hint_catches_real_foreign_words(word, lang):
    hint = V.foreign_language_hint(word)
    assert hint is not None and hint[0] == lang


@pytest.mark.parametrize("word", [
    "armiger", "cangue", "bogoak", "aftersong", "homometrically", "stele", "silkman",
])
def test_foreign_language_hint_does_not_flag_real_rare_english_words(word):
    assert V.foreign_language_hint(word) is None


@pytest.mark.parametrize("word,neighbor", [
    ("assunder", "asunder"),
    ("beneficiall", "beneficial"),
    ("apparrell", "apparel"),
    ("allyance", "alliance"),
    ("adventrous", "adventurous"),
])
def test_unambiguous_dominant_neighbor_catches_archaic_spelling_variants(word, neighbor):
    assert V.unambiguous_dominant_neighbor(word) == neighbor


@pytest.mark.parametrize("word", [
    "bogoak",     # ties with bogota/bogor/boga/bogon -- genuinely ambiguous, not "book"
    "armiger",    # a real archaic word this project explicitly wants to keep
    "cangue",     # ditto -- ties with unrelated "gangue"
    "befalne",    # ties with beaune/betaine as well as befall/befallen -- ambiguous
    "aftersong",  # no SymSpell candidates at all
])
def test_unambiguous_dominant_neighbor_does_not_flag_real_or_ambiguous_words(word):
    assert V.unambiguous_dominant_neighbor(word) is None


def test_variant_reject_reason_foreign_wins_over_misspelling():
    from concordance.model import RejectReason
    reason, note = V.variant_reject_reason("acte")
    assert reason is RejectReason.FOREIGN_LANGUAGE
    assert "fr" in note


def test_variant_reject_reason_flags_archaic_spelling():
    from concordance.model import RejectReason
    reason, note = V.variant_reject_reason("assunder")
    assert reason is RejectReason.MISSPELLING
    assert "asunder" in note


def test_variant_reject_reason_none_for_a_real_word():
    assert V.variant_reject_reason("armiger") is None


# --- spelling-alone rejects (design rule 3) ---------------------------------

def test_fold_spelling():
    from concordance.validity_score import fold_spelling
    assert fold_spelling("fellòw") == "fellow"
    assert fold_spelling("hæmorrhage") == "haemorrhage"
    assert fold_spelling("cortège") == "cortege"
    assert fold_spelling("mustnʼt") == "mustn't"
    assert fold_spelling("ceraunomancy") == "ceraunomancy"


def test_script_reject_reason_narrow_classes():
    from concordance.validity_score import script_reject_reason as r
    assert r("χαλκὸς", 3.5)[0] == "script_foreign"
    assert r("þusent", 3.5)[0] == "script_archaic_letter"
    assert r("monèy", 3.5)[0] == "script_common_variant"      # accented common word
    # real rare words in a variant spelling are NOT rejected -- review, not drop
    for w in ("mélange", "uræus", "crispèd", "cortège", "ceraunomancy", "mustnʼt"):
        assert r(w, 3.5) is None, w


def test_validity_gate_drops_script_junk_before_local_dict():
    from concordance.config import Config
    from concordance.model import Candidate, RejectReason, Verdict
    from concordance.validity import ValidityGate
    gate = ValidityGate(Config(), local_dict={"þusent", "χαλκὸς"})   # would otherwise KEEP
    for word, reason in (("þusent", RejectReason.MISSPELLING), ("χαλκὸς", RejectReason.FOREIGN_LANGUAGE)):
        c = Candidate(lemma=word, pos="NOUN")
        gate.judge(c)
        assert c.verdict is Verdict.DROP and c.reject_reason is reason


def test_dialect_respelling_target():
    from concordance.validity_score import dialect_respelling_target as t
    assert t("Pronunciation spelling of better.") == "better"
    assert t("Eye dialect spelling of talk, representing New York City English.") == "talk"
    assert t("An obsolete or dialectal form of bayonet.") == "bayonet"
    assert t("(now Appalachia) Pronunciation spelling of coil. [To wind...]") == "coil"
    assert t("Pronunciation spelling of by and by.") == "by and by"
    # real glosses that merely mention a form/spelling are not respellings
    assert t("An informal form of address; see guv.") is None                 # guvnor
    assert t("(botany) Nut-shaped; (humorous) Pronunciation spelling of nuclear.") is None
    assert t("UK standard spelling of colorization.") is None                 # regional standard, not dialect
    assert t("A protein that inhibits caspase activity") is None              # survivin
    assert t(None) is None


def test_archaic_inflection_target():
    from concordance.validity_score import archaic_inflection_target as a
    assert a("thinketh", "third-person singular simple present indicative of think") == "think"
    assert a("risest", "second-person singular simple present indicative of rise") == "rise"
    assert a("brakest", "second-person singular simple past indicative of break") == "break"
    # shape without the inflection gloss is not enough -- real words / superlatives
    assert a("hest", "Command, injunction.") is None
    assert a("stickiest", "Able or likely to stick.") is None
    # gloss without the archaic ending is not this rule's business
    assert a("thinks", "third-person singular simple present indicative of think") is None


def test_early_modern_uv_target():
    from concordance.validity_score import early_modern_uv_target as u
    assert u("reuelation", 3.5) == "revelation"
    assert u("nerue", 3.5) == "nerve"
    for real in ("moue", "bovver", "survivin"):       # wordfreq knows these
        assert u(real, 3.5) is None, real


def test_obsolete_spelling_target():
    from concordance.validity_score import obsolete_spelling_target as o
    assert o("Obsolete spelling of bread.") == "bread"
    assert o("An obsolete form of spill.") == "spill"
    assert o("Archaic spelling of boulder.") == "boulder"
    assert o("Obsolete form of embassage (“message, embassy”).") == "embassage"
    assert o("Obsolete form of good nature.") == "good nature"
    # a real sense first, or a non-spelling gloss, is not a pointer
    assert o("(obsolete) To attend to; to apply oneself to.; Obsolete form of intend.") is None
    assert o("A grotesque creature of folklore.") is None


def test_foreign_cast_out_reason_requires_every_english_check_to_fail():
    from concordance.validity_score import foreign_cast_out_reason as f
    note = f("miteinander", ["German"], "Web (LLM-extracted)")
    assert note and "German" in note
    assert f("miteinander", None, "Web (LLM-extracted)") is None              # no foreign-only evidence
    assert f("miteinander", ["German"], "Merriam-Webster API") is None          # English dictionary defined it
    assert f("miteinander", ["German"], "Local Wiktionary (DB) (synonym of 'x')") is None
    assert f("haft", ["Danish"], "") is None                                   # common in English
    assert f("glaive", ["French"], "") is None                                 # Webster list / WordNet


def test_english_evidence():
    from concordance.validity_score import english_evidence as e
    assert e("zzqx", "", True)                                 # English Wiktionary / 0 Dict entry
    assert e("zzqx", "Merriam-Webster API", False)             # English dictionary definition
    assert e("haft", "", False)                                # wordfreq / wordlist
    assert e("zzqx", "datamuse", False) is None
    assert e("zzqx", "Web (LLM-extracted)", False) is None


def test_english_evidence_strict_mode_for_misspellings():
    from concordance.validity_score import english_evidence as e
    # common typos have web footprints: wordfreq alone doesn't clear a misspelling flag
    assert e("recieve", "", False, use_wordfreq=True)
    assert e("recieve", "", False, use_wordfreq=False) is None
    # a dictionary that only glosses it as a misspelling doesn't vouch for it
    assert e("zzqx", "Wiktionary", False, use_wordfreq=False, definition="Misspelling of receive.") is None
    assert e("zzqx", "Wiktionary", False, use_wordfreq=False, definition="A small songbird.")
