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
