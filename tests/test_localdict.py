"""localdict: enrichment/sense-picking from the pre-built lexicon dict —
pure logic, no DB needed (build_lexicon's query is the only DB-touching
part, exercised separately via the live pipeline/CLI)."""

from concordance.localdict import enrich, resolve_stub_definition
from concordance.model import Candidate


def _entry(pos, definition, ipa="", etymology=""):
    return (pos, definition, ipa, etymology, False, False)


def test_enrich_fills_fields_from_single_entry():
    lexicon = {"cangue": [_entry("NOUN", "A wooden collar or yoke.", "/kæŋ/")]}
    cand = Candidate(lemma="cangue", pos="NOUN")
    assert enrich(cand, lexicon) is True
    assert cand.part_of_speech == "noun"
    assert cand.definition == "A wooden collar or yoke."
    assert cand.ipa == "/kæŋ/"
    assert cand.definition_source == "Local Wiktionary (DB)"


def test_enrich_takes_first_sense_of_semicolon_joined_definition():
    lexicon = {"tram": [_entry("NOUN", "A streetcar.; A cable car.; A mine cart.")]}
    cand = Candidate(lemma="tram", pos="NOUN")
    enrich(cand, lexicon)
    assert cand.definition == "A streetcar."


def test_enrich_prefers_entry_matching_tagger_pos():
    lexicon = {"tram": [_entry("VERB", "To transport by tram."), _entry("NOUN", "A streetcar.")]}
    cand = Candidate(lemma="tram", pos="NOUN")  # spaCy tagged it a noun here
    enrich(cand, lexicon)
    assert cand.part_of_speech == "noun"
    assert cand.definition == "A streetcar."


def test_enrich_returns_false_on_miss_leaving_candidate_untouched():
    cand = Candidate(lemma="zblargnum", pos="NOUN")
    assert enrich(cand, {}) is False
    assert cand.definition == ""
    assert cand.definition_source == ""


def test_enrich_is_case_insensitive_on_lemma():
    lexicon = {"cobloaf": [_entry("NOUN", "A rounded loaf.")]}
    cand = Candidate(lemma="Cobloaf", pos="NOUN")
    assert enrich(cand, lexicon) is True


# --- "abbreviation of X" stub resolution -----------------------------------
# The dump often glosses a headword as a bare "abbreviation of X." even when
# X isn't really an abbreviation target at all -- a spelling variant
# (vizor/visor), a compound-spacing variant (waterpower/water power), or (in
# the reported bug) a claim that doesn't even hold up under the label
# (waterfit/aquafitness, which share almost no letters). enrich() resolves
# through to X's real content instead of trusting the label.

def test_resolve_stub_definition_implausible_abbreviation_gets_real_gloss():
    # The reported bug: "waterfit" is not an abbreviation of "aquafitness"
    # by any stretch, but aquafitness does have a real definition to surface.
    lookup = {"aquafitness": _entry("NOUN", "water aerobics, typically preceded by stretching")}.get
    result = resolve_stub_definition("waterfit", "abbreviation of aquafitness.", lookup)
    assert result == "Variant spelling of aquafitness — water aerobics, typically preceded by stretching."


def test_resolve_stub_definition_labels_true_truncation_as_abbreviation():
    lookup = {"contemporary": _entry("ADJ", "Existing or occurring at the same time.")}.get
    result = resolve_stub_definition("contemp", "abbreviation of contemporary.", lookup)
    assert result == "Abbreviation of contemporary — Existing or occurring at the same time."


def test_resolve_stub_definition_labels_spelling_variant_as_variant_not_abbreviation():
    lookup = {"visor": _entry("NOUN", "A piece of armor covering the face.")}.get
    result = resolve_stub_definition("vizor", "abbreviation of visor.", lookup)
    assert result == "Variant spelling of visor — A piece of armor covering the face."


def test_resolve_stub_definition_trailing_letter_variant_is_not_labeled_abbreviation():
    # "chield" is only one letter longer than "chiel" -- a spelling variant,
    # not a real clipping, even though the headword is technically a prefix.
    lookup = {"chield": _entry("NOUN", "A man.")}.get
    result = resolve_stub_definition("chiel", "abbreviation of chield.", lookup)
    assert result == "Variant spelling of chield — A man."


def test_resolve_stub_definition_follows_chain_to_real_gloss():
    lookup = {
        "acronycal": _entry("ADJ", "abbreviation of acronical."),
        "acronical": _entry("ADJ", "Occurring at sunset."),
    }.get
    result = resolve_stub_definition("acronychal", "abbreviation of acronycal.", lookup)
    assert result == "Variant spelling of acronycal — Occurring at sunset."


def test_resolve_stub_definition_returns_none_when_target_unresolvable():
    assert resolve_stub_definition("gloop", "abbreviation of glorp.", lambda k: None) is None


def test_resolve_stub_definition_returns_none_on_corrupted_source_markup():
    lookup = {"foo": _entry("NOUN", "Something.")}.get
    assert resolve_stub_definition("bar", "abbreviation of [[w:foo.", lookup) is None


def test_resolve_stub_definition_returns_none_for_non_stub_definition():
    assert resolve_stub_definition("cangue", "A wooden collar or yoke.", lambda k: None) is None


def test_enrich_resolves_stub_definition_through_lexicon():
    lexicon = {
        "waterfit": [_entry("NOUN", "abbreviation of aquafitness.")],
        "aquafitness": [_entry("NOUN", "water aerobics, typically preceded by stretching")],
    }
    cand = Candidate(lemma="waterfit", pos="NOUN")
    enrich(cand, lexicon)
    assert cand.definition == (
        "Variant spelling of aquafitness — water aerobics, typically preceded by stretching."
    )
    assert cand.variant_flag_reason == ""


def test_enrich_flags_unresolvable_stub_for_human_review_instead_of_guessing():
    lexicon = {"gloop": [_entry("NOUN", "abbreviation of glorp.")]}
    cand = Candidate(lemma="gloop", pos="NOUN")
    enrich(cand, lexicon)
    assert cand.definition == "abbreviation of glorp."  # left as-is, not guessed at
    assert cand.variant_flag_reason == "abbreviation_stub"
    assert "glorp" in cand.variant_flag_note


def test_pick_entry_prefers_substantive_sibling_over_stub():
    # Real dump case: "churrigueresque" carries both its real definition and
    # a redundant case-variant stub as separate rows under the same POS.
    lexicon = {
        "churrigueresque": [
            _entry("ADJ", "abbreviation of Churrigueresque."),
            _entry("ADJ", "Relating to a Spanish baroque architectural style."),
        ]
    }
    cand = Candidate(lemma="churrigueresque", pos="ADJ")
    enrich(cand, lexicon)
    assert cand.definition == "Relating to a Spanish baroque architectural style."
