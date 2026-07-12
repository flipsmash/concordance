"""localdict: enrichment/sense-picking from the pre-built lexicon dict —
pure logic, no DB needed (build_lexicon's query is the only DB-touching
part, exercised separately via the live pipeline/CLI)."""

from concordance.localdict import enrich
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
