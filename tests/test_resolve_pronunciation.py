"""Unified pronunciation-acquisition cascade (concordance/resolve_pronunciation.py).
Pure logic over pre-built per-word lexicon slices -- no DB or network."""

from __future__ import annotations

from concordance import resolve_pronunciation as rp


def test_kaikki_wins_when_present():
    hit = rp.resolve_ipa(kaikki_entry={"ipa": [{"ipa": "ˈtɛstwɜːd", "tags": ["US"]}]},
                          wordnik_raw="T EH1 S T", wordnik_type="arpabet",
                          local_entries=[("noun", "def", "ˈlkl", False, False)])
    assert hit == ("ˈtɛstwɜːd", rp.Tier.KAIKKI)


def test_falls_through_to_wordnik_when_kaikki_has_nothing():
    hit = rp.resolve_ipa(kaikki_entry=None, wordnik_raw="T EH1 S T", wordnik_type="arpabet",
                          local_entries=[("noun", "def", "ˈlkl", False, False)])
    assert hit is not None
    ipa, tier = hit
    assert tier is rp.Tier.WORDNIK
    assert ipa  # arpabet.to_ipa("T EH1 S T") -> a real IPA string


def test_wordnik_gcide_diacritical_has_no_converter_falls_through():
    hit = rp.resolve_ipa(kaikki_entry=None, wordnik_raw="tehst", wordnik_type="gcide-diacritical",
                          local_entries=[("noun", "def", "ˈlkl", False, False)])
    assert hit == ("ˈlkl", rp.Tier.LOCAL_WIKTIONARY)


def test_falls_through_to_local_wiktionary():
    hit = rp.resolve_ipa(kaikki_entry=None, wordnik_raw=None, wordnik_type=None,
                          local_entries=[("noun", "def", "ˈlkl", False, False)],
                          oed_matches=["ˈoʊiːdiː"])
    assert hit == ("ˈlkl", rp.Tier.LOCAL_WIKTIONARY)


def test_falls_through_to_oed_when_only_source():
    hit = rp.resolve_ipa(oed_matches=["ˈoʊiːdiː"])
    assert hit == ("ˈoʊiːdiː", rp.Tier.OED)


def test_oed_ambiguous_homographs_contribute_nothing():
    # Two disagreeing oed.entry rows for the same headword -- a genuine
    # conflict is skipped, not guessed at (OED's part_of_speech field is
    # unparsed OCR soup, not usable to pick the "right" homograph).
    hit = rp.resolve_ipa(oed_matches=["ˈwʌn", "ˈtuː"])
    assert hit is None


def test_nothing_resolves_returns_none():
    assert rp.resolve_ipa() is None


def test_max_tier_cutoff_stops_before_wordnik():
    hit = rp.resolve_ipa(kaikki_entry=None, wordnik_raw="T EH1 S T", wordnik_type="arpabet",
                          local_entries=[("noun", "def", "ˈlkl", False, False)],
                          max_tier=rp.Tier.KAIKKI)
    assert hit is None


def test_invalid_kaikki_ipa_falls_through_to_next_tier():
    # A source producing something that fails the English-IPA sanity check
    # doesn't get to win just because it technically returned a string.
    hit = rp.resolve_ipa(kaikki_entry={"ipa": [{"ipa": "myʁ.my.ʁe", "tags": []}]},
                          local_entries=[("noun", "def", "ˈlkl", False, False)])
    assert hit == ("ˈlkl", rp.Tier.LOCAL_WIKTIONARY)
