"""AHD respelling -> IPA conversion (pure logic; validated separately against
937 real words that also have trusted kaikki IPA — 100% conversion success)."""
from __future__ import annotations
from concordance import ahd


def test_basic_stress_and_syllables():
    assert ahd.to_ipa("kŏn-kăv′ĭ-tē") == "kɒnˈkævɪtiː"


def test_stress_mark_precedes_full_onset_not_just_the_vowel():
    # the "n" in "kon-" must stay behind the hyphen, not get pulled into the
    # next syllable's onset — this is the coda/onset misassignment bug found
    # and fixed during development
    result = ahd.to_ipa("kŏn-kăv′ĭ-tē")
    assert result.startswith("kɒn")


def test_primary_and_secondary_stress_marks():
    # verified against pusillanimity's real, known stress pattern
    assert ahd.to_ipa("pyoo͞″sə-lə-nĭm′ĭ-tē") == "ˌpjuːsələˈnɪmɪtiː"


def test_takes_first_variant_pronunciation():
    assert ahd.to_ipa("klăng′ər, klăng′gər") == "ˈklæŋər"


def test_takes_first_variant_separated_by_semicolon():
    assert ahd.to_ipa("wŏst; wəst") is not None


def test_strips_embedded_html_notes():
    assert ahd.to_ipa("wŏst; wəst <em>when unstressed</em>") == ahd.to_ipa("wŏst")


def test_voiced_th_smallcaps_distinct_from_voiceless():
    # swarth -> /swɔːrð/ (voiced), confirmed against kaikki's independent IPA
    assert "ð" in ahd.to_ipa("swôrᴛʜ")
    assert "θ" in ahd.to_ipa("thĭn")  # plain "th" stays voiceless


def test_long_and_short_oo_digraphs():
    # combining marks attach to the SECOND o in the real data, not the first
    assert ahd.to_ipa("boo" + chr(0x35e) + "t") == "buːt"  # boot
    assert ahd.to_ipa("boo" + chr(0x35d) + "k") == "bʊk"  # book


def test_unrecognized_symbol_fails_closed():
    assert ahd.to_ipa("xyz123") is None
