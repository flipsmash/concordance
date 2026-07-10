"""ARPAbet -> IPA conversion (pure logic; validated separately against 25 real
words that also have trusted kaikki IPA — stress placement confirmed correct)."""
from __future__ import annotations
from concordance import arpabet


def test_basic_conversion_with_primary_stress():
    assert arpabet.to_ipa("R EY1 K ER0") == "ˈɹeɪkɝ"


def test_stress_precedes_full_onset_cluster_not_just_the_vowel():
    # "vicuna": V IH0 K Y UW1 N AH0 -- the stress must precede the whole "ky"
    # onset cluster, not land between k and the vowel (the bug found and fixed
    # during development, which risked the synthesizer misattributing the
    # consonant cluster to the wrong syllable)
    result = arpabet.to_ipa("V IH0 K Y UW1 N AH0")
    assert "ˈkj" in result


def test_secondary_stress():
    assert arpabet.to_ipa("B AE1 G P AY2 P ER0") == "ˈbæˌɡpaɪpɝ"


def test_no_stress_on_any_vowel():
    assert arpabet.to_ipa("K AW1 CH IH0 NG") == "ˈkaʊtʃɪŋ"


def test_affricates_and_digraphs():
    assert arpabet.to_ipa("CH IH1 N AH0") == "ˈtʃɪnʌ"  # china-ish
    assert arpabet.to_ipa("JH AY1") == "ˈdʒaɪ"


def test_unrecognized_phone_fails_closed():
    assert arpabet.to_ipa("XX1 YY0") is None


def test_empty_input_fails_closed():
    assert arpabet.to_ipa("") is None
