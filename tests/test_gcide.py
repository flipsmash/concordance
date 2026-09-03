"""GCIDE diacritical -> IPA conversion (pure logic; validated separately by
running against all 918 real words that had wordnik_pron_raw under this
rawType but no word.ipa as of 2026-09-03 — 387 converted, the rest genuinely
unrecoverable OCR corruption or elided-prefix source rows)."""
from __future__ import annotations
from concordance import gcide


def test_basic_stress_and_syllables():
    assert gcide.to_ipa('(băt"fụl)') == "ˈbætfəl"


def test_stress_mark_keeps_coda_with_completed_syllable():
    # "klŏk\"līk`" -- the coda /k/ of "clock" must stay in the first
    # syllable, not get pulled into the second ("līk")'s onset. This is the
    # coda/onset misassignment bug found and fixed during development.
    assert gcide.to_ipa('(klŏk"līk`)') == "ˈklɒkˌlaɪk"


def test_primary_and_secondary_stress_marks():
    assert gcide.to_ipa('(răs`pȧ*tō"rĭ*ŭm)') == "ˌɹæspæˈtoʊɹɪʌm"


def test_e_tilde_before_r():
    assert gcide.to_ipa('(gẽrt"līn`)') == "ˈɡɜrtˌlaɪn"


def test_n_submacron_is_velar_nasal():
    assert gcide.to_ipa("(bŭṉk\"bĕd)") == "ˈbʌŋkbɛd"


def test_u_dot_below_in_ful_suffix():
    assert gcide.to_ipa('(tīm"fụl)') == "ˈtaɪmfəl"


def test_plain_oi_and_ou_digraphs():
    assert gcide.to_ipa('(toi"sŭm)') == "ˈtɔɪsʌm"
    assert gcide.to_ipa('(hāl"shŏt`)') == "ˈheɪlˌʃɒt"


def test_apostrophe_is_elided_schwa():
    assert gcide.to_ipa('(bŏks"\'n)') == "ˈbɒksən"


def test_takes_first_of_semicolon_variants():
    assert gcide.to_ipa('(snŏf; 115)') == "snɒf"


def test_takes_first_of_or_variants():
    assert gcide.to_ipa('(lĕp"ĭ*dĭn or *dēn)') is not None


def test_fails_closed_on_ocr_placeholder():
    assert gcide.to_ipa("(?; 277)") is None
    assert gcide.to_ipa("(k?r-r?k\"t?-f?)") is None


def test_fails_closed_on_elided_prefix():
    # cabalist -- GCIDE shows only the changed suffix, reusing the base
    # word's own pronunciation for the rest. Not a complete pronunciation:
    # synthesizing this against the full headword would anchor Azure's
    # <phoneme> tag to the wrong, truncated sound.
    assert gcide.to_ipa('(-lĭst)') is None


def test_fails_closed_on_french_nasal():
    assert gcide.to_ipa("(räNt)") is None


def test_fails_closed_on_plain_english_stand_in():
    assert gcide.to_ipa("(-sound)") is None


def test_unrecognized_symbol_fails_closed():
    assert gcide.to_ipa("(xyz123!!!)") is None
