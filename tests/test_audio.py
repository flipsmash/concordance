"""Pronunciation audio: pure logic only (Commons fetch + Azure synthesis are live calls)."""
from __future__ import annotations
from concordance import audio


def test_normalize_strips_slash_delimiters():
    assert audio.normalize_ipa("/bɑːtɪˈzæn/") == "bɑːtɪˈzæn"


def test_normalize_strips_bracket_delimiters():
    assert audio.normalize_ipa("[ˈbætəɫmənt]") == "ˈbætəɫmənt"


def test_normalize_strips_tie_bar_keeps_both_letters():
    # t͡ʃ (tie-barred) -> tʃ (decomposed): verified empirically that both Azure
    # and the local test model expect the decomposed form, not a ligature.
    assert audio.normalize_ipa("/bɪˈt͡ʃæns/") == "bɪˈtʃæns"


def test_normalize_strips_syllable_dots():
    assert audio.normalize_ipa("/ˈbɝ.ɡəˌnɛt/") == "ˈbɝɡəˌnɛt"


def test_normalize_handles_plain_ipa_with_no_delimiters():
    assert audio.normalize_ipa("əˈɹɔɪnt") == "əˈɹɔɪnt"


def test_normalize_idempotent_on_already_clean_input():
    clean = "ɑːˈbɪtɹəmənt"
    assert audio.normalize_ipa(clean) == clean


def test_normalize_strips_optional_sound_parentheses_keeps_contents():
    # real bug found in production: kaikki marks a dialectal-optional sound in
    # parens (e.g. dropped r in non-rhotic dialects); literal "(" ")" aren't
    # valid phoneme characters and Azure silently rejected them, dropping 165
    # words with perfectly good IPA into the no-data bucket. Keep the sound
    # (fuller pronunciation), just remove the parens.
    assert audio.normalize_ipa("/ˈdʒɪbə(ɹ)/") == "ˈdʒɪbəɹ"
    assert audio.normalize_ipa("/kənˈvɛntɪk(ə)l/") == "kənˈvɛntɪkəl"


# --- language sanity guard --------------------------------------------------

def test_rejects_french_ipa_leaked_via_cross_reference():
    # real bug found in production: word.ipa for "murmurer"/"angelus" had the
    # French cognate's transcription instead of English
    assert not audio.looks_like_english_ipa("/myʁ.my.ʁe/")
    assert not audio.looks_like_english_ipa("/ɑ̃.ʒe.lys/")


def test_accepts_plain_english_ipa():
    assert audio.looks_like_english_ipa("/bɑːtɪˈzæn/")
    assert audio.looks_like_english_ipa("/bɪˈt͡ʃæns/")


def test_rejects_empty_or_none_ipa():
    assert not audio.looks_like_english_ipa("")
