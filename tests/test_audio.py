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


def test_normalize_keep_optional_false_drops_the_parenthetical_entirely():
    # OED's (r) marks an RP linking/intrusive r -- pronounced only in
    # connected speech before a following vowel, dropped in citation form.
    # keep_optional=True (the default, used above) is right for the US voice
    # where post-vocalic r is always pronounced; a UK voice needs the whole
    # bracketed span dropped instead, or it gets a rhotic mispronunciation
    # exactly backwards from what the notation means.
    assert audio.normalize_ipa("/ˈdʒɪbə(ɹ)/", keep_optional=False) == "ˈdʒɪbə"


def test_normalize_keep_optional_false_handles_a_still_unclosed_paren():
    # defensive: _close_unbalanced_paren should already have fixed this at
    # write time, but normalize_ipa shouldn't leave a dangling "(" if it sees
    # one anyway.
    assert audio.normalize_ipa("ˈbændənə(r", keep_optional=False) == "ˈbændənə"


# --- dialect selection -------------------------------------------------------

def test_ipa_dialect_for_source_oed_is_uk():
    assert audio.ipa_dialect_for_source("oed") == "uk"


def test_ipa_dialect_for_source_everything_else_is_us():
    # every non-oed source is deliberately US-biased already (kaikki prefers
    # US-tagged entries, local Wiktionary's column is us_pronunciation,
    # Wordnik's ARPAbet/AHD-5 converters are both US phoneme systems); None
    # covers legacy rows that predate ipa_source entirely.
    for source in ("kaikki", "wordnik", "local_wiktionary", None):
        assert audio.ipa_dialect_for_source(source) == "us"


def test_voice_for_dialect_maps_to_matching_azure_voice_and_lang():
    assert audio.voice_for_dialect("us") == (audio.AZURE_VOICE, "en-US")
    assert audio.voice_for_dialect("uk") == (audio.AZURE_VOICE_UK, "en-GB")


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


# --- SSML lang/voice forwarding ----------------------------------------------

def test_synthesize_azure_forwards_voice_and_lang_into_ssml(monkeypatch):
    captured = {}

    def fake_synthesize_ssml(ssml, key, region, tries=4):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure("abandoner", "ˈbændənər", "key", "region",
                            voice=audio.AZURE_VOICE_UK, lang="en-GB")
    assert "xml:lang='en-GB'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE_UK}'" in captured["ssml"]


def test_synthesize_azure_guess_forwards_voice_and_lang_into_ssml(monkeypatch):
    captured = {}

    def fake_synthesize_ssml(ssml, key, region, tries=4):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure_guess("abandoner", "key", "region",
                                  voice=audio.AZURE_VOICE_UK, lang="en-GB")
    assert "xml:lang='en-GB'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE_UK}'" in captured["ssml"]


def test_synthesize_azure_defaults_to_us_voice_and_lang(monkeypatch):
    captured = {}

    def fake_synthesize_ssml(ssml, key, region, tries=4):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure("abandoner", "ˈbændənər", "key", "region")
    assert "xml:lang='en-US'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE}'" in captured["ssml"]
