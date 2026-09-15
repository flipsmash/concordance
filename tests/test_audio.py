"""Pronunciation audio: pure logic only (Commons fetch + Azure synthesis are live calls)."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import Mock
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

    def fake_synthesize_ssml(ssml, key, region, tries=4, word=""):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure("abandoner", "ˈbændənər", "key", "region",
                            voice=audio.AZURE_VOICE_UK, lang="en-GB")
    assert "xml:lang='en-GB'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE_UK}'" in captured["ssml"]


def test_synthesize_azure_guess_forwards_voice_and_lang_into_ssml(monkeypatch):
    captured = {}

    def fake_synthesize_ssml(ssml, key, region, tries=4, word=""):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure_guess("abandoner", "key", "region",
                                  voice=audio.AZURE_VOICE_UK, lang="en-GB")
    assert "xml:lang='en-GB'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE_UK}'" in captured["ssml"]


def test_synthesize_azure_defaults_to_us_voice_and_lang(monkeypatch):
    captured = {}

    def fake_synthesize_ssml(ssml, key, region, tries=4, word=""):
        captured["ssml"] = ssml
        return b"fake-mp3-bytes"

    monkeypatch.setattr(audio, "_synthesize_ssml", fake_synthesize_ssml)
    audio.synthesize_azure("abandoner", "ˈbændənər", "key", "region")
    assert "xml:lang='en-US'" in captured["ssml"]
    assert f"name='{audio.AZURE_VOICE}'" in captured["ssml"]


# --- failure visibility -------------------------------------------------------
# Regression tests for a real production incident: a whole Azure run (10,690
# words, logs/audio_20260904_123145.log) produced 0 Azure-synthesized audio
# with nothing in the logs to say why, because every failure path returned
# None bare. _synthesize_ssml must now print the actual reason.

def test_synthesize_ssml_prints_status_and_body_on_non_200(monkeypatch, capsys):
    resp = Mock(status_code=401, text="Access denied due to invalid subscription key")
    monkeypatch.setattr(audio.requests, "post", lambda *a, **k: resp)
    result = audio._synthesize_ssml("<speak/>", "bad-key", "eastus", tries=1, word="abandoner")
    assert result is None
    out = capsys.readouterr().out
    assert "401" in out
    assert "abandoner" in out
    assert "Access denied" in out


def test_synthesize_ssml_prints_after_retries_exhausted_on_network_error(monkeypatch, capsys):
    def raise_conn_error(*a, **k):
        raise audio.requests.RequestException("boom")
    monkeypatch.setattr(audio.requests, "post", raise_conn_error)
    monkeypatch.setattr(audio.time, "sleep", lambda s: None)
    result = audio._synthesize_ssml("<speak/>", "key", "eastus", tries=2, word="abandoner")
    assert result is None
    out = capsys.readouterr().out
    assert "abandoner" in out
    assert "boom" in out


def test_synthesize_ssml_silent_on_success(monkeypatch, capsys):
    resp = Mock(status_code=200, content=b"fake-mp3-bytes")
    monkeypatch.setattr(audio.requests, "post", lambda *a, **k: resp)
    result = audio._synthesize_ssml("<speak/>", "key", "eastus", tries=1, word="abandoner")
    assert result == b"fake-mp3-bytes"
    assert capsys.readouterr().out == ""


# --- Piper: local grapheme-only synthesis ------------------------------------

def test_synthesize_piper_returns_none_when_voice_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(audio, "_piper_voice", lambda: None)
    assert audio.synthesize_piper("abomine") is None


def test_synthesize_piper_transcodes_via_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "AUDIO_DIR", tmp_path)

    class FakeVoice:
        def synthesize_wav(self, word, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00")

    monkeypatch.setattr(audio, "_piper_voice", lambda: FakeVoice())

    def fake_run(cmd, **kwargs):
        dest_mp3 = Path(cmd[-1])
        dest_mp3.write_bytes(b"fake-mp3-bytes")
        return Mock(returncode=0)

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    result = audio.synthesize_piper("abomine")
    assert result == b"fake-mp3-bytes"
    # temp files cleaned up, nothing left behind in AUDIO_DIR
    assert list(tmp_path.iterdir()) == []


def test_synthesize_piper_returns_none_on_ffmpeg_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "AUDIO_DIR", tmp_path)

    class FakeVoice:
        def synthesize_wav(self, word, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00")

    monkeypatch.setattr(audio, "_piper_voice", lambda: FakeVoice())
    monkeypatch.setattr(audio.subprocess, "run", lambda cmd, **kwargs: Mock(returncode=1))
    assert audio.synthesize_piper("abomine") is None
    assert list(tmp_path.iterdir()) == []


def test_piper_voice_caches_across_calls(monkeypatch):
    monkeypatch.setattr(audio, "_piper_voice_cache", None)
    calls = []

    class FakePiperModule:
        class PiperVoice:
            @staticmethod
            def load(path):
                calls.append(path)
                return "the-voice"

    monkeypatch.setitem(sys.modules, "piper", FakePiperModule)
    assert audio._piper_voice() == "the-voice"
    assert audio._piper_voice() == "the-voice"
    assert len(calls) == 1  # loaded once, cached on the second call


def test_piper_voice_degrades_to_none_when_load_fails(monkeypatch):
    monkeypatch.setattr(audio, "_piper_voice_cache", None)

    class FakePiperModule:
        class PiperVoice:
            @staticmethod
            def load(path):
                raise OSError("no such file")

    monkeypatch.setitem(sys.modules, "piper", FakePiperModule)
    assert audio._piper_voice() is None
