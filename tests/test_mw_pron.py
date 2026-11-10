"""Merriam-Webster respelling -> IPA (concordance/mw_pron.py)."""

import pytest

from concordance.audio import looks_like_english_ipa
from concordance.mw_pron import mw_to_ipa


@pytest.mark.parametrize("mw, ipa", [
    ("ˈkän-ˌkȯr-dən(t)s", "ˈkɑnˌkɔrdəns"),         # optional sound dropped
    ("sü-ˈsər-əs", "suˈsərəs"),
    ("ˌka-lə-ˈpi-j(ē-)ən", "ˌkæləˈpɪdʒən"),
    ("ˈnō-(ˌ)lō-kən-ˈten-də-rē", "ˈnoʊloʊkənˈtɛndəri"),   # optional stress dropped
    ("ˈt͟hen", "ˈðɛn"),
    ("ˈchərch", "ˈtʃərtʃ"),
    ("ˈau̇t-ˌsīd", "ˈaʊtˌsaɪd"),
    ("ˈȯi-əl", "ˈɔɪəl"),
    ("ˈsiŋ-ər", "ˈsɪŋər"),
    ("ˈyel-ō", "ˈjɛloʊ"),
    ("ˈmas-hed, -ˌhed", "ˈmæshɛd"),                # s-h across a syllable break; first variant
])
def test_converts_respellings(mw, ipa):
    assert mw_to_ipa(mw) == ipa
    assert looks_like_english_ipa(ipa)


@pytest.mark.parametrize("mw", ["-ˈdäm", "ˌkräk-", "byaⁿ-sā-äⁿs", "ˈfœr", "", "ˈ-"])
def test_refuses_partial_foreign_or_empty(mw):
    assert mw_to_ipa(mw) is None


def test_fetch_recording_keeps_the_old_clip_on_a_bad_download(monkeypatch, tmp_path):
    import requests

    from concordance import mw_pron

    class Resp:
        def __init__(self, status, content):
            self.status_code, self.content = status, content

    dest = tmp_path / "gamin.mp3"
    dest.write_bytes(b"old piper clip")
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, b"<html>blocked</html>"))
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert mw_pron.fetch_recording("https://x/gamin001.mp3", dest) is False
    assert dest.read_bytes() == b"old piper clip"

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, b"ID3\x04mp3 data"))
    assert mw_pron.fetch_recording("https://x/gamin001.mp3", dest) is True
    assert dest.read_bytes() == b"ID3\x04mp3 data"
