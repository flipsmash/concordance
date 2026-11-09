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
