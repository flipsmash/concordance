"""MW Collegiate Dictionary API response parsing (pure logic; the raw dict
fixtures below mirror real response shapes observed from the live API)."""
from __future__ import annotations

from unittest.mock import MagicMock

from concordance import mw


def test_clean_mw_markup_bold_colon():
    assert mw._clean_mw_markup("{bc}an alphabetical index") == ":an alphabetical index"


def test_clean_mw_markup_italics_stripped():
    assert mw._clean_mw_markup("from {it}concordantia{/it}, cf.") == "from concordantia, cf."


def test_clean_mw_markup_cross_reference_keeps_target_word():
    assert mw._clean_mw_markup("{sx|concord||}, {sx|agreement||}") == "concord, agreement"


def test_clean_mw_markup_unknown_token_dropped_not_crashed():
    assert mw._clean_mw_markup("14th century{ds||1||}") == "14th century"


def test_audio_subdir_bix_prefix():
    assert mw._audio_subdir("bixspecial") == "bix"


def test_audio_subdir_gg_prefix():
    assert mw._audio_subdir("gg1234") == "gg"


def test_audio_subdir_leading_digit():
    assert mw._audio_subdir("1running") == "number"


def test_audio_subdir_plain_word():
    assert mw._audio_subdir("concor02") == "c"


def test_parse_api_entry_extracts_all_fields():
    raw = {
        "meta": {"id": "concordance"},
        "fl": "noun",
        "hwi": {
            "hw": "con*cord*ance",
            "prs": [{"mw": "kən-ˈkȯr-dᵊn(t)s", "sound": {"audio": "concor02"}}],
        },
        "shortdef": ["{bc}an alphabetical index", "{sx|concord||}, {sx|agreement||}"],
        "et": [["text", "Middle English, from {it}concordantia{/it}"]],
        "date": "14th century{ds||1||}",
    }
    entry = mw._parse_api_entry(raw)
    assert entry.headword == "concordance"
    assert entry.part_of_speech == "noun"
    assert entry.definitions == [":an alphabetical index", "concord, agreement"]
    assert entry.pronunciations[0].respelling == "kən-ˈkȯr-dᵊn(t)s"
    assert entry.pronunciations[0].audio_url == (
        "https://media.merriam-webster.com/audio/prons/en/us/mp3/c/concor02.mp3"
    )
    assert entry.etymology == "Middle English, from concordantia"
    assert entry.first_known_use == "14th century"
    assert entry.source == "Merriam-Webster API"


def test_parse_api_entry_returns_none_without_definitions():
    raw = {"meta": {"id": "x"}, "fl": "noun", "hwi": {}, "shortdef": []}
    assert mw._parse_api_entry(raw) is None


def _isolate_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(mw, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(mw, "_CACHE_FILE", tmp_path / "mw_api_cache.json")
    monkeypatch.setattr(mw, "_USAGE_FILE", tmp_path / "mw_api_usage.json")


def test_lookup_api_no_exact_match_suggestions_are_a_cacheable_miss(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    resp = MagicMock(status_code=200)
    resp.json.return_value = ["run", "runt", "rune"]   # MW's "did you mean" shape
    monkeypatch.setattr(mw, "_get", lambda *a, **k: resp)

    assert mw.lookup_api("runn", "key") == []
    assert mw._load_json(mw._CACHE_FILE) == {"runn": ["run", "runt", "rune"]}


def test_lookup_api_does_not_cache_a_bad_key_response(monkeypatch, tmp_path):
    # Regression: an early version cached `[]` on ANY non-200 (bad key, 5xx,
    # network death), permanently marking that word unresolvable via the API
    # -- silently falling through to the scrape fallback forever after, even
    # once the key was fixed.
    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(mw, "_get", lambda *a, **k: MagicMock(status_code=403))

    assert mw.lookup_api("concordance", "bad-key") == []
    assert mw._load_json(mw._CACHE_FILE) == {}

    # once the key is fixed, the SAME word must still be able to hit the real API
    resp = MagicMock(status_code=200)
    resp.json.return_value = [{"meta": {"id": "concordance"}, "fl": "noun",
                                "hwi": {}, "shortdef": ["a thing"]}]
    monkeypatch.setattr(mw, "_get", lambda *a, **k: resp)
    entries = mw.lookup_api("concordance", "good-key")
    assert len(entries) == 1 and entries[0].headword == "concordance"


def test_lookup_api_does_not_cache_unreachable_network(monkeypatch, tmp_path):
    _isolate_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(mw, "_get", lambda *a, **k: None)

    assert mw.lookup_api("concordance", "key") == []
    assert mw._load_json(mw._CACHE_FILE) == {}


# --- pick_entry / exact_matches / is_foreign_pos (db.mw_backfill support) ---

def _entry(headword: str, pos: str, defs=("a definition",)) -> mw.MWEntry:
    return mw.MWEntry(headword=headword, part_of_speech=pos, definitions=list(defs))


def test_pick_entry_single_entry_short_circuits():
    only = _entry("run", "noun")
    assert mw.pick_entry([only], tagger_pos="VERB") is only


def test_pick_entry_matches_tagger_pos_over_first_listed():
    noun, verb = _entry("run", "noun"), _entry("run", "verb")
    assert mw.pick_entry([noun, verb], tagger_pos="VERB") is verb


def test_pick_entry_falls_back_to_first_when_no_pos_hint():
    first, second = _entry("run", "noun"), _entry("run", "verb")
    assert mw.pick_entry([first, second], tagger_pos="") is first


def test_pick_entry_falls_back_to_first_when_no_pos_matches():
    first, second = _entry("run", "noun"), _entry("run", "verb")
    assert mw.pick_entry([first, second], tagger_pos="ADJ") is first


def test_exact_matches_rejects_fuzzy_idiom_hit():
    # Regression: MW's API fuzzy-matched the literal query "atune" to the
    # unrelated idiom "sing a different tune" -- confirmed on live data.
    # Writing that idiom's definition under "atune" would be wrong.
    idiom = _entry("sing a different tune", "phrase")
    assert mw.exact_matches([idiom], "atune") == []


def test_exact_matches_ignores_case_and_punctuation():
    # "con*cord*ance" is how MW's API literally returns headword syllable
    # marks; a real match must still be recognized despite them.
    entry = _entry("con*cord*ance", "noun")
    assert mw.exact_matches([entry], "Concordance") == [entry]


def test_is_foreign_pos_recognizes_capitalized_language_tag():
    # Confirmed on live data: querying "hatari" returned fl="Swahili noun" --
    # a real foreign loanword this project's other sources never surface.
    assert mw.is_foreign_pos("Swahili noun") is True
    assert mw.is_foreign_pos("French adjective") is True


def test_is_foreign_pos_does_not_flag_ordinary_lowercase_modifiers():
    assert mw.is_foreign_pos("plural noun") is False
    assert mw.is_foreign_pos("combining form") is False
    assert mw.is_foreign_pos("noun") is False
