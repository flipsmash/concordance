"""wikt.sound (loaded once by `wiktextract-sounds`) must answer exactly what a
scan of the dump would. Needs CONCORDANCE_TEST_DB_URL (else skipped)."""

from __future__ import annotations

import gzip
import json
import os

import pytest

from concordance import db, wiktextract

_URL = os.environ.get("CONCORDANCE_TEST_DB_URL", "")


def _connectable(url):
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not (_URL and _connectable(_URL)),
                        reason="set CONCORDANCE_TEST_DB_URL to a disposable Postgres to run")

_ENTRIES = [
    {"word": "Lead", "lang_code": "en", "sounds": [{"ipa": "/liːd/", "tags": ["UK"]},
                                                  {"ogg_url": "https://x/lead1.ogg", "tags": ["US"]}]},
    {"word": "lead", "lang_code": "fr", "sounds": [{"ipa": "/lɛd/"}]},            # not English
    {"word": "lead", "lang_code": "en", "sounds": [{"ipa": "/lɛd/", "tags": ["US"]},
                                                  {"mp3_url": "https://x/lead2.mp3"}]},
    {"word": "quire", "lang_code": "en", "senses": []},                          # no sounds
    {"word": "ogg", "lang_code": "en", "sounds": [{"ogg_url": "https://x/ogg.ogg"}]},
]


@pg
def test_sound_table_matches_a_dump_scan(tmp_path):
    dump = tmp_path / "dump.jsonl.gz"
    with gzip.open(dump, "wt", encoding="utf-8") as f:
        for e in _ENTRIES:
            f.write(json.dumps(e) + "\n")
        f.write("{not json\n")
    schema = "cc_test_wikt"
    conn = db.connect(_URL)
    try:
        assert wiktextract.load_sounds(conn, dump, wikt_schema=schema) == {"entries": 3, "terms": 2}
        lemmas = {"lead", "quire", "ogg", "absent"}
        from_table = wiktextract.sound_lexicon(conn, lemmas, wikt_schema=schema)
        assert from_table == wiktextract.build_lexicon(dump, lemmas)
        # both entries' sounds, concatenated in dump order (best_ipa's fallback depends on it)
        assert [e["ipa"] for e in from_table["lead"]["ipa"]] == ["/liːd/", "/lɛd/"]
        assert wiktextract.best_ipa(from_table["lead"]["ipa"]) == "/lɛd/"
        # an explicit dump path still scans the dump
        assert wiktextract.sound_lexicon(conn, {"ogg"}, dump, wikt_schema=schema) == \
            wiktextract.build_lexicon(dump, {"ogg"})
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
        conn.close()


def test_affix_root_takes_the_single_free_component_only():
    ser = {"word": "serpenticide", "etymology_templates": [
        {"name": "affix", "args": {"1": "en", "2": "serpent", "3": "-i-"}},
        {"name": "suffix", "args": {"1": "en", "2": "", "3": "-cide", "nocat": "1"}}]}
    assert wiktextract.affix_root(ser) == "serpent"
    # a compound of two free words is left to a transparency judgment
    assert wiktextract.affix_root({"word": "downland", "etymology_templates": [
        {"name": "compound", "args": {"1": "en", "2": "down", "3": "land"}}]}) is None
    assert wiktextract.affix_root({"word": "x", "etymology_templates": [
        {"name": "affix", "args": {"1": "en", "2": "down", "3": "land"}}]}) is None
    # another language's affixes, or an inline gloss modifier
    assert wiktextract.affix_root({"word": "y", "etymology_templates": [
        {"name": "affix", "args": {"1": "la", "2": "serpens", "3": "-cida"}}]}) is None
    assert wiktextract.affix_root({"word": "weedicide", "etymology_templates": [
        {"name": "suffix", "args": {"1": "en", "2": "weed<t:plant>", "3": "-icide"}}]}) == "weed"
    assert wiktextract.affix_root({"word": "coliform", "etymology_templates": [
        {"name": "affix", "args": {"1": "en", "2": "la:colon", "3": "-iform"}}]}) is None
    assert wiktextract.affix_root({"word": "cardinalate", "etymology_templates": [
        {"name": "suffix", "args": {"1": "en", "2": "cardinal#Noun", "3": "-ate"}}]}) == "cardinal"


def test_compound_parts_only_when_they_spell_the_word():
    def obj(word, *templates):
        return {"word": word, "etymology_templates": list(templates)}
    assert wiktextract.compound_parts(obj("downland",
        {"name": "compound", "args": {"1": "en", "2": "down", "3": "land"}})) == ["down", "land"]
    # {{affix}} with two free words is a compound too; a trailing suffix may ride along
    assert wiktextract.compound_parts(obj("drygulcher",
        {"name": "af", "args": {"1": "en", "2": "dry", "3": "gulch", "4": "-er"}})) == ["dry", "gulch"]
    # gloss / POS modifiers dropped
    assert wiktextract.compound_parts(obj("eelpot",
        {"name": "com", "args": {"1": "en", "2": "eel<t:fish>", "3": "pot#Noun"}})) == ["eel", "pot"]
    # history, not a visible split
    assert wiktextract.compound_parts(obj("belfry",
        {"name": "compound", "args": {"1": "en", "2": "barrow", "3": "frith"}})) is None
    # root + affixes is affix_root's business; other languages' compounds are not ours
    assert wiktextract.compound_parts(obj("serpenticide",
        {"name": "affix", "args": {"1": "en", "2": "serpent", "3": "-icide"}})) is None
    assert wiktextract.compound_parts(obj("downland",
        {"name": "compound", "args": {"1": "enm", "2": "down", "3": "land"}})) is None


def test_etymology_links_classic_and_etymon_markup():
    grease = {"lang_code": "en", "word": "grease", "etymology_templates": [
        {"name": "inh", "args": {"1": "en", "2": "enm", "3": "grece"}},
        {"name": "der", "args": {"1": "en", "2": "xno", "3": "craisse"}},
        {"name": "der", "args": {"1": "en", "2": "la", "3": "crassus"}},
        {"name": "cog", "args": {"1": "fr", "2": "graisse"}}]}             # a cognate is not a source
    assert wiktextract.etymology_links(grease) == (["enm:grece", "xno:craisse", "la:crassus"], [])
    free = {"lang_code": "en", "word": "free", "etymology_templates": [
        {"name": "etymon", "args": {"1": "en", "id": "adjective", "2": ":inh", "3": "enm:fre<id:x>", "tree": "1"}}]}
    assert wiktextract.etymology_links(free) == (["enm:fre"], [])


def test_etymology_links_formation_parts():
    def parts(name, *args):
        a = {"1": "en", **{str(i + 2): v for i, v in enumerate(args)}}
        return wiktextract.etymology_links({"lang_code": "en", "etymology_templates": [{"name": name, "args": a}]})[1]
    assert parts("suffix", "grease", "y") == ["grease"]          # the affix comes without its hyphen
    assert parts("prefix", "un", "smiling") == ["smiling"]
    assert parts("confix", "ichthyo", "logy") == []              # no stem in the middle
    assert parts("confix", "en", "light", "en") == ["light"]
    assert parts("compound", "wood", "chopper") == ["wood", "chopper"]
    assert parts("af", "dry", "gulch", "-er") == ["dry", "gulch"]
