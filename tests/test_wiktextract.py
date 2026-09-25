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
