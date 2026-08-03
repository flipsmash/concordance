"""oed schema: pronunciation lookup helpers used by concordance.db.backfill_ipa_from_oed
(needs a real, disposable Postgres — see tests/test_db.py's own `pg` marker for setup)."""
from __future__ import annotations

import os

import pytest

from concordance import db
from concordance.oed import db as oed_db

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


def test_close_unbalanced_paren_closes_a_real_dangling_linking_r():
    # real corpus pattern, confirmed live: 615/5007 resolved OED entries end
    # in an unclosed linking-r marker, e.g. "abandoner" -> "ˈbændənə(r"
    # (missing the closing paren) -- an upstream oed/pronunciation.py
    # extraction artifact, worked around here rather than at the source.
    assert oed_db._close_unbalanced_paren("ˈbændənə(r") == "ˈbændənə(r)"


def test_close_unbalanced_paren_leaves_already_balanced_input_alone():
    assert oed_db._close_unbalanced_paren("ˈstrɒfɔɪd") == "ˈstrɒfɔɪd"
    assert oed_db._close_unbalanced_paren("ˈdʒɪbə(ɹ)") == "ˈdʒɪbə(ɹ)"


def test_close_unbalanced_paren_does_not_guess_at_other_mismatches():
    # anything other than "exactly one more ( than )" is left alone rather
    # than guessed at -- this helper targets one specific known artifact,
    # not general paren-repair.
    weird = "a)b(c"
    assert oed_db._close_unbalanced_paren(weird) == weird


@pg
def test_pronunciation_lexicon_empty_input_returns_empty_dict():
    conn = db.connect(_URL)
    assert oed_db.pronunciation_lexicon(conn, set(), schema="oed") == {}
    conn.close()


@pg
def test_pronunciation_lexicon_bulk_lookup_and_dedup_and_paren_fix():
    schema = "oed_test_pron"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    oed_db.apply_schema(conn, schema)

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="abc123",
                                      volume_label="Test Volume", page_count=1, schema=schema)

    def _entry(headword, ipa, needs_review=False):
        entry_id = oed_db.insert_entry(
            conn, volume_id=volume_id, headword=headword, homograph_number=None,
            part_of_speech="n", etymology=None, entry_type="main", parent_entry_id=None,
            page_number=1, raw_text="raw", schema=schema)
        oed_db.update_pronunciation(conn, entry_id, pronunciation_raw="raw",
                                     pass1=ipa, pass2=ipa, ipa=ipa, source="vision_llm",
                                     needs_review=needs_review, schema=schema)
        return entry_id

    _entry("abandoner", "ˈbændənə(r")           # unclosed paren -> should be fixed on lookup
    _entry("bay", "beɪ")                          # homograph entry 1, agrees...
    _entry("bay", "beɪ")                          # ...homograph entry 2, same IPA -- still 1 distinct
    _entry("fleet", "fliːt")                       # homograph entry 1
    _entry("fleet", "flɛt")                        # homograph entry 2 -- genuinely conflicting
    oed_db.insert_entry(                           # no resolved pronunciation at all -- excluded
        conn, volume_id=volume_id, headword="unresolved", homograph_number=None,
        part_of_speech="n", etymology=None, entry_type="main", parent_entry_id=None,
        page_number=1, raw_text="raw", schema=schema)
    conn.commit()

    lexicon = oed_db.pronunciation_lexicon(
        conn, {"abandoner", "bay", "fleet", "unresolved", "not-present-at-all"}, schema=schema)

    assert lexicon["abandoner"] == ["ˈbændənə(r)"]   # paren closed
    assert lexicon["bay"] == ["beɪ"]                  # deduped to 1 distinct value
    assert sorted(lexicon["fleet"]) == ["fliːt", "flɛt"]  # genuine conflict preserved, not collapsed
    assert "unresolved" not in lexicon
    assert "not-present-at-all" not in lexicon

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
