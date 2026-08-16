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
def test_pronunciation_lexicon_degrades_gracefully_when_oed_schema_absent():
    # compute_ipa (via resolve_pronunciation) now calls this unconditionally
    # from the regular `ipa`/`maintain` path, not just the standalone
    # oed-ipa command someone would only run after already setting up
    # oed-ingest -- must not hard-error on a DB that never touched OED.
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS oed_test_absent CASCADE")
    conn.commit()
    assert oed_db.pronunciation_lexicon(conn, {"whatever"}, schema="oed_test_absent") == {}
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


@pg
def test_compute_lemma_flags_backfills_only_uncomputed_entries():
    schema = "oed_test_lemma"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    oed_db.apply_schema(conn, schema)

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="def456",
                                      volume_label="Test Volume", page_count=1, schema=schema)

    def _entry(headword, pos):
        return oed_db.insert_entry(
            conn, volume_id=volume_id, headword=headword, homograph_number=None,
            part_of_speech=pos, etymology=None, entry_type="main", parent_entry_id=None,
            page_number=1, raw_text="raw", schema=schema)

    lemma_id = _entry("abandon", "v")
    inflected_id = _entry("abandoned", "ppl. a")
    conn.commit()

    stats = oed_db.compute_lemma_flags(conn, schema)
    assert stats == {"entries": 2, "lemma": 1, "not_lemma": 1}

    with conn.cursor() as cur:
        cur.execute(f"SELECT lemma, lemma_computed_at FROM {schema}.entry WHERE id = %s", (lemma_id,))
        lemma, computed_at = cur.fetchone()
        assert lemma is True and computed_at is not None

        cur.execute(f"SELECT lemma FROM {schema}.entry WHERE id = %s", (inflected_id,))
        assert cur.fetchone()[0] is False

    # A second run with only_missing=True (the default) touches nothing new.
    stats2 = oed_db.compute_lemma_flags(conn, schema)
    assert stats2["entries"] == 0

    # A third, freshly-added entry is picked up incrementally.
    _entry("cats", "sb")
    conn.commit()
    stats3 = oed_db.compute_lemma_flags(conn, schema)
    assert stats3 == {"entries": 1, "lemma": 0, "not_lemma": 1}

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_compute_concordance_match_all_four_states_and_recheck_policy():
    oed_schema = "oed_test_match"
    main_schema = "cc_test_oedmatch_main"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {oed_schema} CASCADE")
        cur.execute(f"DROP SCHEMA IF EXISTS {main_schema} CASCADE")
    conn.commit()
    oed_db.apply_schema(conn, oed_schema)
    db.apply_schema(conn, main_schema)

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="matchtest",
                                      volume_label="Test Volume", page_count=1, schema=oed_schema)

    def _lemma_entry(headword):
        eid = oed_db.insert_entry(
            conn, volume_id=volume_id, headword=headword, homograph_number=None,
            part_of_speech=None, etymology=None, entry_type="main", parent_entry_id=None,
            page_number=1, raw_text="raw", schema=oed_schema)
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {oed_schema}.entry SET lemma = true, lemma_computed_at = now() WHERE id = %s",
                        (eid,))
        return eid

    accepted_id = _lemma_entry("keeper")     # active concordance.word row
    pruned_id = _lemma_entry("archaism")     # concordance.word row, active=false
    rejected_id = _lemma_entry("misprint")   # only in rejected_lemma_index
    unique_id = _lemma_entry("zyzzyva")      # nowhere in concordance
    not_lemma_id = oed_db.insert_entry(      # lemma=false (default) -- must be skipped entirely
        conn, volume_id=volume_id, headword="running", homograph_number=None,
        part_of_speech=None, etymology=None, entry_type="main", parent_entry_id=None,
        page_number=1, raw_text="raw", schema=oed_schema)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {main_schema}.word (lemma, definition, active) VALUES ('keeper', 'def', true)")
        cur.execute(f"INSERT INTO {main_schema}.word (lemma, definition, active) VALUES ('archaism', 'def', false)")
        cur.execute(f"INSERT INTO {main_schema}.book (title) VALUES ('Test Book') RETURNING id")
        book_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {main_schema}.rejected_word (book_id, lemma, reason) VALUES (%s, 'misprint', 'not_a_word')",
                    (book_id,))
    conn.commit()
    db.refresh_rejected_lemma_index(conn, main_schema)

    stats = oed_db.compute_concordance_match(conn, oed_schema, main_schema)
    assert stats == {"entries": 4, "accepted": 1, "pruned": 1, "rejected": 1, "unique": 1}

    with conn.cursor() as cur:
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (accepted_id,))
        assert cur.fetchone()[0] == "accepted"
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (pruned_id,))
        assert cur.fetchone()[0] == "pruned"
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (rejected_id,))
        assert cur.fetchone()[0] == "rejected"
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (unique_id,))
        assert cur.fetchone()[0] == "unique"
        # Never touched -- lemma=false entries aren't in this cross-reference's scope at all.
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (not_lemma_id,))
        assert cur.fetchone()[0] is None

    # Re-run with only_missing=True (the default): settled states
    # (accepted/pruned/rejected) are left alone; only 'unique' is re-checked.
    stats2 = oed_db.compute_concordance_match(conn, oed_schema, main_schema)
    assert stats2["entries"] == 1
    assert stats2["unique"] == 1

    # Concordance gains the word that used to be 'unique' -- next run picks
    # it up and flips it, proving 'unique' really is re-checked, not just
    # re-selected-and-ignored.
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {main_schema}.word (lemma, definition, active) VALUES ('zyzzyva', 'def', true)")
    conn.commit()
    stats3 = oed_db.compute_concordance_match(conn, oed_schema, main_schema)
    assert stats3 == {"entries": 1, "accepted": 1, "pruned": 0, "rejected": 0, "unique": 0}
    with conn.cursor() as cur:
        cur.execute(f"SELECT concordance_match FROM {oed_schema}.entry WHERE id = %s", (unique_id,))
        assert cur.fetchone()[0] == "accepted"

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {oed_schema} CASCADE")
        cur.execute(f"DROP SCHEMA {main_schema} CASCADE")
    conn.commit()
    conn.close()
