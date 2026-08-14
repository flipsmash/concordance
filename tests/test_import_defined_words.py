"""concordance import-defined: importing genuinely-new terms from the legacy
vocab.defined table into word as book-less words. vocab.defined is a real,
stable reference table in the same dev DB (not something to fake), so these
tests query it live for fixtures rather than hardcoding term names."""

from __future__ import annotations

import os

import pytest

from concordance import db

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


def _defined_available(url):
    if not (url and _connectable(url)):
        return False
    import psycopg
    conn = psycopg.connect(url, connect_timeout=3)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('vocab.defined')")
        available = cur.fetchone()[0] is not None
    conn.close()
    return available


has_defined = pytest.mark.skipif(not _defined_available(_URL),
                                 reason="vocab.defined table not present in this DB")


class _FakeCursor:
    def __init__(self):
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, *args):
        assert "to_regclass" in query  # this fake only models the existence check
        self._result = (None,)

    def fetchone(self):
        return self._result


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


def test_import_defined_words_reports_unavailable_when_table_missing():
    stats = db.import_defined_words(_FakeConn(), "concordance")
    assert stats == {"available": False, "candidates": 0, "imported": 0, "skipped_conflict": 0}


@pg
@has_defined
def test_import_defined_words_excludes_existing_and_rejected_and_imports_new():
    schema = "cc_test_import_defined"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    with conn.cursor() as cur:
        cur.execute(
            """SELECT lower(term) FROM vocab.defined
               WHERE phrase IS DISTINCT FROM 1 AND position(' ' in term) = 0
                 AND COALESCE(bad,0) != 1
               GROUP BY lower(term) LIMIT 3""")
        terms = [r[0] for r in cur.fetchall()]
    assert len(terms) == 3, "need at least 3 real non-phrase, non-bad vocab.defined terms to test against"
    already_in_word, already_rejected, genuinely_new = terms

    with conn.cursor() as cur:
        # Seed term #1 as already present in `word`.
        cur.execute(f"INSERT INTO {schema}.word (lemma, definition) VALUES (%s, 'preexisting definition')",
                    (already_in_word,))
        # Seed term #2 as previously rejected (needs a book row for the FK).
        cur.execute(f"INSERT INTO {schema}.book (title) VALUES ('Some Book') RETURNING id")
        book_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.rejected_word (book_id, lemma, reason) VALUES (%s, %s, 'not_interesting')",
                    (book_id, already_rejected))
    conn.commit()

    stats = db.import_defined_words(conn, schema, limit=0)
    assert stats["available"] is True

    with conn.cursor() as cur:
        # #1 untouched -- still has the seeded definition, not vocab.defined's.
        cur.execute(f"SELECT definition FROM {schema}.word WHERE lemma_lc = %s", (already_in_word,))
        assert cur.fetchone()[0] == "preexisting definition"

        # #2 never imported -- previously rejected in some book.
        cur.execute(f"SELECT count(*) FROM {schema}.word WHERE lemma_lc = %s", (already_rejected,))
        assert cur.fetchone()[0] == 0

        # #3 imported with a real definition and no word_book row (book-less).
        cur.execute(f"SELECT id, definition, part_of_speech, vocab1_import, vocab1_import_at "
                    f"FROM {schema}.word WHERE lemma_lc = %s", (genuinely_new,))
        row = cur.fetchone()
        assert row is not None
        word_id, definition, pos, vocab1_import, vocab1_import_at = row
        assert definition
        assert pos == pos.lower()  # normalize_pos always lowercases
        assert vocab1_import is True
        assert vocab1_import_at is not None

        cur.execute(f"SELECT count(*) FROM {schema}.word_book WHERE word_id = %s", (word_id,))
        assert cur.fetchone()[0] == 0

        # #1 (pre-seeded, not touched by the import) has no provenance flag.
        cur.execute(f"SELECT vocab1_import FROM {schema}.word WHERE lemma_lc = %s", (already_in_word,))
        assert cur.fetchone()[0] is False

        # No imported lemma is a phrase (contains a space).
        cur.execute(f"SELECT lemma FROM {schema}.word WHERE lemma_lc = %s", (genuinely_new,))
        assert " " not in cur.fetchone()[0]

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
@has_defined
def test_import_defined_words_is_idempotent():
    # A full, unlimited run (~seconds against the real vocab.defined) so the
    # second pass has nothing left to find -- a --limit run naturally
    # advances to the *next* batch of candidates on a repeat call (the
    # previous batch is now excluded by the NOT EXISTS filters), which is
    # correct resumability, not something a small-limit idempotency check
    # could tell apart from a bug.
    schema = "cc_test_import_defined_idempotent"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    stats1 = db.import_defined_words(conn, schema)
    assert stats1["available"] is True
    assert stats1["imported"] > 0

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.word")
        count_after_first = cur.fetchone()[0]

    stats2 = db.import_defined_words(conn, schema)
    assert stats2["candidates"] == 0  # every vocab.defined candidate now already in word
    assert stats2["imported"] == 0

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {schema}.word")
        assert cur.fetchone()[0] == count_after_first

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
