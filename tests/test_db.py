"""Postgres sync. Pure helpers run always; the round-trip test runs only when a
throwaway DB is provided via CONCORDANCE_TEST_DB_URL (else skipped)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from concordance import db
from concordance.master import MASTER_COLUMNS


# --- pure helpers (no database) -------------------------------------------

def test_synonyms_and_books_split():
    assert db._synonyms("a; b ;c") == ["a", "b", "c"]
    assert db._synonyms("") == []
    assert db._books("BookA; BookB") == ["BookA", "BookB"]


def test_safe_schema_rejects_injection():
    assert db._safe_schema("concordance") == "concordance"
    for bad in ["public; drop table x", "a-b", "1abc", "a b", ""]:
        with pytest.raises(ValueError):
            db._safe_schema(bad)


def test_read_master_rows_keeps_master_columns(tmp_path):
    p = tmp_path / "master_vocab.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        row = {c: "" for c in MASTER_COLUMNS}
        row.update(word="cangue", date_added="2026-07-05", source_book="BookA; BookB")
        w.writerow(row)
    rows = db._read_master_rows(p)
    assert rows[0]["source_book"] == "BookA; BookB"   # NOT dropped
    assert rows[0]["date_added"] == "2026-07-05"


# --- round trip (needs a real, disposable Postgres) -----------------------

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


@pg
def test_ingest_never_clobbers_a_definition_and_flags_undefined(tmp_path):
    from concordance.model import Candidate

    schema = "cc_test2"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    # Book 1: "cangue" is successfully defined.
    defined = Candidate(lemma="cangue", pos="NOUN")
    defined.definition = "a wooden collar"
    defined.definition_source = "Local Wiktionary (DB)"
    db.sync_book_results(conn, "Book One", kept=[defined], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='cangue'")
        row = cur.fetchone()
        assert row == ("a wooden collar", False)

    # Book 2: same word recurs but this time enrichment fails (blank definition)
    # — the existing definition must survive, not be clobbered to blank.
    undefined_repeat = Candidate(lemma="cangue", pos="NOUN")
    db.sync_book_results(conn, "Book Two", kept=[undefined_repeat], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='cangue'")
        row = cur.fetchone()
        assert row == ("a wooden collar", False)   # not clobbered, not flagged (still defined)

    # A brand-new word that comes in with no definition at all must be flagged,
    # and stay flagged even if refill later fills it in (sticky by design).
    never_defined = Candidate(lemma="fuligin", pos="NOUN")
    db.sync_book_results(conn, "Book One", kept=[never_defined], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select definition, flagged_undefined, flagged_undefined_at "
                    f"from {schema}.word where lemma='fuligin'")
        d, flagged, flagged_at = cur.fetchone()
        assert d == "" and flagged is True and flagged_at is not None

    with conn.cursor() as cur:
        cur.execute(f"update {schema}.word set definition='a fictional black pigment' "
                    f"where lemma='fuligin'")
        cur.execute(f"select definition, flagged_undefined from {schema}.word where lemma='fuligin'")
        assert cur.fetchone() == ("a fictional black pigment", True)   # flag persists

        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_ingest_invalidates_stale_definition_dependents_on_change():
    """The "changeful" bug: a word's quiz_definition/categories/embedding get
    computed once, then the same lemma resolves to a DIFFERENT dictionary
    sense on a later book's ingest -- definition changes, but nothing used to
    tell the downstream only-missing-gated artifacts to recompute, so they
    silently kept describing the old text. sync_book_results/sync_master
    should now clear them whenever an upsert actually changes an existing
    definition (never on a first-time fill, never when it's unchanged)."""
    from pgvector.psycopg import register_vector

    from concordance.model import Candidate

    schema = "cc_test_definition_invalidation"
    conn = db.connect(_URL)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    first = Candidate(lemma="changeful", pos="ADJ")
    first.definition = "very susceptible to change; changing frequently"
    db.sync_book_results(conn, "Book One", kept=[first], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {schema}.word WHERE lemma='changeful'")
        word_id = cur.fetchone()[0]
        # Simulate a maintenance pass having already run on the ORIGINAL definition.
        cur.execute(f"UPDATE {schema}.word SET quiz_definition='stale clue', quiz_def_source='redacted', "
                    f"ipa='tʃeɪndʒfʊl' WHERE id=%s", (word_id,))
        cur.execute(f"INSERT INTO {schema}.category (taxonomy, code, name) VALUES ('usas','A1','test cat') "
                    f"ON CONFLICT (taxonomy, code) DO NOTHING")
        cur.execute(f"SELECT id FROM {schema}.category WHERE taxonomy='usas' AND code='A1'")
        cat_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.word_category (word_id, category_id, is_primary) VALUES (%s,%s,true)",
                    (word_id, cat_id))
        cur.execute(
            f"""INSERT INTO {schema}.word_embedding (word_id, definition_vector, definition_model, fasttext_vector, fasttext_model)
                VALUES (%s, %s, 'test-def-model', %s, 'test-ft-model')""",
            (word_id, [0.1] * 384, [0.2] * 300))
    conn.commit()

    # Re-ingesting Book One again with the SAME definition must not invalidate anything.
    same = Candidate(lemma="changeful", pos="ADJ")
    same.definition = "very susceptible to change; changing frequently"
    db.sync_book_results(conn, "Book One", kept=[same], rejected=[], schema=schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT quiz_definition FROM {schema}.word WHERE id=%s", (word_id,))
        assert cur.fetchone()[0] == "stale clue"

    # Book Two resolves "changeful" to a different, shorter sense -- this is
    # the actual trigger: definition changes on an already-enriched word.
    changed = Candidate(lemma="changeful", pos="ADJ")
    changed.definition = "Changing frequently"
    db.sync_book_results(conn, "Book Two", kept=[changed], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"SELECT definition, quiz_definition, quiz_def_source, ipa FROM {schema}.word WHERE id=%s",
                    (word_id,))
        defn, quiz_def, quiz_src, ipa = cur.fetchone()
        assert defn == "Changing frequently"
        assert quiz_def is None and quiz_src is None            # invalidated
        assert ipa == "tʃeɪndʒfʊl"                               # untouched -- not definition-derived

        cur.execute(f"SELECT count(*) FROM {schema}.word_category WHERE word_id=%s", (word_id,))
        assert cur.fetchone()[0] == 0                            # invalidated

        cur.execute(f"SELECT definition_vector, fasttext_vector FROM {schema}.word_embedding WHERE word_id=%s",
                    (word_id,))
        def_vec, ft_vec = cur.fetchone()
        assert def_vec is None                                   # invalidated
        assert ft_vec is not None                                # untouched -- lemma-derived, not definition-derived

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_sync_roundtrip_and_idempotent(tmp_path):
    schema = "cc_test"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    p = tmp_path / "master_vocab.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_COLUMNS)
        w.writeheader()
        for word, books in [("cangue", "BookA; BookB"), ("fuligin", "BookA")]:
            r = {c: "" for c in MASTER_COLUMNS}
            r.update(word=word, definition=f"def {word}", synonyms="x; y",
                     date_added="2026-07-05", source_book=books)
            w.writerow(r)

    s1 = db.sync_master(p, conn, schema)
    assert s1 == {"words": 2, "books": 2, "links": 3, "rows": 2}
    s2 = db.sync_master(p, conn, schema)          # idempotent
    assert s2["words"] == 2 and s2["links"] == 0  # no new links second time

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.word"); assert cur.fetchone()[0] == 2
        cur.execute(f"select synonyms from {schema}.word where lemma='cangue'")
        assert cur.fetchone()[0] == ["x", "y"]
        cur.execute(f"""select count(*) from {schema}.word_book wb
                        join {schema}.word w on w.id=wb.word_id where w.lemma='cangue'""")
        assert cur.fetchone()[0] == 2             # linked to both books
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()


@pg
def test_junk_pos_rejection_casts_out_an_already_active_word():
    # Regression: a lemma accepted in an earlier book (e.g. its first-ever
    # dictionary lookup landed on a non-junk sense) must actually be
    # un-accepted the moment a LATER book's lookup resolves it to a proper
    # noun/symbol -- pipeline.py's post-enrichment junk-POS check now runs on
    # every re-encounter, and sync_book_results is what has to act on it.
    from concordance.model import Candidate, RejectReason

    schema = "cc_test_castout"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)

    accepted = Candidate(lemma="linnaea", pos="NOUN")
    accepted.definition = "a genus of plants"
    db.sync_book_results(conn, "Book One", kept=[accepted], rejected=[], schema=schema)

    with conn.cursor() as cur:
        cur.execute(f"select active from {schema}.word where lemma='linnaea'")
        assert cur.fetchone() == (True,)

    later_lookup = Candidate(lemma="linnaea", pos="NOUN")
    later_lookup.reject_reason = RejectReason.PROPER_NOUN
    later_lookup.interesting_reason = "dictionary lookup resolved this as 'proper noun' — cast out"
    stats = db.sync_book_results(conn, "Book Two", kept=[], rejected=[later_lookup], schema=schema)

    assert stats["cast_out"] == 1
    with conn.cursor() as cur:
        cur.execute(f"select active from {schema}.word where lemma='linnaea'")
        assert cur.fetchone() == (False,)
        cur.execute(f"""select reason from {schema}.rejected_word r
                        join {schema}.book b on b.id=r.book_id
                        where r.lemma='linnaea' and b.title='Book Two'""")
        assert cur.fetchone() == ("proper_noun",)

    # A junk-POS rejection for a lemma with no pre-existing word row is a
    # harmless no-op cast-out (0 rows affected), not an error.
    never_seen = Candidate(lemma="acac", pos="NOUN")
    never_seen.reject_reason = RejectReason.PROPER_NOUN
    stats2 = db.sync_book_results(conn, "Book Two", kept=[], rejected=[never_seen], schema=schema)
    assert stats2["cast_out"] == 0

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
