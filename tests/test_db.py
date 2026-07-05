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
    conn.close()
