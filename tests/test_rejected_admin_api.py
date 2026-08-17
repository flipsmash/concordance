"""GET /api/rejected/reasons and /api/rejected/books -- the admin curation
page's two filter-dropdown sources. DB-backed tests run only when a
throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL (else skipped),
same convention as test_admin_word_sort.py, including the main.SCHEMA
monkeypatch pattern for exercising real registered routes against a
disposable schema."""

from __future__ import annotations

import os

import pytest

from concordance import db
from concordance.model import RejectReason
from webapp.backend import auth

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


def _fresh_schema(name: str):
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE")
    conn.commit()
    db.apply_schema(conn, name)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {name}.users (username, password_hash, is_admin) VALUES ('adminuser', %s, true)",
            (auth.hash_password("password123"),),
        )
    conn.commit()
    conn.close()
    return conn


@pg
def test_rejected_reasons_returns_the_fixed_enum_excluding_frequency_floor():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_rejected_reasons"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/rejected/reasons")
        assert res.status_code == 200, res.text
        reasons = res.json()
        # Every RejectReason value except FREQUENCY_FLOOR (sync_book_results
        # no longer persists it -- deterministic per-lemma, never book-
        # specific, so it would always return zero rows now).
        assert "frequency_floor" not in reasons
        assert reasons == sorted(r.value for r in RejectReason if r is not RejectReason.FREQUENCY_FLOOR)
        # No DB query behind this at all -- an empty rejected_word table
        # still returns the full fixed list.
        assert len(reasons) == len(RejectReason) - 1
    finally:
        main.SCHEMA = old_schema


@pg
def test_rejected_books_only_lists_books_with_at_least_one_rejected_word():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_rejected_books"
    _fresh_schema(schema)
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.book (title, author) VALUES ('Has Rejects', 'Author A') RETURNING id")
        has_rejects_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.book (title, author) VALUES ('No Rejects', 'Author B') RETURNING id")
        cur.execute(
            f"""INSERT INTO {schema}.rejected_word (book_id, lemma, reason)
                VALUES (%s, 'somelemma', 'not_interesting')""",
            (has_rejects_id,),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/rejected/books")
        assert res.status_code == 200, res.text
        # "No Rejects" has a book row but no rejected_word row -- must not
        # appear (the old DISTINCT-through-rejected_word query and this
        # EXISTS-based rewrite must agree on this either way, but it's the
        # rewrite's own correctness this test is really pinning down).
        assert res.json() == ["Has Rejects"]
    finally:
        main.SCHEMA = old_schema
