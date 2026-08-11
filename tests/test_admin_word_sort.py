"""Admin word list (GET /api/words) alphabetical sort. DB-backed tests run
only when a throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL (else
skipped) -- same convention as test_browse_api.py/test_suggest_word.py,
including the main.SCHEMA monkeypatch pattern for exercising real registered
routes against a disposable schema."""

from __future__ import annotations

import os

import pytest

from concordance import db
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
        for lemma in ("Zoology", "aardvark", "middle"):
            cur.execute(
                f"INSERT INTO {name}.word (lemma, definition, active) VALUES (%s, 'a definition', true)",
                (lemma,),
            )
    conn.commit()
    conn.close()


@pg
def test_admin_word_list_alphabetical_sort_is_case_insensitive():
    """This DB's collation (C.UTF-8) sorts every uppercase letter before
    every lowercase one -- ORDER BY on the raw (case-preserving) lemma
    column put a capitalized word like "Hellenomania" first in an A-Z
    listing, ahead of every ordinary lowercase word (this is what the admin
    curation view -- /api/words, sort=lemma -- looked like before this fix).
    lemma_lc (already the unique-constraint column) sorts case-insensitively
    and must be used instead."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_admin_word_case_sort"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/words", params={"sort": "lemma", "dir": "asc"})
        assert res.status_code == 200, res.text
        assert [w["lemma"] for w in res.json()["items"]] == ["aardvark", "middle", "Zoology"]
    finally:
        main.SCHEMA = old_schema


@pg
def test_admin_word_search_ranks_by_relevance_not_the_active_sort_column():
    """Searching used to leave results sorted by whatever column was already
    selected (difficulty, by default) rather than by match quality -- found
    live: searching an exact lemma ("Hellenomania") returned hundreds of
    loosely-related trigram matches with the actual word 7 pages deep,
    indistinguishable from "not found." A search must surface the best match
    first regardless of the active sort column (matching browse.py's public
    search, which already worked this way) -- and clearing the search
    resumes that sort column, proven here by the no-`q` request below."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_admin_word_search_relevance"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES ('adminuser', %s, true)",
            (auth.hash_password("password123"),),
        )
        # "hellenomania" is the best textual match for the search below but
        # the WORST (highest) difficulty; "hell" is a weak textual match but
        # the BEST (lowest) difficulty -- so the two orderings disagree,
        # proving which one actually won.
        for lemma, difficulty in (("hellenomania", 90.0), ("hell", 1.0), ("hello", 50.0)):
            cur.execute(
                f"INSERT INTO {schema}.word (lemma, definition, active) VALUES (%s, 'a definition', true) RETURNING id",
                (lemma,),
            )
            wid = cur.fetchone()[0]
            cur.execute(
                f"INSERT INTO {schema}.word_difficulty (word_id, difficulty) VALUES (%s, %s)",
                (wid, difficulty),
            )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        # No search: default difficulty-ascending sort puts "hell" first.
        res = client.get("/api/words", params={"sort": "difficulty", "dir": "asc"})
        assert res.json()["items"][0]["lemma"] == "hell"

        # Searching the exact word overrides that sort -- best match first.
        res = client.get("/api/words", params={"q": "hellenomania", "sort": "difficulty", "dir": "asc"})
        assert res.status_code == 200, res.text
        items = res.json()["items"]
        assert items[0]["lemma"] == "hellenomania"

        # Clearing the search resumes the chosen sort column.
        res = client.get("/api/words", params={"sort": "difficulty", "dir": "asc"})
        assert res.json()["items"][0]["lemma"] == "hell"
    finally:
        main.SCHEMA = old_schema
