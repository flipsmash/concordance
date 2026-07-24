"""Word-set + flashcard API. DB-backed tests run only when a throwaway
Postgres is provided via CONCORDANCE_TEST_DB_URL (else skipped) -- same
convention as test_quiz_api.py/test_auth.py, including the main.SCHEMA
monkeypatch pattern for exercising real registered routes against a
disposable schema."""

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


def _seed_words(conn, schema: str, n: int = 6) -> list[int]:
    ids = []
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                f"""INSERT INTO {schema}.word (lemma, definition, part_of_speech, active)
                    VALUES (%s, %s, 'noun', true) RETURNING id""",
                (f"setword{i}", f"definition {i}"),
            )
            ids.append(cur.fetchone()[0])
    conn.commit()
    return ids


@pg
def test_word_sets_full_round_trip():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_word_sets_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    word_ids = _seed_words(conn, schema, n=6)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES ('setuser', %s, false)",
            (auth.hash_password("password123"),),
        )
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES ('setuser2', %s, false)",
            (auth.hash_password("password456"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")

        # Anonymous is refused everywhere.
        assert client.get("/api/sets").status_code == 401
        assert client.post("/api/sets", json={"name": "My set"}).status_code == 401

        client.post("/api/auth/login", json={"username": "setuser", "password": "password123"})

        # Create.
        res = client.post("/api/sets", json={"name": "GRE words"})
        assert res.status_code == 201, res.text
        created = res.json()
        set_id = created["id"]
        assert created["word_count"] == 0
        assert created["mastered_count"] == 0

        # Duplicate name for the same user conflicts.
        assert client.post("/api/sets", json={"name": "GRE words"}).status_code == 409

        # List shows it.
        listing = client.get("/api/sets").json()
        assert any(s["id"] == set_id and s["name"] == "GRE words" for s in listing)

        # Add words (bulk, as Browse's multi-select would).
        res = client.post(f"/api/sets/{set_id}/words", json={"word_ids": word_ids[:4]})
        assert res.status_code == 204, res.text

        # Adding again is a harmless no-op (ON CONFLICT DO NOTHING).
        assert client.post(f"/api/sets/{set_id}/words", json={"word_ids": word_ids[:4]}).status_code == 204

        detail = client.get(f"/api/sets/{set_id}").json()
        assert detail["name"] == "GRE words"
        assert len(detail["items"]) == 4
        assert all(item["mastered"] is False for item in detail["items"])
        assert {item["word_id"] for item in detail["items"]} == set(word_ids[:4])

        # Flashcard deck has all 4, none mastered yet.
        deck = client.get(f"/api/sets/{set_id}/flashcards").json()
        assert {item["word_id"] for item in deck["items"]} == set(word_ids[:4])

        # Mark one mastered.
        mastered_word = word_ids[0]
        res = client.patch(f"/api/sets/{set_id}/words/{mastered_word}", json={"mastered": True})
        assert res.status_code == 204, res.text

        # Flashcard deck now excludes the mastered word.
        deck = client.get(f"/api/sets/{set_id}/flashcards").json()
        deck_ids = {item["word_id"] for item in deck["items"]}
        assert mastered_word not in deck_ids
        assert deck_ids == set(word_ids[1:4])

        # Summary page still shows the mastered word, flagged correctly.
        detail = client.get(f"/api/sets/{set_id}").json()
        by_id = {item["word_id"]: item["mastered"] for item in detail["items"]}
        assert by_id[mastered_word] is True
        assert all(not v for k, v in by_id.items() if k != mastered_word)

        # Un-master it -- reappears in the deck.
        client.patch(f"/api/sets/{set_id}/words/{mastered_word}", json={"mastered": False})
        deck = client.get(f"/api/sets/{set_id}/flashcards").json()
        assert mastered_word in {item["word_id"] for item in deck["items"]}

        # Remove a word from the set entirely.
        res = client.delete(f"/api/sets/{set_id}/words/{word_ids[1]}")
        assert res.status_code == 204
        detail = client.get(f"/api/sets/{set_id}").json()
        assert word_ids[1] not in {item["word_id"] for item in detail["items"]}

        # Rename.
        res = client.patch(f"/api/sets/{set_id}", json={"name": "GRE vocab"})
        assert res.status_code == 200, res.text
        assert res.json()["name"] == "GRE vocab"

        client.post("/api/auth/logout")

        # A different user can't see or modify someone else's set.
        client.post("/api/auth/login", json={"username": "setuser2", "password": "password456"})
        assert client.get(f"/api/sets/{set_id}").status_code == 404
        assert client.post(f"/api/sets/{set_id}/words", json={"word_ids": [word_ids[0]]}).status_code == 404
        assert client.delete(f"/api/sets/{set_id}").status_code == 404
        assert client.get("/api/sets").json() == []

        client.post("/api/auth/logout")
        client.post("/api/auth/login", json={"username": "setuser", "password": "password123"})

        # Delete the set.
        assert client.delete(f"/api/sets/{set_id}").status_code == 204
        assert client.get("/api/sets").json() == []
        assert client.get(f"/api/sets/{set_id}").status_code == 404
    finally:
        main.SCHEMA = old_schema
