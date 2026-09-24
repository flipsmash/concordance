"""Home page summary API -- word of the day. DB-backed; runs only with a
throwaway Postgres in CONCORDANCE_TEST_DB_URL (else skipped), using the same
main.SCHEMA-monkeypatch convention as test_browse_api.py."""

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


@pg
def test_word_of_the_day_is_eligible_and_stable():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_home_wotd"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('homeuser', %s)",
                    (auth.hash_password("password123"),))
        # (lemma, difficulty, quizzable, definition, flag) -- only 'eligible' qualifies
        for lemma, diff, quizzable, definition, flag in [
            ("eligible", 88, True, "A fine hard word.", None),
            ("tooeasy", 40, True, "An easy word.", None),
            ("notquizzable", 90, False, "Obsolete form of foo.", None),
            ("undefined", 90, True, "", None),
            ("flagged", 90, True, "A flagged word.", "archaic_review"),
        ]:
            cur.execute(f"""INSERT INTO {schema}.word (lemma, definition, part_of_speech, active,
                                                       variant_flag_reason)
                            VALUES (%s, %s, 'noun', true, %s) RETURNING id""", (lemma, definition, flag))
            wid = cur.fetchone()[0]
            cur.execute(f"""INSERT INTO {schema}.word_difficulty (word_id, difficulty, quizzable)
                            VALUES (%s, %s, %s)""", (wid, diff, quizzable))
    conn.commit()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "homeuser", "password": "password123"})
        first = client.get("/api/home/summary").json()["word_of_the_day"]
        again = client.get("/api/home/summary").json()["word_of_the_day"]
        assert first["lemma"] == "eligible" and first["definition"] == "A fine hard word."
        assert first == again                                   # stable within the day
    finally:
        main.SCHEMA = old_schema
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
        conn.close()
