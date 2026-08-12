"""Admin hand-edit of a word's definition (PATCH /api/words/{id}/definition).
DB-backed tests run only when a throwaway Postgres is provided via
CONCORDANCE_TEST_DB_URL (else skipped) -- same convention as
test_word_sets.py/test_suggest_word.py, including the main.SCHEMA
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


def _fresh_schema(name: str) -> int:
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
        cur.execute(
            f"INSERT INTO {name}.word (lemma, definition, active) VALUES ('quixotic', 'original definition', true) RETURNING id"
        )
        word_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return word_id


@pg
def test_edit_definition_notes_the_edit_and_keeps_the_former_value():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_basic"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.patch(f"/api/words/{word_id}/definition", json={"definition": "a revised definition"})
        assert res.status_code == 200, res.text
        assert res.json() == {"id": word_id, "lemma": "quixotic", "definition": "a revised definition"}

        conn = db.connect(_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT definition, previous_definition, definition_edited_by, definition_edited_at
                    FROM {schema}.word WHERE id = %s""",
                (word_id,),
            )
            definition, previous, edited_by, edited_at = cur.fetchone()
        conn.close()

        assert definition == "a revised definition"
        assert previous == "original definition"
        assert edited_by == "adminuser"
        assert edited_at is not None
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_frontend_response_has_no_edit_marker():
    """Per the feature request: the front end gets no indication a definition
    was manually edited -- the word detail response carries only the plain
    definition field, nothing about previous_definition/edited_by/edited_at."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_no_marker"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})
        client.patch(f"/api/words/{word_id}/definition", json={"definition": "a revised definition"})

        detail = client.get(f"/api/words/{word_id}").json()
        assert detail["definition"] == "a revised definition"
        assert "previous_definition" not in detail
        assert "definition_edited_by" not in detail
        assert "definition_edited_at" not in detail
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_resubmitting_same_value_is_a_no_op():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_noop"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.patch(f"/api/words/{word_id}/definition", json={"definition": "original definition"})
        assert res.status_code == 200, res.text

        conn = db.connect(_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT previous_definition, definition_edited_by, definition_edited_at
                    FROM {schema}.word WHERE id = %s""",
                (word_id,),
            )
            previous, edited_by, edited_at = cur.fetchone()
        conn.close()

        assert previous is None
        assert edited_by is None
        assert edited_at is None
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_second_edit_overwrites_previous_with_the_immediately_prior_value():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_second"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        client.patch(f"/api/words/{word_id}/definition", json={"definition": "first revision"})
        client.patch(f"/api/words/{word_id}/definition", json={"definition": "second revision"})

        conn = db.connect(_URL)
        with conn.cursor() as cur:
            cur.execute(f"SELECT definition, previous_definition FROM {schema}.word WHERE id = %s", (word_id,))
            definition, previous = cur.fetchone()
        conn.close()

        assert definition == "second revision"
        assert previous == "first revision"
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_clears_stale_definition_links():
    """word_definition_link rows are computed against the OLD text
    (concordance link-definitions). LinkedDefinition matches a link's stale
    `surface` against whatever text is live NOW, so leaving the row in place
    after an edit risks hyperlinking a coincidental substring of the NEW
    prose to the wrong word -- not just a missing link, a wrong one. The
    edit must delete this word's own outgoing links; they get recomputed
    fresh on the next link-definitions run, same as a newly-added word."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_links"
    word_id = _fresh_schema(schema)

    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.word (lemma, definition, active) VALUES ('errant', 'wandering', true) RETURNING id"
        )
        target_id = cur.fetchone()[0]
        cur.execute(
            f"""INSERT INTO {schema}.word_definition_link (source_word_id, target_word_id, surface)
                VALUES (%s, %s, 'original')""",
            (word_id, target_id),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.patch(f"/api/words/{word_id}/definition", json={"definition": "a revised definition"})
        assert res.status_code == 200, res.text

        conn = db.connect(_URL)
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {schema}.word_definition_link WHERE source_word_id = %s", (word_id,))
            remaining = cur.fetchone()[0]
        conn.close()
        assert remaining == 0
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_requires_admin():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_auth"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        res = client.patch(f"/api/words/{word_id}/definition", json={"definition": "hijacked"})
        assert res.status_code == 403
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_404_for_missing_word():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_404"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})
        res = client.patch("/api/words/999999999/definition", json={"definition": "x"})
        assert res.status_code == 404
    finally:
        main.SCHEMA = old_schema


@pg
def test_edit_definition_rejects_empty():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_empty"
    word_id = _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})
        res = client.patch(f"/api/words/{word_id}/definition", json={"definition": ""})
        assert res.status_code == 422
    finally:
        main.SCHEMA = old_schema


@pg
def test_word_detail_exposes_quiz_definition_and_source():
    """Regression: an admin flagged "codpieced" as wrongly quizzable because
    its plain `definition` still said "codpiece" -- quizzable is actually
    judged against `quiz_definition` (what a quiz really shows), which had
    already been safely rewritten, but nothing in the word detail response
    exposed that field to let an admin see the two disagree. Word detail is
    require_viewer (not admin-gated) at the API layer -- the frontend hides
    this field for non-admins, but the API itself must always return it."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_def_edit_quiz_definition"
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
        cur.execute(
            f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, quiz_def_source, active)
                VALUES ('codpieced', 'Wearing, or fitted with, a codpiece.',
                        'Wearing a covering for the genitals', 'rewritten', true)
                RETURNING id""",
        )
        word_id = cur.fetchone()[0]
        cur.execute(
            f"""INSERT INTO {schema}.word_difficulty (word_id, quizzable) VALUES (%s, true)""",
            (word_id,),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        detail = client.get(f"/api/words/{word_id}").json()
        assert detail["definition"] == "Wearing, or fitted with, a codpiece."
        assert detail["quiz_definition"] == "Wearing a covering for the genitals"
        assert detail["quiz_def_source"] == "rewritten"
        assert detail["quizzable"] is True
    finally:
        main.SCHEMA = old_schema
