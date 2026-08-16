"""Admin OED entries API (GET /api/admin/oed/entries[/:id]) -- specifically
the concordance_match filter/field this session added; the rest of the
router has no prior coverage to extend. DB-backed, same CONCORDANCE_TEST_DB_URL
convention as test_admin_word_sort.py, including its main.SCHEMA-monkeypatch
pattern -- extended here to also monkeypatch webapp.backend.oed.OED_SCHEMA,
since that module deliberately hardcodes the oed schema name at import time
(see its own docstring) rather than reading a per-request schema like
browse.py/quiz.py do. Patching the module attribute directly is still safe
for a test: production code never overrides it, this only redirects where
THIS test's requests land."""

from __future__ import annotations

import os

import pytest

from concordance import db
from concordance.oed import db as oed_db
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


def _setup(main_schema: str, entry_oed_schema: str):
    from starlette.testclient import TestClient

    from webapp.backend import main, oed as oed_router

    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {main_schema} CASCADE")
        cur.execute(f"DROP SCHEMA IF EXISTS {entry_oed_schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, main_schema)
    oed_db.apply_schema(conn, entry_oed_schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {main_schema}.users (username, password_hash, is_admin) VALUES ('oedadmin', %s, true)",
            (auth.hash_password("password123"),),
        )
        cur.execute(f"INSERT INTO {main_schema}.word (lemma, definition, active) VALUES ('keeper', 'def', true) "
                    "RETURNING id")
        accepted_word_id = cur.fetchone()[0]
    conn.commit()

    volume_id = oed_db.upsert_volume(conn, file_name="test.pdf", file_hash_="apitest",
                                      volume_label="Test Volume", page_count=1, schema=entry_oed_schema)
    accepted_entry_id = oed_db.insert_entry(
        conn, volume_id=volume_id, headword="keeper", homograph_number=None,
        part_of_speech=None, etymology=None, entry_type="main", parent_entry_id=None,
        page_number=1, raw_text="raw", schema=entry_oed_schema)
    unique_entry_id = oed_db.insert_entry(
        conn, volume_id=volume_id, headword="zyzzyva", homograph_number=None,
        part_of_speech=None, etymology=None, entry_type="main", parent_entry_id=None,
        page_number=2, raw_text="raw", schema=entry_oed_schema)
    conn.commit()

    stats = oed_db.compute_concordance_match(conn, entry_oed_schema, main_schema)
    assert stats == {"entries": 0, "accepted": 0, "pruned": 0, "rejected": 0, "unique": 0}  # neither is lemma=true yet
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {entry_oed_schema}.entry SET lemma = true")
    conn.commit()
    stats = oed_db.compute_concordance_match(conn, entry_oed_schema, main_schema)
    assert stats == {"entries": 2, "accepted": 1, "pruned": 0, "rejected": 0, "unique": 1}
    conn.close()

    old_main_schema = main.SCHEMA
    old_oed_schema = oed_router.OED_SCHEMA
    main.SCHEMA = main_schema
    oed_router.OED_SCHEMA = entry_oed_schema
    client = TestClient(main.app, base_url="https://testserver")
    client.post("/api/auth/login", json={"username": "oedadmin", "password": "password123"})

    def restore():
        main.SCHEMA = old_main_schema
        oed_router.OED_SCHEMA = old_oed_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {main_schema} CASCADE")
            cur.execute(f"DROP SCHEMA IF EXISTS {entry_oed_schema} CASCADE")
        cleanup.commit()
        cleanup.close()

    return client, restore, accepted_entry_id, unique_entry_id, accepted_word_id


@pg
def test_oed_entries_filters_by_concordance_match():
    client, restore, accepted_entry_id, unique_entry_id, _word_id = _setup(
        "cc_test_oedapi_main", "oed_test_api_filter")
    try:
        all_items = client.get("/api/admin/oed/entries").json()["items"]
        matches = {i["id"]: i["concordance_match"] for i in all_items}
        assert matches[accepted_entry_id] == "accepted"
        assert matches[unique_entry_id] == "unique"

        accepted_only = client.get("/api/admin/oed/entries", params={"concordance_match": "accepted"}).json()
        assert [i["id"] for i in accepted_only["items"]] == [accepted_entry_id]

        unique_only = client.get("/api/admin/oed/entries", params={"concordance_match": "unique"}).json()
        assert [i["id"] for i in unique_only["items"]] == [unique_entry_id]
    finally:
        restore()


@pg
def test_oed_entry_detail_links_matched_concordance_word():
    client, restore, accepted_entry_id, unique_entry_id, word_id = _setup(
        "cc_test_oedapi_detail_main", "oed_test_api_detail")
    try:
        accepted_detail = client.get(f"/api/admin/oed/entries/{accepted_entry_id}").json()
        assert accepted_detail["concordance_match"] == "accepted"
        assert accepted_detail["concordance_word_id"] == word_id

        unique_detail = client.get(f"/api/admin/oed/entries/{unique_entry_id}").json()
        assert unique_detail["concordance_match"] == "unique"
        assert unique_detail["concordance_word_id"] is None
    finally:
        restore()
