"""Admin "suggest a new word" API. DB-backed tests run only when a throwaway
Postgres is provided via CONCORDANCE_TEST_DB_URL (else skipped) -- same
convention as test_word_sets.py/test_quiz_api.py/test_auth.py, including the
main.SCHEMA monkeypatch pattern for exercising real registered routes against
a disposable schema."""

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
    conn.commit()
    conn.close()


@pg
def test_search_existing_word_short_circuits():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_suggest_word_exists"
    _fresh_schema(schema)
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.word (lemma, definition, active) VALUES ('ineffable', 'x', true) RETURNING id")
        existing_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/admin/suggest-word/search", params={"lemma": "ineffable"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["exists"] is True
        assert body["word_id"] == existing_id
        assert body["active"] is True
        assert body["candidates"] == []
    finally:
        main.SCHEMA = old_schema


@pg
def test_search_rejects_phrases():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_suggest_word_phrase"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})
        res = client.get("/api/admin/suggest-word/search", params={"lemma": "not a word"})
        assert res.status_code == 422
    finally:
        main.SCHEMA = old_schema


@pg
def test_search_gathers_candidates_from_every_source_independently(monkeypatch):
    """Each source is called directly (not the stop-at-first-hit enrich()/
    deep_enrich() wrappers), so a hit from EVERY source should show up as its
    own candidate, not collapse to one."""
    from starlette.testclient import TestClient

    from concordance import deepdef, dictionary, localdict, mw
    from webapp.backend import main, suggest_word

    schema = "cc_test_suggest_word_search"
    _fresh_schema(schema)

    monkeypatch.setattr(localdict, "lookup_one",
                         lambda conn, lemma, schema="vocab": [("noun", "local sense one", "", "", False, False),
                                                              ("verb", "local sense two", "", "", False, False)])

    def fake_freedict(cand, session):
        cand.definition = "freedict definition"
        cand.part_of_speech = "noun"
        return True
    monkeypatch.setattr(dictionary, "_from_freedict", fake_freedict)

    def fake_wiktionary(cand, session):
        cand.definition = "wiktionary definition"
        cand.part_of_speech = "noun"
        return True
    monkeypatch.setattr(dictionary, "_from_wiktionary", fake_wiktionary)

    # A real key configured AND a real hit -- proves this source is actually
    # called (not just "would have been skipped anyway"), and that its own
    # card survives independently rather than being swallowed by an earlier
    # tier's stop-at-first-hit behavior (the whole point of NOT reusing
    # resolve.py's cascade for this endpoint).
    monkeypatch.setattr(deepdef, "wordnik_key", lambda: "fake-key")
    monkeypatch.setattr(suggest_word, "_pace_wordnik", lambda: None)  # skip the real 12.5s rate-limit sleep

    def fake_wordnik(cand, session, key):
        cand.definition = "wordnik definition"
        cand.part_of_speech = "noun"
        cand.definition_source = "Wordnik (century)"
        return True
    monkeypatch.setattr(deepdef, "_from_wordnik", fake_wordnik)

    def fake_yourdict(cand, session):
        cand.definition = "yourdictionary definition"
        cand.part_of_speech = "noun"
        return True
    monkeypatch.setattr(deepdef, "_from_yourdictionary", fake_yourdict)

    monkeypatch.setattr(mw, "mw_api_key", lambda: "")  # no key -- tier skipped, not an error

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/admin/suggest-word/search", params={"lemma": "zorbling"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["exists"] is False
        sources = {c["source"]: c for c in body["candidates"]}
        assert "Local Wiktionary" in sources
        assert "Local Wiktionary (sense 2)" in sources
        assert sources["Local Wiktionary"]["definition"] == "local sense one"
        assert sources["Local Wiktionary (sense 2)"]["definition"] == "local sense two"
        assert sources["Free Dictionary API"]["definition"] == "freedict definition"
        assert sources["Wiktionary"]["definition"] == "wiktionary definition"
        assert sources["Wordnik (century)"]["definition"] == "wordnik definition"
        assert sources["yourdictionary.com"]["definition"] == "yourdictionary definition"
        # MW was skipped (no key) -- omitted, not an empty entry.
        assert not any("Merriam" in s for s in sources)
    finally:
        main.SCHEMA = old_schema


@pg
def test_search_flags_mw_foreign_loanword_pos_as_a_warning(monkeypatch):
    """mw.is_foreign_pos checks MW's RAW '<Language> noun' string -- a signal
    normalize_pos's lowercasing destroys (see mw.py's own comment on
    is_foreign_pos, and db.py's mw_backfill, which applies the identical
    check the same way). Confirms this endpoint's MW branch does the same,
    not just a plain junk_pos_reason recheck against the already-normalized
    POS (which would never catch this case)."""
    from starlette.testclient import TestClient

    from concordance import deepdef, dictionary, localdict, mw
    from webapp.backend import main

    schema = "cc_test_suggest_word_mw_foreign"
    _fresh_schema(schema)

    monkeypatch.setattr(localdict, "lookup_one", lambda conn, lemma, schema="vocab": [])
    monkeypatch.setattr(dictionary, "_from_freedict", lambda cand, session: False)
    monkeypatch.setattr(dictionary, "_from_wiktionary", lambda cand, session: False)
    monkeypatch.setattr(deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(deepdef, "_from_yourdictionary", lambda cand, session: False)

    monkeypatch.setattr(mw, "mw_api_key", lambda: "fake-key")
    monkeypatch.setattr(mw, "quota_exhausted", lambda: False)
    entry = mw.MWEntry(headword="hatari", part_of_speech="Swahili noun",
                        definitions=["danger"], source="Merriam-Webster API")
    monkeypatch.setattr(mw, "lookup_api", lambda word, key, session: [entry])
    monkeypatch.setattr(mw, "exact_matches", lambda entries, word: entries)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/admin/suggest-word/search", params={"lemma": "hatari"})
        assert res.status_code == 200, res.text
        candidates = res.json()["candidates"]
        assert len(candidates) == 1
        assert candidates[0]["source"] == "Merriam-Webster"
        assert candidates[0]["junk_pos_warning"] == "foreign_language"
    finally:
        main.SCHEMA = old_schema


@pg
def test_search_web_tier_success_path_when_every_other_source_misses(monkeypatch):
    """The one path none of the other tests exercise: every deterministic
    source misses, the model loads fine, and websearch.define_via_web
    actually returns a hit -- proving the web-search candidate itself gets
    built and returned, not just that failure is handled gracefully."""
    from starlette.testclient import TestClient

    from concordance import deepdef, dictionary, localdict, mw, websearch
    from webapp.backend import main

    schema = "cc_test_suggest_word_webhit"
    _fresh_schema(schema)

    monkeypatch.setattr(localdict, "lookup_one", lambda conn, lemma, schema="vocab": [])
    monkeypatch.setattr(dictionary, "_from_freedict", lambda cand, session: False)
    monkeypatch.setattr(dictionary, "_from_wiktionary", lambda cand, session: False)
    monkeypatch.setattr(deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(deepdef, "_from_yourdictionary", lambda cand, session: False)
    monkeypatch.setattr(mw, "mw_api_key", lambda: "")

    import llama_cpp

    class _FakeLlm:
        def close(self):
            pass
    monkeypatch.setattr(llama_cpp, "Llama", lambda *a, **k: _FakeLlm())

    def fake_define_via_web(cand, llm):
        cand.definition = "extracted from a real search result"
        cand.part_of_speech = "adjective"
        return True
    monkeypatch.setattr(websearch, "define_via_web", fake_define_via_web)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/admin/suggest-word/search", params={"lemma": "zorbling"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["web_search_unavailable"] is False
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["source"] == "Web search + local model"
        assert body["candidates"][0]["definition"] == "extracted from a real search result"
    finally:
        main.SCHEMA = old_schema


@pg
def test_search_web_tier_gracefully_degrades_when_model_load_fails(monkeypatch):
    """Every deterministic source misses and the GPU is unavailable (the
    common case when a bulk maintain/ingest job already holds it) -- the
    request should still succeed with an honest flag, not 500."""
    from starlette.testclient import TestClient

    from concordance import deepdef, dictionary, localdict, mw
    from webapp.backend import main

    schema = "cc_test_suggest_word_webfail"
    _fresh_schema(schema)

    monkeypatch.setattr(localdict, "lookup_one", lambda conn, lemma, schema="vocab": [])
    monkeypatch.setattr(dictionary, "_from_freedict", lambda cand, session: False)
    monkeypatch.setattr(dictionary, "_from_wiktionary", lambda cand, session: False)
    monkeypatch.setattr(deepdef, "wordnik_key", lambda: "")
    monkeypatch.setattr(deepdef, "_from_yourdictionary", lambda cand, session: False)
    monkeypatch.setattr(mw, "mw_api_key", lambda: "")

    import llama_cpp

    def raising_llama(*a, **k):
        raise ValueError("Failed to load model from file: models/whatever.gguf")
    monkeypatch.setattr(llama_cpp, "Llama", raising_llama)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.get("/api/admin/suggest-word/search", params={"lemma": "zorbling"})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["candidates"] == []
        assert body["web_search_unavailable"] is True
    finally:
        main.SCHEMA = old_schema


@pg
def test_finalize_inserts_book_less_word_with_provenance_columns():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_suggest_word_finalize"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.post("/api/admin/suggest-word/finalize", json={
            "lemma": "perendinate",
            "definition": "to put off until the day after tomorrow",
            "part_of_speech": "verb",
            "ipa": "/pəˈrɛndɪneɪt/",
            "etymology": "Latin perendinare",
            "synonyms": ["postpone"],
            "definition_source": "Wordnik (century)",
        })
        assert res.status_code == 200, res.text
        word_id = res.json()["id"]
        assert res.json()["lemma"] == "perendinate"

        conn = db.connect(_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT active, admin_suggested, admin_suggested_by, admin_suggested_at,
                           part_of_speech, definition
                    FROM {schema}.word WHERE id = %s""",
                (word_id,),
            )
            active, admin_suggested, by, at, pos, definition = cur.fetchone()
            cur.execute(f"SELECT count(*) FROM {schema}.word_book WHERE word_id = %s", (word_id,))
            book_count = cur.fetchone()[0]
        conn.close()

        assert active is True
        assert admin_suggested is True
        assert by == "adminuser"
        assert at is not None
        assert pos == "verb"
        assert definition == "to put off until the day after tomorrow"
        # Book-less on creation -- a future book using it attaches via
        # sync_book_results' own upsert, nothing special needed here.
        assert book_count == 0
    finally:
        main.SCHEMA = old_schema


@pg
def test_finalize_conflicts_when_lemma_already_exists():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_suggest_word_race"
    _fresh_schema(schema)
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.word (lemma, active) VALUES ('gallimaufry', true)")
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "adminuser", "password": "password123"})

        res = client.post("/api/admin/suggest-word/finalize", json={"lemma": "gallimaufry", "definition": "x"})
        assert res.status_code == 409
    finally:
        main.SCHEMA = old_schema


@pg
def test_finalize_requires_admin():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_suggest_word_auth"
    _fresh_schema(schema)

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        # require_admin (unlike require_viewer/require_user) reports 403 for
        # an anonymous caller, not 401 -- it has no "you could log in" state,
        # only "you aren't allowed" (see its own implementation in main.py).
        res = client.post("/api/admin/suggest-word/finalize", json={"lemma": "foo", "definition": "x"})
        assert res.status_code == 403
        res = client.get("/api/admin/suggest-word/search", params={"lemma": "foo"})
        assert res.status_code == 403
    finally:
        main.SCHEMA = old_schema
