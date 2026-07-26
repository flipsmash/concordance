"""Personal quiz-history/progress API (§ progress). DB-backed tests run only
when a throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL (else
skipped) -- same convention as test_auth.py/test_quiz_api.py.

Deliberately does NOT drive real quizzes through /api/quiz/start to seed data
-- question/distractor selection there is randomized, making exact-percentage
assertions impossible. Every fixture below seeds quiz_session/quiz_question/
quiz_answer with literal INSERTs instead, so expected numbers are known
exactly."""

from __future__ import annotations

import os

import pytest

from concordance import db
from webapp.backend import auth

# `webapp.backend.main` must be imported (fully, module-level) before anything
# imports from `webapp.backend.progress` directly -- progress.py does
# `from webapp.backend import main as _main` at its own top level, and main.py
# imports progress.py back (to register its router) only at the very bottom of
# its own module body, once SCHEMA/require_user/etc already exist. Importing
# `progress` as the very first touch of either module (as a bare
# `from webapp.backend.progress import ...` at this file's top would do)
# starts the cycle from the wrong end and fails with "partially initialized
# module ... has no attribute 'router'" -- see main.py's own comment on this
# same ordering requirement for quiz.py/browse.py/word_sets.py.
from webapp.backend import main as _main  # noqa: F401
from webapp.backend.progress import (
    _accuracy_by_domain,
    _accuracy_by_question_type,
    _by_book,
    _kpi_tiles,
    _score_trend,
    _struggling_words,
    _word_history,
)

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


def _mk_word(cur, schema: str, lemma: str) -> int:
    cur.execute(f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, part_of_speech, active)
                    VALUES (%s, %s, %s, 'noun', true) RETURNING id""",
                (lemma, f"{lemma} definition", f"{lemma} quiz definition"))
    return cur.fetchone()[0]


def _mk_session(cur, schema: str, user_id: int, finished: bool, score_pct: float | None = None) -> int:
    cur.execute(
        f"""INSERT INTO {schema}.quiz_session (user_id, config, feedback_timing, finished_at, score_pct)
            VALUES (%s, '{{}}', 'immediate', {"now()" if finished else "NULL"}, %s)
            RETURNING id""",
        (user_id, score_pct),
    )
    return cur.fetchone()[0]


def _mk_question(cur, schema: str, session_id: int, seq: int, qtype: str, word_ids: list[int]) -> int:
    cur.execute(
        f"""INSERT INTO {schema}.quiz_question (session_id, seq, question_type, target_word_ids, payload)
            VALUES (%s, %s, %s, %s, '{{}}') RETURNING id""",
        (session_id, seq, qtype, word_ids),
    )
    return cur.fetchone()[0]


def _mk_answer(cur, schema: str, question_id: int, word_id: int, is_correct: bool) -> None:
    cur.execute(
        f"""INSERT INTO {schema}.quiz_answer (question_id, word_id, response, is_correct)
            VALUES (%s, %s, '{{}}', %s)""",
        (question_id, word_id, is_correct),
    )


@pg
def test_matching_fan_out_guard():
    """One finished session, one matching question with 4 pairs, 3/4 correct
    -- by_question_type must report total=1 (one QUESTION) and
    accuracy_pct == 75.0, not total=4 (four answer rows)."""
    schema = "cc_test_progress_matching"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('u', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
            word_ids = [_mk_word(cur, schema, f"matchword{i}") for i in range(4)]
            session_id = _mk_session(cur, schema, user_id, finished=True, score_pct=75.0)
            question_id = _mk_question(cur, schema, session_id, 1, "matching", word_ids)
            for i, wid in enumerate(word_ids):
                _mk_answer(cur, schema, question_id, wid, is_correct=(i < 3))
        conn.commit()

        with conn.cursor() as cur:
            buckets = {b.key: b for b in _accuracy_by_question_type(cur, schema, user_id)}
        assert buckets["matching"].total == 1, "must count questions, not answer rows"
        assert buckets["matching"].accuracy_pct == 75.0
        assert buckets["mc"].total == 0 and buckets["mc"].accuracy_pct is None
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_cross_user_isolation():
    schema = "cc_test_progress_isolation"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('a', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_a = cur.fetchone()[0]
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('b', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_b = cur.fetchone()[0]
            word_a = _mk_word(cur, schema, "wordforA")
            word_b = _mk_word(cur, schema, "wordforB")
            sess_a = _mk_session(cur, schema, user_a, finished=True, score_pct=100.0)
            q_a = _mk_question(cur, schema, sess_a, 1, "mc", [word_a])
            _mk_answer(cur, schema, q_a, word_a, is_correct=True)
            sess_b = _mk_session(cur, schema, user_b, finished=True, score_pct=0.0)
            q_b = _mk_question(cur, schema, sess_b, 1, "mc", [word_b])
            _mk_answer(cur, schema, q_b, word_b, is_correct=False)
        conn.commit()

        with conn.cursor() as cur:
            trend_a = _score_trend(cur, schema, user_a)
            trend_b = _score_trend(cur, schema, user_b)
        assert [p.session_id for p in trend_a] == [sess_a]
        assert [p.session_id for p in trend_b] == [sess_b]
        assert trend_a[0].score_pct == 100.0
        assert trend_b[0].score_pct == 0.0
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_unfinished_session_excluded():
    schema = "cc_test_progress_unfinished"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('u', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
            word_id = _mk_word(cur, schema, "abandonedword")
            sess_id = _mk_session(cur, schema, user_id, finished=False)
            q_id = _mk_question(cur, schema, sess_id, 1, "mc", [word_id])
            _mk_answer(cur, schema, q_id, word_id, is_correct=True)
        conn.commit()

        with conn.cursor() as cur:
            trend = _score_trend(cur, schema, user_id)
            tiles = _kpi_tiles(cur, schema, user_id)
        assert trend == []
        quizzes_taken = next(t for t in tiles if t.label == "Quizzes taken")
        assert quizzes_taken.value == 0
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_domain_bucket_overlap_not_partitioned():
    """A word in two USAS categories spanning two different buckets must
    increment BOTH buckets' totals -- confirms the deliberate non-partitioning
    EXISTS semantics (copied from browse.py's _bucket_counts()), not a bug."""
    schema = "cc_test_progress_domain"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('u', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
            word_id = _mk_word(cur, schema, "straddleword")
            # one category in mind_language (A), one in nature_science (F)
            cur.execute(f"""INSERT INTO {schema}.category (taxonomy, code, name)
                            VALUES ('usas', 'A1', 'General mind') RETURNING id""")
            cat_a = cur.fetchone()[0]
            cur.execute(f"""INSERT INTO {schema}.category (taxonomy, code, name)
                            VALUES ('usas', 'F1', 'General science') RETURNING id""")
            cat_f = cur.fetchone()[0]
            cur.execute(f"""INSERT INTO {schema}.word_category (word_id, category_id, is_primary)
                            VALUES (%s, %s, true), (%s, %s, false)""",
                        (word_id, cat_a, word_id, cat_f))
            sess_id = _mk_session(cur, schema, user_id, finished=True, score_pct=100.0)
            q_id = _mk_question(cur, schema, sess_id, 1, "mc", [word_id])
            _mk_answer(cur, schema, q_id, word_id, is_correct=True)
        conn.commit()

        with conn.cursor() as cur:
            buckets = {b.key: b for b in _accuracy_by_domain(cur, schema, user_id)}
        assert buckets["mind_language"].total == 1
        assert buckets["nature_science"].total == 1
        assert buckets["people_society"].total == 0
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_empty_state_is_null_not_zero():
    schema = "cc_test_progress_empty"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('freshuser', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
        conn.commit()

        with conn.cursor() as cur:
            tiles = _kpi_tiles(cur, schema, user_id)
            trend = _score_trend(cur, schema, user_id)
            by_type = _accuracy_by_question_type(cur, schema, user_id)
            by_domain = _accuracy_by_domain(cur, schema, user_id)
            books = _by_book(cur, schema, user_id)
            struggling = _struggling_words(cur, schema, user_id, limit=25)
        accuracy_tile = next(t for t in tiles if t.label == "Lifetime accuracy")
        assert accuracy_tile.value is None
        assert trend == []
        assert all(b.accuracy_pct is None for b in by_type)
        assert all(b.accuracy_pct is None for b in by_domain)
        assert books == []
        assert struggling == []
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_struggling_words_exposure_floor():
    schema = "cc_test_progress_struggling"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('u', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
            one_shot = _mk_word(cur, schema, "oneshotword")
            two_shot = _mk_word(cur, schema, "twoshotword")
            # 1 attempt, 0 correct -- below the exposure floor, must be excluded
            cur.execute(
                f"""INSERT INTO {schema}.word_review_schedule
                        (user_id, word_id, streak, correct_count, incorrect_count)
                    VALUES (%s, %s, 0, 0, 1)""",
                (user_id, one_shot),
            )
            # 2 attempts, 1 correct -- included, miss_rate == 0.5
            cur.execute(
                f"""INSERT INTO {schema}.word_review_schedule
                        (user_id, word_id, streak, correct_count, incorrect_count)
                    VALUES (%s, %s, 0, 1, 1)""",
                (user_id, two_shot),
            )
        conn.commit()

        with conn.cursor() as cur:
            rows = _struggling_words(cur, schema, user_id, limit=25)
        word_ids = {r.word_id for r in rows}
        assert one_shot not in word_ids
        assert two_shot in word_ids
        assert next(r for r in rows if r.word_id == two_shot).miss_rate == 0.5
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


@pg
def test_word_history_empty_and_populated():
    schema = "cc_test_progress_word_history"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    try:
        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('u', %s) RETURNING id",
                        (auth.hash_password("password123"),))
            user_id = cur.fetchone()[0]
            never_quizzed = _mk_word(cur, schema, "neverquizzed")
            quizzed = _mk_word(cur, schema, "quizzedword")
            sess_id = _mk_session(cur, schema, user_id, finished=True, score_pct=100.0)
            q_id = _mk_question(cur, schema, sess_id, 1, "mc", [quizzed])
            _mk_answer(cur, schema, q_id, quizzed, is_correct=True)
        conn.commit()

        with conn.cursor() as cur:
            empty = _word_history(cur, schema, user_id, never_quizzed)
            populated = _word_history(cur, schema, user_id, quizzed)
        assert empty.answers == []
        assert empty.streak == 0 and empty.personal_difficulty is None
        assert len(populated.answers) == 1
        assert populated.answers[0].is_correct is True
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()


# --- HTTP-level tests (real auth/routing, not just direct function calls) -----

@pg
def test_progress_endpoints_require_auth_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_progress_http_auth"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        assert client.get("/api/progress/overview").status_code == 401
        assert client.get("/api/progress/books").status_code == 401
        assert client.get("/api/progress/struggling").status_code == 401
        assert client.get("/api/progress/words/1").status_code == 401
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_progress_full_round_trip_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_progress_http_roundtrip"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('proguser', %s) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
        word_id = _mk_word(cur, schema, "roundtripword")
        book_word_id = _mk_word(cur, schema, "bookword")
        cur.execute(f"INSERT INTO {schema}.book (title, author) VALUES ('Test Book', 'Test Author') RETURNING id")
        book_id = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.word_book (word_id, book_id) VALUES (%s, %s)", (book_word_id, book_id))
        sess_id = _mk_session(cur, schema, user_id, finished=True, score_pct=100.0)
        q1 = _mk_question(cur, schema, sess_id, 1, "mc", [word_id])
        _mk_answer(cur, schema, q1, word_id, is_correct=True)
        q2 = _mk_question(cur, schema, sess_id, 2, "mc", [book_word_id])
        _mk_answer(cur, schema, q2, book_word_id, is_correct=True)
        cur.execute(
            f"""INSERT INTO {schema}.word_review_schedule (user_id, word_id, streak, correct_count, incorrect_count)
                VALUES (%s, %s, 1, 1, 0)""",
            (user_id, word_id),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "proguser", "password": "password123"})

        overview = client.get("/api/progress/overview")
        assert overview.status_code == 200, overview.text
        body = overview.json()
        assert next(t for t in body["tiles"] if t["label"] == "Quizzes taken")["value"] == 1
        assert len(body["trend"]) == 1 and body["trend"][0]["score_pct"] == 100.0
        assert sum(b["total"] for b in body["by_question_type"]) == 2  # 2 mc questions

        books = client.get("/api/progress/books")
        assert books.status_code == 200
        assert any(b["title"] == "Test Book" and b["author"] == "Test Author" for b in books.json())

        history = client.get(f"/api/progress/words/{word_id}")
        assert history.status_code == 200
        hbody = history.json()
        assert hbody["streak"] == 1
        assert len(hbody["answers"]) == 1

        # a word never quizzed for this user gets a legitimate empty history, not a 404
        blank = client.get("/api/progress/words/999999")
        assert blank.status_code == 200
        assert blank.json()["answers"] == []
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_progress_cross_user_isolation_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_progress_http_isolation"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('httpa', %s) RETURNING id",
                    (auth.hash_password("password123"),))
        user_a = cur.fetchone()[0]
        cur.execute(f"INSERT INTO {schema}.users (username, password_hash) VALUES ('httpb', %s)",
                    (auth.hash_password("password123"),))
        word_a = _mk_word(cur, schema, "httpAword")
        sess_a = _mk_session(cur, schema, user_a, finished=True, score_pct=100.0)
        q_a = _mk_question(cur, schema, sess_a, 1, "mc", [word_a])
        _mk_answer(cur, schema, q_a, word_a, is_correct=True)
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        # log in as user B, who has zero history -- must never see user A's session
        client.post("/api/auth/login", json={"username": "httpb", "password": "password123"})
        overview = client.get("/api/progress/overview").json()
        assert overview["trend"] == []
        assert next(t for t in overview["tiles"] if t["label"] == "Quizzes taken")["value"] == 0
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
