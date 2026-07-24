"""Quiz-taking + admin-settings API. DB-backed tests run only when a throwaway
Postgres is provided via CONCORDANCE_TEST_DB_URL (else skipped) -- same
convention as test_auth.py, including its main.SCHEMA-monkeypatch pattern for
exercising real registered routes against a disposable schema."""

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


def _seed_corpus(conn, schema: str, n: int = 12) -> None:
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, part_of_speech, active)
                    VALUES (%s, %s, %s, 'noun', true) RETURNING id""",
                (f"quizword{i}", f"definition {i}", f"quiz definition {i}"),
            )
            wid = cur.fetchone()[0]
            cur.execute(
                f"INSERT INTO {schema}.word_difficulty (word_id, quizzable, difficulty) VALUES (%s, true, 50.0)",
                (wid,),
            )
    conn.commit()


@pg
def test_quiz_round_trip_and_admin_settings_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    _seed_corpus(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES ('quizuser', %s, false) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash, is_admin) VALUES ('quizadmin', %s, true)",
            (auth.hash_password("adminpassword1"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")

        # Anonymous: every quiz/admin route is refused.
        assert client.get("/api/quiz/meta").status_code == 401
        assert client.post("/api/quiz/start", json={}).status_code == 401
        assert client.get("/api/admin/settings").status_code == 403

        client.post("/api/auth/login", json={"username": "quizuser", "password": "password123"})

        meta = client.get("/api/quiz/meta").json()
        assert "noun" in meta["pos_values"]

        # Non-admin still can't touch admin settings.
        assert client.get("/api/admin/settings").status_code == 403

        res = client.post("/api/quiz/start", json={"length": 3, "mc_choice_count": 4})
        assert res.status_code == 200, res.text
        started = res.json()
        session_id = started["session_id"]
        assert started["feedback_timing"] == "immediate"  # seeded default
        assert started["total_questions"] == 3

        seen_labels = set()
        for _ in range(started["total_questions"]):
            state = client.get(f"/api/quiz/{session_id}").json()
            assert state["completed"] is False
            q = state["question"]
            # The answer key must never appear in a client-facing question payload.
            assert "correct_word_id" not in q
            assert "nota_is_correct" not in q
            for opt in q["options"]:
                assert set(opt.keys()) == {"word_id", "label"}
            seen_labels.add(q["prompt"])

            ans = client.post(
                f"/api/quiz/{session_id}/answer",
                json={"question_id": q["question_id"], "selected_word_id": q["options"][0]["word_id"]},
            )
            assert ans.status_code == 200
            body = ans.json()
            # feedback_timing=immediate -> correctness disclosed right away.
            assert body["is_correct"] in (True, False)
            assert body["correct_label"]

            # A second answer to the same question is rejected.
            dup = client.post(
                f"/api/quiz/{session_id}/answer",
                json={"question_id": q["question_id"], "selected_word_id": q["options"][0]["word_id"]},
            )
            assert dup.status_code == 400

        assert len(seen_labels) == started["total_questions"]  # no repeated question

        state = client.get(f"/api/quiz/{session_id}").json()
        assert state["completed"] is True
        assert state["question"] is None

        fin = client.post(f"/api/quiz/{session_id}/finish")
        assert fin.status_code == 200
        finished = fin.json()
        assert finished["total_questions"] == 3
        assert 0.0 <= finished["score_pct"] <= 100.0

        # Idempotent: finishing again returns the same score, doesn't recompute.
        fin2 = client.post(f"/api/quiz/{session_id}/finish")
        assert fin2.json()["score_pct"] == finished["score_pct"]

        review = client.get(f"/api/quiz/{session_id}/review").json()
        assert review["score_pct"] == finished["score_pct"]
        assert len(review["items"]) == 3
        for item in review["items"]:
            assert item["quiz_definition"].startswith("quiz definition")
            assert item["correct_label"]

        # A quiz session belongs to its creator only.
        client.post("/api/auth/logout")
        client.post("/api/auth/login", json={"username": "quizadmin", "password": "adminpassword1"})
        assert client.get(f"/api/quiz/{session_id}").status_code == 404

        # --- admin settings, now logged in as the admin ---
        settings = client.get("/api/admin/settings").json()["settings"]
        assert settings["quiz_feedback_timing"]["mode"] == "immediate"  # seeded default

        put = client.put("/api/admin/settings", json={"key": "quiz_feedback_timing", "value": {"mode": "end_of_test"}})
        assert put.status_code == 200
        assert put.json()["settings"]["quiz_feedback_timing"]["mode"] == "end_of_test"

        # A freshly-started session snapshots the new mode...
        res2 = client.post("/api/quiz/start", json={"length": 1})
        session2 = res2.json()
        assert session2["feedback_timing"] == "end_of_test"
        q2 = client.get(f"/api/quiz/{session2['session_id']}").json()["question"]
        ans2 = client.post(
            f"/api/quiz/{session2['session_id']}/answer",
            json={"question_id": q2["question_id"], "selected_word_id": q2["options"][0]["word_id"]},
        ).json()
        # ...so correctness is withheld now, unlike the earlier immediate-mode session.
        assert ans2 == {"accepted": True, "is_correct": None, "correct_word_id": None,
                         "correct_label": None, "correct_answer": None, "pair_results": None,
                         "quiz_definition": None}
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


def _seed_corpus_at_difficulty(conn, schema: str, prefix: str, n: int, difficulty: float) -> None:
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                f"""INSERT INTO {schema}.word (lemma, definition, quiz_definition, part_of_speech, active)
                    VALUES (%s, %s, %s, 'noun', true) RETURNING id""",
                (f"{prefix}{i}", f"definition {prefix}{i}", f"quiz definition {prefix}{i}"),
            )
            wid = cur.fetchone()[0]
            cur.execute(
                f"INSERT INTO {schema}.word_difficulty (word_id, quizzable, difficulty) VALUES (%s, true, %s)",
                (wid, difficulty),
            )
    conn.commit()


@pg
def test_select_target_words_prefers_personal_difficulty_over_ex_ante():
    # Whitebox: _select_target_words is where the plan's COALESCE(personal,
    # ex-ante) filtering actually lives -- going through the full HTTP
    # question-build pipeline instead would conflate this with distractor
    # selection's own (unrelated) difficulty-band filter, which reads only
    # the raw ex-ante wd.difficulty column and has no per-user context to
    # COALESCE against, so it can't be used to isolate this behavior.
    from webapp.backend import main
    from webapp.backend import quiz as quiz_module

    schema = "cc_test_quiz_personal_difficulty_select"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    # "otherword*": ex-ante 50, no personal override -- always in the [0,60]
    # band, never in [80,100].
    _seed_corpus_at_difficulty(conn, schema, "otherword", 5, 50.0)
    # "personalword0": ex-ante 50 (same as otherword*) but a personal
    # override of 90 -- selection must follow the override, not the ex-ante.
    _seed_corpus_at_difficulty(conn, schema, "personalword", 1, 50.0)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('personaldiffuser', %s) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
        cur.execute(f"SELECT id FROM {schema}.word WHERE lemma = 'personalword0'")
        personal_word_id = cur.fetchone()[0]
        cur.execute(
            f"""INSERT INTO {schema}.word_personal_difficulty
                    (user_id, word_id, item_rating, personal_difficulty, based_on_correct)
                VALUES (%s, %s, 2.5, 90.0, false)""",
            (user_id, personal_word_id),
        )
    conn.commit()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        # High band: only personalword0 qualifies -- its ex-ante (50) is out
        # of range, so only the personal override (90) can put it here.
        body_high = quiz_module.QuizStartRequest(length=1, difficulty_min=80, difficulty_max=100)
        pool_high = quiz_module._select_target_words(conn, body_high, 100, set(), user_id)
        assert [w["lemma"] for w in pool_high] == ["personalword0"]

        # Low band: the ex-ante 50 alone would qualify personalword0, but
        # its personal override (90) is what selection actually consults,
        # so it must be excluded here -- only the otherword* words remain.
        body_low = quiz_module.QuizStartRequest(length=10, difficulty_min=0, difficulty_max=60)
        pool_low = quiz_module._select_target_words(conn, body_low, 100, set(), user_id)
        lemmas_low = {w["lemma"] for w in pool_low}
        assert "personalword0" not in lemmas_low
        assert lemmas_low == {f"otherword{i}" for i in range(5)}
    finally:
        main.SCHEMA = old_schema
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
        conn.close()


@pg
def test_guessing_floor_written_reflects_actual_assembled_count_not_nominal_http():
    # calibration.py's guessing_floor is load-bearing for personalized
    # difficulty -- it must come from the ACTUAL number of options/members a
    # question ends up with, not the nominal mc_choice_count/matching_set_size
    # requested, since both builders can fall short of nominal when the
    # corpus can't supply enough candidates (see _build_mc_payload's
    # distractor_count reassignment and _build_matching_payload's `members`
    # length). A corpus deliberately too small to fill either nominal count
    # forces both shortfall paths.
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_guessing_floor_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    # Exactly 3 quizzable words total: any mc/matching question drawn from
    # this corpus tops out at 3 options/members, however large the nominal
    # request.
    _seed_corpus(conn, schema, n=3)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('floorhttpuser', %s) RETURNING id",
            (auth.hash_password("password123"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "floorhttpuser", "password": "password123"})

        # MC, nominal 8 choices -- only 3 words exist, so the real option
        # count tops out at 3 (1 target + up to 2 distractors), never 8.
        res = client.post("/api/quiz/start", json={"length": 1, "types": ["mc"], "mc_choice_count": 8})
        assert res.status_code == 200, res.text
        assert res.json()["total_questions"] == 1
        session_id = res.json()["session_id"]
        q = client.get(f"/api/quiz/{session_id}").json()["question"]
        actual_option_count = len(q["options"])
        assert actual_option_count < 8
        client.post(f"/api/quiz/{session_id}/answer",
                    json={"question_id": q["question_id"], "selected_word_id": q["options"][0]["word_id"]})

        # Matching, nominal set size 8 -- same 3-word ceiling.
        res2 = client.post("/api/quiz/start", json={"length": 1, "types": ["matching"], "matching_set_size": 8})
        assert res2.status_code == 200, res2.text
        assert res2.json()["total_questions"] == 1
        session2_id = res2.json()["session_id"]
        q2 = client.get(f"/api/quiz/{session2_id}").json()["question"]
        actual_member_count = len(q2["word_slots"])
        assert actual_member_count < 8
        pairs = [{"word_id": ws["word_id"], "definition_slot": q2["definition_slots"][0]["slot"]}
                 for ws in q2["word_slots"]]
        client.post(f"/api/quiz/{session2_id}/answer",
                    json={"question_id": q2["question_id"], "pairs": pairs})
    finally:
        main.SCHEMA = old_schema

    verify_conn = db.connect(_URL)
    try:
        with verify_conn.cursor() as cur:
            cur.execute(f"""select qa.guessing_floor, qa.question_type
                            from {schema}.quiz_answer qa
                            join {schema}.quiz_question qq on qq.id = qa.question_id
                            where qq.question_type = 'mc'""")
            mc_rows = cur.fetchall()
            assert len(mc_rows) == 1
            assert mc_rows[0] == (1.0 / actual_option_count, "mc")

            cur.execute(f"""select qa.guessing_floor, qa.question_type
                            from {schema}.quiz_answer qa
                            join {schema}.quiz_question qq on qq.id = qa.question_id
                            where qq.question_type = 'matching'""")
            matching_rows = cur.fetchall()
            assert len(matching_rows) == actual_member_count
            expected = 1.0 / actual_member_count
            assert all(row == (expected, "matching") for row in matching_rows)

            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        verify_conn.commit()
    finally:
        verify_conn.close()


@pg
def test_feedback_timing_defaults_to_immediate_when_setting_row_missing():
    from webapp.backend import quiz as quiz_module

    schema = "cc_test_quiz_default_setting"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {schema}.app_settings WHERE key = 'quiz_feedback_timing'")
    conn.commit()

    assert _call_feedback_timing(quiz_module, conn, schema) == "immediate"

    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    conn.close()


def _call_feedback_timing(quiz_module, conn, schema):
    from webapp.backend import main
    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        return quiz_module._feedback_timing(conn)
    finally:
        main.SCHEMA = old_schema


@pg
def test_true_false_round_trip_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_tf_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    _seed_corpus(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('tfuser', %s)",
            (auth.hash_password("password123"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "tfuser", "password": "password123"})

        res = client.post("/api/quiz/start", json={"length": 3, "types": ["true_false"]})
        assert res.status_code == 200, res.text
        session_id = res.json()["session_id"]

        for _ in range(3):
            state = client.get(f"/api/quiz/{session_id}").json()
            q = state["question"]
            assert q["question_type"] == "true_false"
            assert q["statement_word"] and q["statement_definition"]
            assert "is_true" not in q  # answer key never reaches the client

            ans = client.post(f"/api/quiz/{session_id}/answer",
                               json={"question_id": q["question_id"], "answer": True}).json()
            assert ans["correct_answer"] in (True, False)
            assert ans["is_correct"] == (ans["correct_answer"] is True)

            dup = client.post(f"/api/quiz/{session_id}/answer",
                               json={"question_id": q["question_id"], "answer": True})
            assert dup.status_code == 400

        fin = client.post(f"/api/quiz/{session_id}/finish").json()
        assert fin["total_questions"] == 3
        assert fin["score_pct"] in (0.0, 33.3, 66.7, 100.0)  # binary credit per question -- n/3 exactly
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_matching_round_trip_with_partial_credit_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_matching_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    _seed_corpus(conn, schema, n=8)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('matchuser', %s)",
            (auth.hash_password("password123"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "matchuser", "password": "password123"})

        res = client.post("/api/quiz/start",
                           json={"length": 1, "types": ["matching"], "matching_set_size": 4})
        assert res.status_code == 200, res.text
        session_id = res.json()["session_id"]

        state = client.get(f"/api/quiz/{session_id}").json()
        q = state["question"]
        assert q["question_type"] == "matching"
        assert len(q["word_slots"]) == 4
        assert len(q["definition_slots"]) == 4
        assert "correct_mapping" not in q  # answer key never reaches the client
        # Leak-safety: comparing the two lists must never reveal the pairing --
        # definition_slots must carry no word_id at all.
        for d in q["definition_slots"]:
            assert set(d.keys()) == {"slot", "quiz_definition"}

        # Peek at the real answer key server-side (never exposed to the client)
        # to deliberately submit 2 correct + 2 incorrect pairs.
        conn2 = db.connect(_URL)
        with conn2.cursor() as cur:
            cur.execute(f"SELECT payload FROM {schema}.quiz_question WHERE id = %s", (q["question_id"],))
            correct_mapping = cur.fetchone()[0]["correct_mapping"]
        conn2.close()

        word_ids = [w["word_id"] for w in q["word_slots"]]
        all_slots = [d["slot"] for d in q["definition_slots"]]
        pairs = []
        for i, wid in enumerate(word_ids):
            if i < 2:
                pairs.append({"word_id": wid, "definition_slot": correct_mapping[str(wid)]})
            else:
                wrong_slot = next(s for s in all_slots if s != correct_mapping[str(wid)])
                pairs.append({"word_id": wid, "definition_slot": wrong_slot})

        ans = client.post(f"/api/quiz/{session_id}/answer",
                           json={"question_id": q["question_id"], "pairs": pairs})
        assert ans.status_code == 200, ans.text
        body = ans.json()
        assert sum(1 for p in body["pair_results"] if p["is_correct"]) == 2

        dup = client.post(f"/api/quiz/{session_id}/answer", json={"question_id": q["question_id"], "pairs": pairs})
        assert dup.status_code == 400

        fin = client.post(f"/api/quiz/{session_id}/finish").json()
        assert fin["total_questions"] == 1
        assert fin["score_pct"] == 50.0  # 2/4 pairs correct, per-pair credit

        review = client.get(f"/api/quiz/{session_id}/review").json()
        assert review["items"][0]["credit"] == 0.5
        assert review["items"][0]["is_correct"] is False
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_blended_quiz_can_produce_multiple_question_types_http():
    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_blend_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    _seed_corpus(conn, schema, n=40)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('blenduser', %s)",
            (auth.hash_password("password123"),),
        )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "blenduser", "password": "password123"})

        res = client.post("/api/quiz/start", json={
            "length": 10, "types": ["mc", "true_false", "matching"], "matching_set_size": 3,
        })
        assert res.status_code == 200, res.text

        seen_types = set()
        session_id = res.json()["session_id"]
        for _ in range(res.json()["total_questions"]):
            q = client.get(f"/api/quiz/{session_id}").json()["question"]
            seen_types.add(q["question_type"])
            if q["question_type"] == "mc":
                body = {"question_id": q["question_id"], "selected_word_id": q["options"][0]["word_id"]}
            elif q["question_type"] == "true_false":
                body = {"question_id": q["question_id"], "answer": True}
            else:
                body = {"question_id": q["question_id"],
                        "pairs": [{"word_id": w["word_id"], "definition_slot": q["definition_slots"][0]["slot"]}
                                  for w in q["word_slots"]]}
            assert client.post(f"/api/quiz/{session_id}/answer", json=body).status_code == 200

        # With a large enough pool and all three types weighted equally, seeing
        # only one type across 10 questions would be a red flag, not just bad luck.
        assert len(seen_types) >= 2
        assert client.post(f"/api/quiz/{session_id}/finish").status_code == 200
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_spaced_repetition_prefers_eligible_words_but_falls_back_when_short_http():
    from datetime import datetime, timedelta, timezone

    from starlette.testclient import TestClient

    from webapp.backend import main

    schema = "cc_test_quiz_sr_http"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    # A generous corpus: 4 questions at mc_choice_count=2 (1 target + 1
    # distractor each, never reusing a word within the session) needs at
    # least 8 distinct words, plus room for the fallback assertion below --
    # 40 keeps every word other than the 8 designated "eligible" ones safely
    # supplied with distractors without exhausting the eligible set.
    _seed_corpus(conn, schema, n=40)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('srhttpuser', %s) RETURNING id",
            (auth.hash_password("password123"),),
        )
        user_id = cur.fetchone()[0]
        cur.execute(f"SELECT id FROM {schema}.word ORDER BY id")
        word_ids = [r[0] for r in cur.fetchall()]
        # First 8 words are "eligible" (due in the past / never scored) --
        # deliberately more than the 4 questions requested below, so that an
        # eligible word incidentally drawn as another eligible word's MC
        # distractor still leaves enough eligible words free to serve as
        # targets (worst case: 4 targets + 3 distractors = 7 of the 8
        # consumed before the last target is locked in). Every other word is
        # explicitly "not yet eligible" (due far in the future) so none of
        # them can be mistaken for eligible via a missing row.
        future = datetime.now(timezone.utc) + timedelta(days=30)
        past = datetime.now(timezone.utc) - timedelta(days=1)
        for wid in word_ids[:4]:
            cur.execute(
                f"""INSERT INTO {schema}.word_review_schedule (user_id, word_id, next_eligible_at)
                    VALUES (%s, %s, %s)""",
                (user_id, wid, past),
            )
        # word_ids[4:8] stay eligible via no row at all (never seen).
        for wid in word_ids[8:]:
            cur.execute(
                f"""INSERT INTO {schema}.word_review_schedule (user_id, word_id, next_eligible_at)
                    VALUES (%s, %s, %s)""",
                (user_id, wid, future),
            )
    conn.commit()
    conn.close()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    try:
        client = TestClient(main.app, base_url="https://testserver")
        client.post("/api/auth/login", json={"username": "srhttpuser", "password": "password123"})

        # Exactly as many questions as eligible words -- every target should
        # come from the eligible set. mc_choice_count=2 keeps each question's
        # distractor draw small so it can't accidentally exhaust the corpus.
        res = client.post("/api/quiz/start",
                           json={"length": 4, "mc_choice_count": 2, "spaced_repetition_enabled": True})
        assert res.status_code == 200, res.text
        session_id = res.json()["session_id"]
        assert res.json()["total_questions"] == 4
        for _ in range(4):
            q = client.get(f"/api/quiz/{session_id}").json()["question"]
            client.post(f"/api/quiz/{session_id}/answer",
                        json={"question_id": q["question_id"], "selected_word_id": q["options"][0]["word_id"]})
        review = client.get(f"/api/quiz/{session_id}/review").json()
        targets = {item["target_lemma"] for item in review["items"]}
        eligible_lemmas = {f"quizword{i}" for i in range(8)}  # per _seed_corpus's naming
        assert targets <= eligible_lemmas, f"expected only eligible words, got {targets}"

        # More questions than the eligible pool has -- must still fill the
        # request by falling back to not-yet-eligible words, never erroring.
        res2 = client.post("/api/quiz/start",
                            json={"length": 10, "mc_choice_count": 2, "spaced_repetition_enabled": True})
        assert res2.status_code == 200, res2.text
        assert res2.json()["total_questions"] == 10
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
