"""Word-browsing API (author/book/domain/difficulty facets). DB-backed tests
run only when a throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL
(else skipped) -- same convention as test_quiz_api.py, including its
main.SCHEMA-monkeypatch pattern for exercising real registered routes
against a disposable schema."""

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


def _login(client, username="browseuser", password="password123"):
    client.post("/api/auth/login", json={"username": username, "password": password})


def _setup(schema: str):
    """Fresh schema + a logged-in TestClient, following test_quiz_api.py's
    main.SCHEMA-monkeypatch convention. Returns (client, conn, restore_fn)."""
    from starlette.testclient import TestClient

    from webapp.backend import main

    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    db.apply_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.users (username, password_hash) VALUES ('browseuser', %s)",
            (auth.hash_password("password123"),),
        )
    conn.commit()

    old_schema = main.SCHEMA
    main.SCHEMA = schema
    client = TestClient(main.app, base_url="https://testserver")
    _login(client)

    def restore():
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()

    return client, conn, restore


def _insert_word(conn, schema, lemma, *, definition="a definition", pos="noun",
                  difficulty=None, archaic=None, quizzable=None):
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.word (lemma, definition, part_of_speech, active)
                VALUES (%s, %s, %s, true) RETURNING id""",
            (lemma, definition, pos),
        )
        wid = cur.fetchone()[0]
        if difficulty is not None or archaic is not None or quizzable is not None:
            cur.execute(
                f"""INSERT INTO {schema}.word_difficulty (word_id, difficulty, archaic, quizzable)
                    VALUES (%s, %s, %s, %s)""",
                (wid, difficulty, archaic, quizzable),
            )
    return wid


def _insert_book(conn, schema, title, author=None):
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.book (title, author) VALUES (%s, %s) RETURNING id",
            (title, author),
        )
        return cur.fetchone()[0]


def _link(conn, schema, word_id, book_id):
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {schema}.word_book (word_id, book_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (word_id, book_id),
        )


def _category(conn, schema, code, name="Test Category"):
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.category (taxonomy, code, name, level, assignable)
                VALUES ('usas', %s, %s, 0, true)
                ON CONFLICT (taxonomy, code) DO UPDATE SET name = EXCLUDED.name
                RETURNING id""",
            (code, name),
        )
        return cur.fetchone()[0]


def _tag_domain(conn, schema, word_id, category_id, is_primary=True):
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.word_category (word_id, category_id, is_primary, source)
                VALUES (%s, %s, %s, 'llm')""",
            (word_id, category_id, is_primary),
        )


@pg
def test_word_in_multiple_books_dedupes_and_does_not_inflate_total():
    client, conn, restore = _setup("cc_test_browse_dedup")
    try:
        b1 = _insert_book(conn, "cc_test_browse_dedup", "Book One", author="Author, Some")
        b2 = _insert_book(conn, "cc_test_browse_dedup", "Book Two", author="Author, Some")
        shared = _insert_word(conn, "cc_test_browse_dedup", "shared")
        only_b1 = _insert_word(conn, "cc_test_browse_dedup", "onlyfirst")
        _link(conn, "cc_test_browse_dedup", shared, b1)
        _link(conn, "cc_test_browse_dedup", shared, b2)
        _link(conn, "cc_test_browse_dedup", only_b1, b1)
        conn.commit()

        # Filtering by BOTH books the shared word belongs to must still
        # return it exactly once, and `total` must reflect that, not the
        # (word, book) pair count.
        res = client.get("/api/browse/words", params={"book_id": [b1, b2]})
        assert res.status_code == 200, res.text
        data = res.json()
        lemmas = [w["lemma"] for w in data["items"]]
        assert lemmas.count("shared") == 1
        assert data["total"] == 2  # shared + onlyfirst, not 3

        # Filtering by author (same author on both books) hits the same
        # word_book fan-out through a different join path -- must also dedupe.
        res = client.get("/api/browse/words", params={"author": "Author, Some"})
        assert res.json()["total"] == 2

        # The authors listing aggregates the OTHER direction (word_book is
        # the count target here, not a filter) -- word_count must use
        # count(DISTINCT word), not double-count "shared" for appearing in
        # both of this author's books.
        authors = client.get("/api/browse/authors").json()["items"]
        row = next(a for a in authors if a["author"] == "Author, Some")
        assert row["word_count"] == 2
        assert row["book_count"] == 2
    finally:
        restore()


@pg
def test_combined_facets_intersect_regardless_of_which_is_set_first():
    schema = "cc_test_browse_combined"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Alpha", author="Alpha, Writer")
        b2 = _insert_book(conn, schema, "Beta", author="Beta, Writer")
        cat_science = _category(conn, schema, "F", "Nature Science Test")

        # Matches every facet we'll apply together.
        target = _insert_word(conn, schema, "target", difficulty=50.0)
        _link(conn, schema, target, b1)
        _tag_domain(conn, schema, target, cat_science)

        # Right author+book, wrong domain.
        wrong_domain = _insert_word(conn, schema, "wrongdomain", difficulty=50.0)
        _link(conn, schema, wrong_domain, b1)

        # Right domain, wrong book (different author).
        wrong_book = _insert_word(conn, schema, "wrongbook", difficulty=50.0)
        _link(conn, schema, wrong_book, b2)
        _tag_domain(conn, schema, wrong_book, cat_science)

        # Right everything except difficulty out of range.
        wrong_difficulty = _insert_word(conn, schema, "wrongdifficulty", difficulty=5.0)
        _link(conn, schema, wrong_difficulty, b1)
        _tag_domain(conn, schema, wrong_difficulty, cat_science)
        conn.commit()

        params = {
            "author": "Alpha, Writer", "book_id": [b1], "domain": ["nature_science"],
            "difficulty_min": 40, "difficulty_max": 60,
        }
        res = client.get("/api/browse/words", params=params)
        items = res.json()["items"]
        assert [w["lemma"] for w in items] == ["target"]

        # Same filters, submitted as a different dict-iteration/query-param
        # order -- GET params are inherently unordered as a set of ANDed
        # predicates, so this should be identical, confirming no filter
        # accidentally depends on being applied "first."
        reordered = {
            "difficulty_max": 60, "domain": ["nature_science"], "book_id": [b1],
            "difficulty_min": 40, "author": "Alpha, Writer",
        }
        res2 = client.get("/api/browse/words", params=reordered)
        assert [w["lemma"] for w in res2.json()["items"]] == ["target"]
    finally:
        restore()


@pg
def test_difficulty_and_quizzable_filters_exclude_unscored_words_only_when_active():
    schema = "cc_test_browse_sparse"
    client, conn, restore = _setup(schema)
    try:
        scored = _insert_word(conn, schema, "scored", difficulty=70.0, quizzable=True)
        unscored = _insert_word(conn, schema, "unscored")  # no word_difficulty row at all
        conn.commit()

        # No difficulty filter -> both words visible (LEFT JOIN, not INNER).
        res = client.get("/api/browse/words")
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert {"scored", "unscored"} <= lemmas

        # A difficulty filter active -> the unscored word can't satisfy a
        # range predicate against NULL, and is correctly excluded, not a bug.
        res = client.get("/api/browse/words", params={"difficulty_min": 0})
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert "scored" in lemmas
        assert "unscored" not in lemmas

        # Same for quizzable_only.
        res = client.get("/api/browse/words", params={"quizzable_only": True})
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert "scored" in lemmas
        assert "unscored" not in lemmas

        # difficulty-bands surfaces the unscored count explicitly rather
        # than silently dropping it.
        bands = client.get("/api/browse/difficulty-bands").json()
        unscored_band = next(b for b in bands if b["label"] == "Not yet scored")
        assert unscored_band["word_count"] == 1
    finally:
        restore()


@pg
def test_domain_bucket_counts_every_bucket_a_word_belongs_to():
    schema = "cc_test_browse_domains"
    client, conn, restore = _setup(schema)
    try:
        cat_society = _category(conn, schema, "S", "People Society Test")
        cat_science = _category(conn, schema, "F", "Nature Science Test")

        # A word tagged with categories in TWO different buckets should be
        # counted in both buckets' totals, not just its primary category's.
        dual = _insert_word(conn, schema, "dualdomain")
        _tag_domain(conn, schema, dual, cat_society, is_primary=True)
        _tag_domain(conn, schema, dual, cat_science, is_primary=False)

        single = _insert_word(conn, schema, "onedomain")
        _tag_domain(conn, schema, single, cat_society, is_primary=True)
        conn.commit()

        counts = {row["bucket"]: row["word_count"] for row in client.get("/api/browse/domains").json()}
        assert counts["people_society"] == 2  # dual + single
        assert counts["nature_science"] == 1  # dual only, even though not its primary
    finally:
        restore()


@pg
def test_anonymous_requests_are_refused():
    schema = "cc_test_browse_auth"
    from starlette.testclient import TestClient

    from webapp.backend import main

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
        assert client.get("/api/browse/words").status_code == 401
        assert client.get("/api/browse/authors").status_code == 401
        assert client.get("/api/browse/books").status_code == 401
        assert client.get("/api/browse/domains").status_code == 401
        assert client.get("/api/browse/difficulty-bands").status_code == 401
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()
