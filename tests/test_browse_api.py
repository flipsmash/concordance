"""Word-browsing API (author/book/domain/difficulty facets). DB-backed tests
run only when a throwaway Postgres is provided via CONCORDANCE_TEST_DB_URL
(else skipped) -- same convention as test_quiz_api.py, including its
main.SCHEMA-monkeypatch pattern for exercising real registered routes
against a disposable schema."""

from __future__ import annotations

import os
from urllib.parse import quote

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
def test_shared_word_does_not_cross_contaminate_author_or_book_filters():
    # Regression: browse_books(author=X) and browse_authors(book_id=Y) both
    # reused the word-anchored EXISTS filter meant for browse_words, which
    # correlates only to the word -- not to which book the outer row is
    # actually about. Since word_book is highly connected (most words appear
    # in many books), that silently returned every book/author that shares
    # so much as one common word with the target, regardless of authorship.
    # Confirmed in production: filtering books by "Shakespeare, William"
    # returned 497 of 500 results by other authors entirely.
    schema = "cc_test_browse_crosscontam"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Hamlet", author="Shakespeare, William")
        b2 = _insert_book(conn, schema, "The Prose Works of William Wordsworth", author="Wordsworth, William")
        shared = _insert_word(conn, schema, "shared")  # a common word both books happen to use
        _link(conn, schema, shared, b1)
        _link(conn, schema, shared, b2)
        conn.commit()

        res = client.get("/api/browse/books", params={"author": "Shakespeare, William"})
        titles = [b["title"] for b in res.json()["items"]]
        assert titles == ["Hamlet"], f"expected only Shakespeare's book, got {titles}"

        res2 = client.get("/api/browse/authors", params={"book_id": [b2]})
        authors = [a["author"] for a in res2.json()["items"]]
        assert authors == ["Wordsworth, William"], f"expected only Wordsworth, got {authors}"
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
        assert client.get("/api/browse/domain-summary").status_code == 401
        assert client.get("/api/browse/difficulty-bands").status_code == 401
    finally:
        main.SCHEMA = old_schema
        cleanup = db.connect(_URL)
        with cleanup.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        cleanup.commit()
        cleanup.close()


@pg
def test_book_stats_report_scored_count_mean_and_stddev():
    schema = "cc_test_browse_bookstats"
    client, conn, restore = _setup(schema)
    try:
        book = _insert_book(conn, schema, "Stats Book", author="Stats, Author")

        # Zero scored words -> mean/stddev both null, scored_word_count 0.
        unscored = _insert_word(conn, schema, "unscoredword")
        _link(conn, schema, unscored, book)
        conn.commit()
        row = client.get("/api/browse/books", params={"author": "Stats, Author"}).json()["items"][0]
        assert row["scored_word_count"] == 0
        assert row["mean_difficulty"] is None
        assert row["stddev_difficulty"] is None

        # Exactly one scored word -> mean is that value, stddev is null
        # (STDDEV_SAMP is undefined at N=1, not 0 -- 0 would misleadingly
        # read as "no variation" instead of "not enough data").
        one_scored = _insert_word(conn, schema, "onescored", difficulty=40.0)
        _link(conn, schema, one_scored, book)
        conn.commit()
        row = client.get("/api/browse/books", params={"author": "Stats, Author"}).json()["items"][0]
        assert row["scored_word_count"] == 1
        assert row["mean_difficulty"] == 40.0
        assert row["stddev_difficulty"] is None

        # Two scored words -> both mean and stddev are real numbers.
        two_scored = _insert_word(conn, schema, "twoscored", difficulty=60.0)
        _link(conn, schema, two_scored, book)
        conn.commit()
        row = client.get("/api/browse/books", params={"author": "Stats, Author"}).json()["items"][0]
        assert row["scored_word_count"] == 2
        assert row["mean_difficulty"] == 50.0  # (40 + 60) / 2
        assert row["stddev_difficulty"] is not None and row["stddev_difficulty"] > 0
        assert row["word_count"] == 3  # unscored word still counts toward total entries
    finally:
        restore()


@pg
def test_books_endpoint_filters_by_book_id():
    # The work-detail page needs to look up one specific book's title/author/
    # stats by id -- every other browse endpoint already accepts book_id as a
    # filter; browse_books was the one exception.
    schema = "cc_test_browse_bookid"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Wanted", author="Author, Some")
        b2 = _insert_book(conn, schema, "Unwanted", author="Author, Some")
        w1 = _insert_word(conn, schema, "wordone")
        w2 = _insert_word(conn, schema, "wordtwo")
        _link(conn, schema, w1, b1)
        _link(conn, schema, w2, b2)
        conn.commit()

        res = client.get("/api/browse/books", params={"book_id": [b1]})
        titles = [b["title"] for b in res.json()["items"]]
        assert titles == ["Wanted"]
    finally:
        restore()


@pg
def test_domain_summary_includes_uncategorized_and_correct_total():
    schema = "cc_test_browse_domainsummary"
    client, conn, restore = _setup(schema)
    try:
        book = _insert_book(conn, schema, "Summary Book")
        cat_society = _category(conn, schema, "S", "People Society Test")
        cat_science = _category(conn, schema, "F", "Nature Science Test")

        dual = _insert_word(conn, schema, "dualdomain")  # tagged in 2 buckets
        _tag_domain(conn, schema, dual, cat_society, is_primary=True)
        _tag_domain(conn, schema, dual, cat_science, is_primary=False)
        plain = _insert_word(conn, schema, "notagsword")  # zero categories

        for w in (dual, plain):
            _link(conn, schema, w, book)
        conn.commit()

        data = client.get("/api/browse/domain-summary", params={"book_id": [book]}).json()
        assert data["total_words"] == 2

        by_bucket = {b["bucket"]: b["word_count"] for b in data["buckets"]}
        assert by_bucket["people_society"] == 1
        assert by_bucket["nature_science"] == 1
        assert by_bucket["uncategorized"] == 1  # only "plain", not "dual"
        assert "uncategorized" in [b["bucket"] for b in data["buckets"]]

        # /api/browse/domains itself must stay a bare list of the 6 named
        # buckets only -- no uncategorized entry -- since the faceted Browse
        # page's domain-chip click handler depends on that exact shape.
        plain_domains = client.get("/api/browse/domains", params={"book_id": [book]}).json()
        assert isinstance(plain_domains, list)
        assert "uncategorized" not in [b["bucket"] for b in plain_domains]
        assert len(plain_domains) == 6
    finally:
        restore()


@pg
def test_book_related_returns_precomputed_neighbors_sorted_by_score():
    schema = "cc_test_book_related"
    client, conn, restore = _setup(schema)
    try:
        from concordance import db as _db

        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        # A 4th book with its own disjoint vocabulary: needed so N=4 and
        # compute_book_similarity's max_df_fraction (default 0.5, so
        # max_df=2) doesn't exclude the book_a/book_b shared words (df=2)
        # from scoring entirely -- with only 3 books this test's own shared
        # words would score zero everything, a real bug this exact test
        # exposed once (see the db.compute_book_similarity commit history:
        # an early-return-without-commit case this same 2-3-book scale
        # triggered live, hanging a completely unrelated connection's
        # DROP SCHEMA for 10+ minutes).
        book_d = _insert_book(conn, schema, "Book D", author="Author D")
        _link(conn, schema, _insert_word(conn, schema, "bookdword"), book_d)
        # book_a/book_b share 3 rare words; book_a/book_c share only 1 --
        # compute_book_similarity's own min_shared_words default (3) means
        # only the book_a/book_b pair should be precomputed and returned.
        for i in range(3):
            w = _insert_word(conn, schema, f"shared{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)
        w_c = _insert_word(conn, schema, "onlyc")
        _link(conn, schema, w_c, book_a)
        _link(conn, schema, w_c, book_c)
        conn.commit()

        _db.compute_book_similarity(conn, schema)

        resp = client.get(f"/api/browse/books/{book_a}/related")
        assert resp.status_code == 200
        data = resp.json()

        assert data["center"]["id"] == book_a
        assert data["center"]["ring"] == 0
        assert data["center"]["word_count"] == 4  # 3 shared + onlyc

        related_ids = [n["id"] for n in data["nodes"] if n["ring"] == 1]
        assert related_ids == [book_b]  # book_c excluded -- below min_shared_words

        assert len(data["edges"]) == 1
        edge = data["edges"][0]
        assert edge["source"] == book_a and edge["target"] == book_b
        assert edge["shared_word_count"] == 3
        assert edge["score"] > 0

        # Unknown book -> 404, not a silent empty response.
        assert client.get("/api/browse/books/999999/related").status_code == 404
    finally:
        restore()


@pg
def test_book_related_includes_cross_links_between_neighbors():
    schema = "cc_test_book_cross_links"
    client, conn, restore = _setup(schema)
    try:
        from concordance import db as _db

        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        book_d = _insert_book(conn, schema, "Book D", author="Author D")
        _link(conn, schema, _insert_word(conn, schema, "fillerword"), book_d)

        # A-B, A-C, AND B-C each share 3 words of their own (disjoint sets) --
        # so B and C should each be A's neighbor AND be linked to each other,
        # turning book_related(A) from a star into a real (small) graph.
        for i in range(3):
            w = _insert_word(conn, schema, f"ab{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)
        for i in range(3):
            w = _insert_word(conn, schema, f"ac{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_c)
        for i in range(3):
            w = _insert_word(conn, schema, f"bc{i}")
            _link(conn, schema, w, book_b)
            _link(conn, schema, w, book_c)
        conn.commit()

        _db.compute_book_similarity(conn, schema)

        resp = client.get(f"/api/browse/books/{book_a}/related")
        assert resp.status_code == 200
        data = resp.json()

        related_ids = {n["id"] for n in data["nodes"] if n["ring"] == 1}
        assert related_ids == {book_b, book_c}

        center_edges = [e for e in data["edges"] if e["is_center_edge"]]
        cross_edges = [e for e in data["edges"] if not e["is_center_edge"]]
        assert len(center_edges) == 2  # A-B, A-C
        assert len(cross_edges) == 1   # B-C, surfaced from book_similarity's own stored row
        edge = cross_edges[0]
        assert {edge["source"], edge["target"]} == {book_b, book_c}
        assert edge["shared_word_count"] == 3
    finally:
        restore()


@pg
def test_book_shared_words_returns_overlap_sorted_by_idf():
    schema = "cc_test_book_shared_words"
    client, conn, restore = _setup(schema)
    try:
        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        book_d = _insert_book(conn, schema, "Book D", author="Author D")

        # Common word in all 4 books: df=4, max_df_fraction=0.5 * 4 = 2 --
        # excluded from "the what" even though it's technically shared,
        # same cutoff the similarity score itself uses.
        common = _insert_word(conn, schema, "commonword")
        for b in (book_a, book_b, book_c, book_d):
            _link(conn, schema, common, b)

        # Two rare words shared only by A and B (df=2, passes the cutoff).
        rare1 = _insert_word(conn, schema, "rareone")
        rare2 = _insert_word(conn, schema, "raretwo")
        for w in (rare1, rare2):
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)

        # Word only in A, not shared -- must not appear.
        _link(conn, schema, _insert_word(conn, schema, "onlya"), book_a)
        conn.commit()

        resp = client.get(f"/api/browse/books/{book_a}/shared-words/{book_b}")
        assert resp.status_code == 200
        data = resp.json()

        lemmas = {w["lemma"] for w in data["shared_words"]}
        assert lemmas == {"rareone", "raretwo"}
        assert data["total_shared"] == 2
        idfs = [w["idf"] for w in data["shared_words"]]
        assert idfs == sorted(idfs, reverse=True)
    finally:
        restore()


@pg
def test_author_related_returns_neighbors_sorted_by_score():
    schema = "cc_test_author_related"
    client, conn, restore = _setup(schema)
    try:
        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        # A 4th, disjoint author -- needed so N_authors=4 and
        # _author_similarity_candidates' max_df_fraction (default 0.5, so
        # max_df=2) doesn't exclude the Author A/B shared words (author-df=2)
        # entirely -- same reasoning as book_related's own test.
        book_d = _insert_book(conn, schema, "Book D", author="Author D")
        _link(conn, schema, _insert_word(conn, schema, "bookdword"), book_d)

        # Author A/B share 3 words; Author A/C share only 1 -- default
        # min_shared_words=3 means only A/B should come back as related.
        for i in range(3):
            w = _insert_word(conn, schema, f"shared{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)
        w_c = _insert_word(conn, schema, "onlyc")
        _link(conn, schema, w_c, book_a)
        _link(conn, schema, w_c, book_c)
        conn.commit()

        from concordance import db as _db
        _db.compute_author_similarity(conn, schema)

        resp = client.get(f"/api/browse/authors/{quote('Author A')}/related")
        assert resp.status_code == 200
        data = resp.json()

        assert data["center"]["id"] == "Author A"
        assert data["center"]["ring"] == 0
        assert data["center"]["book_count"] == 1
        assert data["center"]["word_count"] == 4  # 3 shared + onlyc

        related_ids = [n["id"] for n in data["nodes"] if n["ring"] == 1]
        assert related_ids == ["Author B"]  # Author C excluded -- below min_shared_words

        assert len(data["edges"]) == 1
        edge = data["edges"][0]
        assert edge["source"] == "Author A" and edge["target"] == "Author B"
        assert edge["shared_word_count"] == 3
        # All 4 of Author A's words here have author-df=2 (shared with
        # exactly one other author each), so every word's idf is identical
        # and the cosine collapses to sqrt(3)/2 -- see
        # compute_author_similarity's docstring in db.py for why author-df
        # (not book-df) is the denominator that makes this number
        # meaningfully different from book_related's metric.
        assert edge["score"] == pytest.approx(0.8660254, abs=1e-4)

        # Unknown author -> 404, not a silent empty response.
        assert client.get(f"/api/browse/authors/{quote('Nobody')}/related").status_code == 404
    finally:
        restore()


@pg
def test_author_related_includes_cross_links_between_neighbors():
    schema = "cc_test_author_cross_links"
    client, conn, restore = _setup(schema)
    try:
        from concordance import db as _db

        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        book_d = _insert_book(conn, schema, "Book D", author="Author D")
        _link(conn, schema, _insert_word(conn, schema, "fillerword"), book_d)

        for i in range(3):
            w = _insert_word(conn, schema, f"ab{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)
        for i in range(3):
            w = _insert_word(conn, schema, f"ac{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_c)
        for i in range(3):
            w = _insert_word(conn, schema, f"bc{i}")
            _link(conn, schema, w, book_b)
            _link(conn, schema, w, book_c)
        conn.commit()

        _db.compute_author_similarity(conn, schema)

        resp = client.get(f"/api/browse/authors/{quote('Author A')}/related")
        assert resp.status_code == 200
        data = resp.json()

        related_ids = {n["id"] for n in data["nodes"] if n["ring"] == 1}
        assert related_ids == {"Author B", "Author C"}

        center_edges = [e for e in data["edges"] if e["is_center_edge"]]
        cross_edges = [e for e in data["edges"] if not e["is_center_edge"]]
        assert len(center_edges) == 2
        assert len(cross_edges) == 1
        edge = cross_edges[0]
        assert {edge["source"], edge["target"]} == {"Author B", "Author C"}
        assert edge["shared_word_count"] == 3
    finally:
        restore()


@pg
def test_author_shared_words_returns_overlap_sorted_by_idf():
    schema = "cc_test_author_shared_words"
    client, conn, restore = _setup(schema)
    try:
        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        book_d = _insert_book(conn, schema, "Book D", author="Author D")

        common = _insert_word(conn, schema, "commonword")
        for b in (book_a, book_b, book_c, book_d):
            _link(conn, schema, common, b)

        rare1 = _insert_word(conn, schema, "rareone")
        rare2 = _insert_word(conn, schema, "raretwo")
        for w in (rare1, rare2):
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)

        _link(conn, schema, _insert_word(conn, schema, "onlya"), book_a)
        conn.commit()

        resp = client.get(f"/api/browse/authors/{quote('Author A')}/shared-words/{quote('Author B')}")
        assert resp.status_code == 200
        data = resp.json()

        lemmas = {w["lemma"] for w in data["shared_words"]}
        assert lemmas == {"rareone", "raretwo"}
        assert data["total_shared"] == 2
        idfs = [w["idf"] for w in data["shared_words"]]
        assert idfs == sorted(idfs, reverse=True)
    finally:
        restore()


@pg
def test_authors_relatedness_global_graph_dedupes_mutual_edges():
    schema = "cc_test_authors_relatedness"
    client, conn, restore = _setup(schema)
    try:
        book_a = _insert_book(conn, schema, "Book A", author="Author A")
        book_b = _insert_book(conn, schema, "Book B", author="Author B")
        book_c = _insert_book(conn, schema, "Book C", author="Author C")
        book_d = _insert_book(conn, schema, "Book D", author="Author D")
        _link(conn, schema, _insert_word(conn, schema, "bookdword"), book_d)
        for i in range(3):
            w = _insert_word(conn, schema, f"shared{i}")
            _link(conn, schema, w, book_a)
            _link(conn, schema, w, book_b)
        # Author C needs at least one linked word too -- otherwise only 3
        # authors (A, B, D) have any active vocabulary, dropping n_authors
        # to 3 and making max_df_fraction (0.5 * 3 = 1.5) exclude the A/B
        # shared words (author-df=2) entirely, same trap the per-author test
        # and book_related's own test both had to route around.
        w_c = _insert_word(conn, schema, "onlyc")
        _link(conn, schema, w_c, book_a)
        _link(conn, schema, w_c, book_c)
        conn.commit()

        from concordance import db as _db
        _db.compute_author_similarity(conn, schema)

        resp = client.get("/api/browse/authors/relatedness")
        assert resp.status_code == 200
        data = resp.json()

        node_ids = {n["id"] for n in data["nodes"]}
        assert node_ids == {"Author A", "Author B", "Author C", "Author D"}

        # Author A and Author B are mutual top-k neighbors of each other --
        # the edge must appear exactly once, not once per direction (cosine
        # similarity is symmetric, so a naive per-author candidate dump
        # would double it).
        matching = [
            e for e in data["edges"]
            if {e["source"], e["target"]} == {"Author A", "Author B"}
        ]
        assert len(matching) == 1
        assert matching[0]["shared_word_count"] == 3
    finally:
        restore()


@pg
def test_authors_map_returns_precomputed_clusters():
    schema = "cc_test_authors_map"
    client, conn, restore = _setup(schema)
    try:
        from concordance import db as _db

        alpha_authors = [f"Alpha{i}" for i in range(5)]
        beta_authors = [f"Beta{i}" for i in range(5)]
        alpha_words = [_insert_word(conn, schema, f"alphaword{i}") for i in range(10)]
        beta_words = [_insert_word(conn, schema, f"betaword{i}") for i in range(10)]
        for i, author in enumerate(alpha_authors):
            book = _insert_book(conn, schema, f"Alpha Book {i}", author=author)
            for w in alpha_words:
                _link(conn, schema, w, book)
        for i, author in enumerate(beta_authors):
            book = _insert_book(conn, schema, f"Beta Book {i}", author=author)
            for w in beta_words:
                _link(conn, schema, w, book)
        conn.commit()

        _db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)

        resp = client.get("/api/browse/authors/map")
        assert resp.status_code == 200
        data = resp.json()

        by_author = {n["author"]: n for n in data["nodes"]}
        assert set(by_author.keys()) == set(alpha_authors) | set(beta_authors)

        alpha_clusters = {by_author[a]["cluster_id"] for a in alpha_authors}
        beta_clusters = {by_author[b]["cluster_id"] for b in beta_authors}
        assert len(alpha_clusters) == 1
        assert len(beta_clusters) == 1
        assert alpha_clusters != beta_clusters
        assert all(isinstance(by_author[a]["x"], float) for a in alpha_authors)
        assert by_author["Alpha0"]["book_count"] == 1
    finally:
        restore()


@pg
def test_authors_matrix_and_dendrogram_read_the_same_clustering_run():
    schema = "cc_test_authors_matrix_dendrogram"
    client, conn, restore = _setup(schema)
    try:
        from concordance import db as _db

        alpha_authors = [f"Alpha{i}" for i in range(5)]
        beta_authors = [f"Beta{i}" for i in range(5)]
        alpha_words = [_insert_word(conn, schema, f"alphaword{i}") for i in range(10)]
        beta_words = [_insert_word(conn, schema, f"betaword{i}") for i in range(10)]
        for i, author in enumerate(alpha_authors):
            book = _insert_book(conn, schema, f"Alpha Book {i}", author=author)
            for w in alpha_words:
                _link(conn, schema, w, book)
        for i, author in enumerate(beta_authors):
            book = _insert_book(conn, schema, f"Beta Book {i}", author=author)
            for w in beta_words:
                _link(conn, schema, w, book)
        conn.commit()

        _db.compute_author_clustering(conn, schema, top_n=200, n_clusters=2)

        matrix_resp = client.get("/api/browse/authors/matrix")
        assert matrix_resp.status_code == 200
        matrix = matrix_resp.json()
        assert len(matrix["authors"]) == 10
        assert len(matrix["grid"]) == 10 and len(matrix["grid"][0]) == 10
        # An author compared with itself: perfect overlap.
        self_idx = matrix["authors"].index("Alpha0")
        assert matrix["grid"][self_idx][self_idx]["score"] == pytest.approx(1.0, abs=1e-6)

        dendro_resp = client.get("/api/browse/authors/dendrogram")
        assert dendro_resp.status_code == 200
        dendro = dendro_resp.json()
        assert set(dendro["leaf_order"]) == set(alpha_authors) | set(beta_authors)
        # Same leaf set as the matrix -- both endpoints reading the same
        # author_cluster_run row, not two different computations.
        assert set(dendro["leaf_order"]) == set(matrix["authors"])
        assert dendro["tree"]["size"] == 10
        assert dendro["tree"]["left"] is not None and dendro["tree"]["right"] is not None
    finally:
        restore()
