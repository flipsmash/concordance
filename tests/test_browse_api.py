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


def _insert_bulk_words(conn, schema, prefix, n):
    """N distinct plain words in one round trip -- for tests that need a
    book/author to clear a qualification floor (e.g. category-leaders' 50-
    words/book minimum) without a slow one-row-at-a-time loop."""
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.word (lemma, definition, part_of_speech, active)
                SELECT %s || gs::text, 'a definition', 'noun', true
                FROM generate_series(1, %s) AS gs
                RETURNING id""",
            (prefix, n),
        )
        return [r[0] for r in cur.fetchall()]


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


def _category(conn, schema, code, name="Test Category", level: int = 0):
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {schema}.category (taxonomy, code, name, level, assignable)
                VALUES ('usas', %s, %s, %s, true)
                ON CONFLICT (taxonomy, code) DO UPDATE SET name = EXCLUDED.name
                RETURNING id""",
            (code, name, level),
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
def test_exclusive_filter_book_vs_author_semantics():
    # "Exclusive to book X" means every word_book row for the word points to
    # X; "exclusive to author X" means every one points to a book BY X --
    # a stricter test for a book than an author, since a word appearing in
    # two books by the SAME author is exclusive to the author but not to
    # either individual book.
    schema = "cc_test_browse_exclusive"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Book A", author="Author One")
        b2 = _insert_book(conn, schema, "Book B", author="Author One")
        b3 = _insert_book(conn, schema, "Book C", author="Author Two")

        solo = _insert_word(conn, schema, "solo")  # only ever in Book A
        _link(conn, schema, solo, b1)

        same_author = _insert_word(conn, schema, "sameauthor")  # A and B, both Author One
        _link(conn, schema, same_author, b1)
        _link(conn, schema, same_author, b2)

        cross_author = _insert_word(conn, schema, "crossauthor")  # A and C, different authors
        _link(conn, schema, cross_author, b1)
        _link(conn, schema, cross_author, b3)
        conn.commit()

        # Exclusive to Book A alone: only "solo" qualifies -- the other two
        # both appear in at least one other book.
        res = client.get("/api/browse/words", params={"book_id": [b1], "exclusive": True})
        assert sorted(w["lemma"] for w in res.json()["items"]) == ["solo"]

        # Exclusive to Author One: "solo" AND "sameauthor" both qualify --
        # every book either appears in is by Author One -- but "crossauthor"
        # doesn't, since Book C is by Author Two.
        res2 = client.get("/api/browse/words", params={"author": "Author One", "exclusive": True})
        assert sorted(w["lemma"] for w in res2.json()["items"]) == ["sameauthor", "solo"]

        # Without exclusive=true, book_id=[b1] returns all three (the
        # baseline this filter narrows from).
        res3 = client.get("/api/browse/words", params={"book_id": [b1]})
        assert sorted(w["lemma"] for w in res3.json()["items"]) == ["crossauthor", "sameauthor", "solo"]
    finally:
        restore()


@pg
def test_exclusive_filter_composes_with_domain_filter():
    schema = "cc_test_browse_exclusive_domain"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Solo Book", author="Solo Author")
        b2 = _insert_book(conn, schema, "Other Book", author="Other Author")
        cat_science = _category(conn, schema, "F", "Nature Science Test")

        # Exclusive to b1 AND tagged -- matches both filters.
        target = _insert_word(conn, schema, "target")
        _link(conn, schema, target, b1)
        _tag_domain(conn, schema, target, cat_science)

        # Exclusive to b1 but untagged -- fails the domain filter alone.
        untagged = _insert_word(conn, schema, "untagged")
        _link(conn, schema, untagged, b1)

        # Tagged AND linked to b1, but ALSO in b2 -- fails exclusivity alone.
        # Without the exclusive filter actually being applied, this decoy
        # would incorrectly appear alongside "target" (domain+book_id alone
        # both match it), which is exactly what a silently-ignored `exclusive`
        # query param (e.g. an unrecognized FastAPI param) would let through.
        shared_but_tagged = _insert_word(conn, schema, "sharedbuttagged")
        _link(conn, schema, shared_but_tagged, b1)
        _link(conn, schema, shared_but_tagged, b2)
        _tag_domain(conn, schema, shared_but_tagged, cat_science)
        conn.commit()

        res = client.get(
            "/api/browse/words",
            params={"book_id": [b1], "exclusive": True, "domain": ["nature_science"]},
        )
        assert [w["lemma"] for w in res.json()["items"]] == ["target"]
    finally:
        restore()


@pg
def test_exclusive_shrinks_domain_summary_and_difficulty_bands_totals():
    schema = "cc_test_browse_exclusive_charts"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Book A", author="Author One")
        b2 = _insert_book(conn, schema, "Book B", author="Author One")

        solo = _insert_word(conn, schema, "solo", difficulty=50.0)
        _link(conn, schema, solo, b1)

        shared = _insert_word(conn, schema, "shared", difficulty=50.0)
        _link(conn, schema, shared, b1)
        _link(conn, schema, shared, b2)
        conn.commit()

        summary_all = client.get("/api/browse/domain-summary", params={"book_id": [b1]}).json()
        assert summary_all["total_words"] == 2

        summary_exclusive = client.get(
            "/api/browse/domain-summary", params={"book_id": [b1], "exclusive": True}
        ).json()
        assert summary_exclusive["total_words"] == 1

        bands_all = client.get("/api/browse/difficulty-bands", params={"book_id": [b1]}).json()
        assert sum(b["word_count"] for b in bands_all) == 2

        bands_exclusive = client.get(
            "/api/browse/difficulty-bands", params={"book_id": [b1], "exclusive": True}
        ).json()
        assert sum(b["word_count"] for b in bands_exclusive) == 1
    finally:
        restore()


@pg
def test_unique_word_histogram_buckets_book_and_author_scopes():
    schema = "cc_test_browse_uw_hist"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Book A", author="Author One")
        b2 = _insert_book(conn, schema, "Book B", author="Author One")
        b3 = _insert_book(conn, schema, "Book C", author="Author Two")

        # Book A: 2 solo words -> book bucket "2". Author One: 3 words total
        # exclusive to them (2 solo + 1 shared-with-own-other-book) -> "3-5".
        solo1 = _insert_word(conn, schema, "solo1")
        solo2 = _insert_word(conn, schema, "solo2")
        _link(conn, schema, solo1, b1)
        _link(conn, schema, solo2, b1)

        same_author = _insert_word(conn, schema, "sameauthor")
        _link(conn, schema, same_author, b1)
        _link(conn, schema, same_author, b2)

        # Book B on its own: 0 solo words (its only word is shared with A).
        # Book C: 1 word, exclusive to both the book and Author Two.
        cross = _insert_word(conn, schema, "crossauthorword")
        _link(conn, schema, cross, b3)
        conn.commit()

        book_hist = {r["label"]: r["count"] for r in
                     client.get("/api/browse/unique-word-histogram", params={"scope": "book"}).json()}
        assert book_hist["2"] == 1     # Book A: solo1 + solo2
        assert book_hist["0"] == 1     # Book B: its only word (sameauthor) isn't exclusive to it alone
        assert book_hist["1"] == 1     # Book C: crossauthorword

        author_hist = {r["label"]: r["count"] for r in
                       client.get("/api/browse/unique-word-histogram", params={"scope": "author"}).json()}
        assert author_hist["3-5"] == 1  # Author One: solo1+solo2+sameauthor = 3
        assert author_hist["1"] == 1    # Author Two: crossauthorword

        # Every named bucket is always present, even when empty.
        assert {r["label"] for r in
                client.get("/api/browse/unique-word-histogram", params={"scope": "book"}).json()} == \
               {"0", "1", "2", "3-5", "6-10", "11-25", "26-50", "51-100", "101+"}
    finally:
        restore()


@pg
def test_unique_word_bucket_filters_books_and_authors():
    # Same fixture as test_unique_word_histogram_buckets_book_and_author_scopes
    # (Book A: 2 solo words -> bucket "2"; Book B: 0 solo words -> bucket "0";
    # Book C: 1 solo word -> bucket "1"; Author One: 3 words total -> "3-5";
    # Author Two: 1 word -> "1"), but exercising the click-through filter on
    # /api/browse/books and /api/browse/authors instead of the histogram
    # endpoint -- the "0" bucket is the one that needs its own NOT EXISTS
    # branch (a book with zero exclusive words has no row at all in the
    # underlying aggregate, so a naive `HAVING count(*) >= 0` would never
    # match it).
    schema = "cc_test_browse_uw_filter"
    client, conn, restore = _setup(schema)
    try:
        b1 = _insert_book(conn, schema, "Book A", author="Author One")
        b2 = _insert_book(conn, schema, "Book B", author="Author One")
        b3 = _insert_book(conn, schema, "Book C", author="Author Two")

        solo1 = _insert_word(conn, schema, "solo1")
        solo2 = _insert_word(conn, schema, "solo2")
        _link(conn, schema, solo1, b1)
        _link(conn, schema, solo2, b1)

        same_author = _insert_word(conn, schema, "sameauthor")
        _link(conn, schema, same_author, b1)
        _link(conn, schema, same_author, b2)

        cross = _insert_word(conn, schema, "crossauthorword")
        _link(conn, schema, cross, b3)
        conn.commit()

        titles = [b["title"] for b in
                  client.get("/api/browse/books", params={"unique_word_bucket": "2"}).json()["items"]]
        assert titles == ["Book A"]

        titles0 = [b["title"] for b in
                   client.get("/api/browse/books", params={"unique_word_bucket": "0"}).json()["items"]]
        assert titles0 == ["Book B"]

        titles1 = [b["title"] for b in
                   client.get("/api/browse/books", params={"unique_word_bucket": "1"}).json()["items"]]
        assert titles1 == ["Book C"]

        authors35 = [a["author"] for a in
                     client.get("/api/browse/authors", params={"unique_word_bucket": "3-5"}).json()["items"]]
        assert authors35 == ["Author One"]

        authors1 = [a["author"] for a in
                    client.get("/api/browse/authors", params={"unique_word_bucket": "1"}).json()["items"]]
        assert authors1 == ["Author Two"]

        res = client.get("/api/browse/books", params={"unique_word_bucket": "not-a-real-bucket"})
        assert res.status_code == 404
    finally:
        restore()


@pg
def test_unique_word_histogram_and_bucket_filter_agree_despite_placeholder_authors():
    # Regression: the histogram's author-scope count and the click-through
    # list's count fell out of sync in production ("close but not the
    # same") because the histogram counted every distinct author, including
    # PLACEHOLDER_AUTHORS values ("Various", "Unknown Author", ...), while
    # browse_authors (the click-through target) already excludes those from
    # its own listing. A placeholder-authored book's word landing in some
    # bucket inflated that bucket's histogram count without a matching row
    # ever showing up in the filtered list.
    schema = "cc_test_browse_uw_placeholder"
    client, conn, restore = _setup(schema)
    try:
        real = _insert_book(conn, schema, "Real Book", author="Real Author")
        placeholder = _insert_book(conn, schema, "Anthology", author="Various")

        real_word = _insert_word(conn, schema, "realword")
        _link(conn, schema, real_word, real)

        placeholder_word = _insert_word(conn, schema, "placeholderword")
        _link(conn, schema, placeholder_word, placeholder)
        conn.commit()

        hist = {r["label"]: r["count"] for r in
                client.get("/api/browse/unique-word-histogram", params={"scope": "author"}).json()}
        # Only "Real Author" should ever be counted -- "Various" is a
        # placeholder, not a real author, and must not inflate any bucket.
        assert hist["1"] == 1

        authors1 = [a["author"] for a in
                    client.get("/api/browse/authors", params={"unique_word_bucket": "1"}).json()["items"]]
        assert authors1 == ["Real Author"]

        # The histogram's own count and the click-through list's length must
        # agree, not just both happen to exclude Various independently.
        assert hist["1"] == len(authors1)
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
def test_unscored_only_filters_words_and_domain_summary_but_rejects_with_difficulty_range():
    """unscored_only is what makes DifficultyHistogram's "Not yet scored" bar
    clickable (it can't be expressed as a difficulty_min/max range, since
    there's no numeric range that means "no row at all") -- verify both the
    word-list and domain-summary sides of that, plus the 400 guard that
    keeps it from silently overriding an explicit range."""
    schema = "cc_test_browse_unscored_only"
    client, conn, restore = _setup(schema)
    try:
        scored = _insert_word(conn, schema, "scored", difficulty=70.0)
        unscored = _insert_word(conn, schema, "unscored")
        conn.commit()

        res = client.get("/api/browse/words", params={"unscored_only": True})
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert lemmas == {"unscored"}

        res = client.get("/api/browse/words", params={"unscored_only": True, "difficulty_min": 0})
        assert res.status_code == 400

        summary = client.get("/api/browse/domain-summary", params={"unscored_only": True}).json()
        assert summary["total_words"] == 1
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
def test_category_overlap_at_parent_level():
    # Real USAS codes throughout -- children_of/bucket_for/subtree_match are
    # keyed to the hardcoded tagset in concordance/usas.py, not whatever's
    # in the test schema's own category table. A1/A2 are real direct
    # children of "A".
    schema = "cc_test_category_overlap_parent"
    client, conn, restore = _setup(schema)
    try:
        cat_a1 = _category(conn, schema, "A1", "General", level=1)
        cat_a2 = _category(conn, schema, "A2", "Affect", level=1)

        only_a1 = _insert_word(conn, schema, "onlya1")
        _tag_domain(conn, schema, only_a1, cat_a1)
        only_a2 = _insert_word(conn, schema, "onlya2")
        _tag_domain(conn, schema, only_a2, cat_a2)
        both = _insert_word(conn, schema, "bothcats")
        _tag_domain(conn, schema, both, cat_a1, is_primary=True)
        _tag_domain(conn, schema, both, cat_a2, is_primary=False)
        conn.commit()

        data = client.get("/api/browse/category-overlap?parent=A").json()
        sizes = {s["code"]: s["word_count"] for s in data["sizes"]}
        assert sizes["A1"] == 2  # onlya1 + bothcats
        assert sizes["A2"] == 2  # onlya2 + bothcats
        assert len(data["cells"]) == 1
        cell = data["cells"][0]
        assert {cell["code_a"], cell["code_b"]} == {"A1", "A2"}
        assert cell["shared_words"] == 1
        assert cell["ratio"] == pytest.approx(1 / 3, abs=1e-4)  # 1 shared / (2+2-1) union; endpoint rounds to 4dp

        # A1 (in this schema) has no children at all -- degrades to empty
        # cells, never an error or a broken single-cell grid.
        no_children = client.get("/api/browse/category-overlap?parent=A1").json()
        assert no_children["cells"] == []

        res = client.get("/api/browse/category-overlap?bucket=mind_language&parent=A")
        assert res.status_code == 400

        # all_top_code (used by the overlap heatmap's own off-diagonal
        # click) is an AND/intersection -- only "bothcats" carries a
        # category under BOTH A1 and A2. Deliberately NOT the same as
        # top_code, which is an OR and would also match onlya1/onlya2.
        words = client.get("/api/browse/words?all_top_code=A1&all_top_code=A2").json()
        assert {w["lemma"] for w in words["items"]} == {"bothcats"}
        either = client.get("/api/browse/words?top_code=A1&top_code=A2").json()
        assert either["total"] == 3  # onlya1 + onlya2 + bothcats -- the OR case, for contrast
    finally:
        restore()


@pg
def test_category_overlap_at_bucket_level():
    # "A" and "X" are both real members of the mind_language bucket.
    schema = "cc_test_category_overlap_bucket"
    client, conn, restore = _setup(schema)
    try:
        cat_a = _category(conn, schema, "A", "General & Abstract", level=0)
        cat_x = _category(conn, schema, "X", "Psychological Actions", level=0)

        a_word = _insert_word(conn, schema, "atagged")
        _tag_domain(conn, schema, a_word, cat_a)
        x_word = _insert_word(conn, schema, "xtagged")
        _tag_domain(conn, schema, x_word, cat_x)
        ax_word = _insert_word(conn, schema, "axtagged")
        _tag_domain(conn, schema, ax_word, cat_a, is_primary=True)
        _tag_domain(conn, schema, ax_word, cat_x, is_primary=False)
        conn.commit()

        data = client.get("/api/browse/category-overlap?bucket=mind_language").json()
        sizes = {s["code"]: s["word_count"] for s in data["sizes"]}
        assert sizes["A"] == 2  # atagged + axtagged
        assert sizes["X"] == 2  # xtagged + axtagged
        by_pair = {frozenset((c["code_a"], c["code_b"])): c for c in data["cells"]}
        assert by_pair[frozenset(("A", "X"))]["shared_words"] == 1  # axtagged only
    finally:
        restore()


@pg
def test_category_overlap_at_top_bucket_level():
    # "A" -> mind_language, "S" -> people_society (usas_domains.DOMAIN_BUCKETS).
    schema = "cc_test_category_overlap_top"
    client, conn, restore = _setup(schema)
    try:
        cat_a = _category(conn, schema, "A", "General & Abstract", level=0)
        cat_s = _category(conn, schema, "S", "Social Actions", level=0)

        a_word = _insert_word(conn, schema, "atagged")
        _tag_domain(conn, schema, a_word, cat_a)
        s_word = _insert_word(conn, schema, "stagged")
        _tag_domain(conn, schema, s_word, cat_s)
        as_word = _insert_word(conn, schema, "astagged")
        _tag_domain(conn, schema, as_word, cat_a, is_primary=True)
        _tag_domain(conn, schema, as_word, cat_s, is_primary=False)
        conn.commit()

        data = client.get("/api/browse/category-overlap").json()
        sizes = {s["code"]: s["word_count"] for s in data["sizes"]}
        assert sizes["mind_language"] == 2   # atagged + astagged
        assert sizes["people_society"] == 2  # stagged + astagged
        by_pair = {frozenset((c["code_a"], c["code_b"])): c for c in data["cells"]}
        assert by_pair[frozenset(("mind_language", "people_society"))]["shared_words"] == 1  # astagged only
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
def test_uncategorized_filters_words_and_difficulty_bands_but_rejects_with_domain():
    """uncategorized is what makes DomainDistribution's "Uncategorized" bar
    clickable (domain=uncategorized isn't a real DOMAIN_BUCKETS key and
    silently no-ops today otherwise) -- verify both the word-list and
    difficulty-bands sides of that, plus the 400 guard against combining it
    with an explicit domain."""
    schema = "cc_test_browse_uncategorized"
    client, conn, restore = _setup(schema)
    try:
        cat_society = _category(conn, schema, "S", "People Society Test")
        tagged = _insert_word(conn, schema, "tagged", difficulty=50.0)
        _tag_domain(conn, schema, tagged, cat_society)
        plain = _insert_word(conn, schema, "plain", difficulty=50.0)
        conn.commit()

        res = client.get("/api/browse/words", params={"uncategorized": True})
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert lemmas == {"plain"}

        res = client.get("/api/browse/words", params={"uncategorized": True, "domain": ["people_society"]})
        assert res.status_code == 400

        bands = client.get("/api/browse/difficulty-bands", params={"uncategorized": True}).json()
        band_50 = next(b for b in bands if b["label"] == "40-60")
        assert band_50["word_count"] == 1
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


# --- Categories section: top_code filter, category-counts, category-leaders ----

@pg
def test_top_code_filters_to_one_field_not_the_whole_bucket():
    # "S" (People) and "G" (Government) both live in the people_society
    # bucket -- domain=people_society must match both, but top_code=S is the
    # level-2 drill-down and must match only the "S" word.
    schema = "cc_test_browse_topcode"
    client, conn, restore = _setup(schema)
    try:
        cat_s = _category(conn, schema, "S", "People Test")
        cat_g = _category(conn, schema, "G", "Government Test")

        s_word = _insert_word(conn, schema, "peopleword")
        _tag_domain(conn, schema, s_word, cat_s)
        g_word = _insert_word(conn, schema, "govword")
        _tag_domain(conn, schema, g_word, cat_g)
        conn.commit()

        res = client.get("/api/browse/words", params={"domain": ["people_society"]})
        lemmas = {w["lemma"] for w in res.json()["items"]}
        assert lemmas == {"peopleword", "govword"}

        res2 = client.get("/api/browse/words", params={"top_code": ["S"]})
        lemmas2 = {w["lemma"] for w in res2.json()["items"]}
        assert lemmas2 == {"peopleword"}
    finally:
        restore()


@pg
def test_category_counts_scoped_by_bucket_and_unscoped():
    schema = "cc_test_browse_catcounts"
    client, conn, restore = _setup(schema)
    try:
        cat_people = _category(conn, schema, "S", "People Test")
        tagged = _insert_word(conn, schema, "peopleword")
        _tag_domain(conn, schema, tagged, cat_people)
        conn.commit()

        # Unscoped: all 21 USAS top-level fields, zero-count ones included --
        # usas.categories() is a fixed module-level constant, not read from
        # this (empty) test schema's own category table.
        all_counts = client.get("/api/browse/category-counts").json()
        assert len(all_counts) == 21
        by_code = {c["code"]: c for c in all_counts}
        assert by_code["S"]["word_count"] == 1
        assert by_code["S"]["bucket"] == "people_society"
        assert by_code["F"]["word_count"] == 0

        # Scoped to the bucket "S" belongs to: only that bucket's own codes.
        scoped = client.get("/api/browse/category-counts", params={"bucket": "people_society"}).json()
        assert {c["code"] for c in scoped} == {"B", "S", "G", "Z"}

        assert client.get("/api/browse/category-counts", params={"bucket": "not_a_real_bucket"}).status_code == 404
    finally:
        restore()


@pg
def test_category_leaders_ranks_by_lift_not_raw_count():
    # Regression target: _domain_vectors_to_map's own docstring found that a
    # raw-share argmax lands on one field for 82% of books, because some
    # fields are corpus-wide dominant everywhere -- ranking leaders by raw
    # category word COUNT has the analogous failure (the same prolific
    # author/long book tops every category's list). This asserts
    # /api/browse/category-leaders instead ranks by LIFT: a book that leans
    # harder on a category (bigger SHARE), even with a smaller raw count,
    # must outrank a book with a bigger raw count but a smaller share.
    schema = "cc_test_browse_leaders"
    client, conn, restore = _setup(schema)
    try:
        cat_f = _category(conn, schema, "F", "Food and Farming Test")

        prolific = _insert_book(conn, schema, "Prolific Tome", author="Prolific, Writer")
        focused = _insert_book(conn, schema, "Focused Pamphlet", author="Focused, Writer")

        # Both clear the 50-word/book qualification floor.
        prolific_words = _insert_bulk_words(conn, schema, "prolific", 200)
        focused_words = _insert_bulk_words(conn, schema, "focused", 50)
        for wid in prolific_words:
            _link(conn, schema, wid, prolific)
        for wid in focused_words:
            _link(conn, schema, wid, focused)

        # Prolific: 30/200 = 15% share, the bigger raw count (30 > 20).
        # Focused: 20/50 = 40% share, the smaller raw count but far bigger share.
        for wid in prolific_words[:30]:
            _tag_domain(conn, schema, wid, cat_f)
        for wid in focused_words[:20]:
            _tag_domain(conn, schema, wid, cat_f)
        conn.commit()

        res = client.get("/api/browse/category-leaders", params={"entity": "book", "top_code": "F"})
        assert res.status_code == 200, res.text
        data = res.json()
        titles = [item["label"] for item in data["items"]]
        assert titles[0] == "Focused Pamphlet", (
            f"expected the higher-SHARE book to rank first despite a lower raw "
            f"category word count, got order {titles}"
        )
        by_title = {item["label"]: item for item in data["items"]}
        assert by_title["Prolific Tome"]["category_word_count"] == 30
        assert by_title["Focused Pamphlet"]["category_word_count"] == 20
        assert by_title["Focused Pamphlet"]["share"] > by_title["Prolific Tome"]["share"]
        assert by_title["Focused Pamphlet"]["lift"] > by_title["Prolific Tome"]["lift"]
    finally:
        restore()


@pg
def test_category_leaders_requires_exactly_one_scope_param():
    schema = "cc_test_browse_leaders_scope"
    client, conn, restore = _setup(schema)
    try:
        assert client.get("/api/browse/category-leaders").status_code == 400
        both = client.get("/api/browse/category-leaders",
                           params={"bucket": "nature_science", "top_code": "F"})
        assert both.status_code == 400
        assert client.get("/api/browse/category-leaders", params={"bucket": "not_a_real_bucket"}).status_code == 404
    finally:
        restore()


# --- 4-level USAS drilldown: top_code/category-counts/category-leaders at any depth ----

@pg
def test_top_code_matches_exact_and_descendant_not_sibling():
    # Direct regression test for the bug this change fixes: top_code=I2 must
    # match a word tagged at I2 itself AND one tagged at the real descendant
    # I2.2, but top_code=A1 must NOT match A10 -- a sibling under "A", not a
    # child, despite "A10" naively starting with the string "A1".
    schema = "cc_test_browse_subtree"
    client, conn, restore = _setup(schema)
    try:
        cat_i2 = _category(conn, schema, "I2", "Business", level=1)
        cat_i22 = _category(conn, schema, "I2.2", "Business: Selling", level=2)
        cat_a10 = _category(conn, schema, "A10", "Open/closed", level=1)

        exact_word = _insert_word(conn, schema, "exactword")
        _tag_domain(conn, schema, exact_word, cat_i2)
        descendant_word = _insert_word(conn, schema, "descendantword")
        _tag_domain(conn, schema, descendant_word, cat_i22)
        sibling_word = _insert_word(conn, schema, "siblingword")
        _tag_domain(conn, schema, sibling_word, cat_a10)
        conn.commit()

        res = client.get("/api/browse/words", params={"top_code": ["I2"]})
        assert {w["lemma"] for w in res.json()["items"]} == {"exactword", "descendantword"}

        res2 = client.get("/api/browse/words", params={"top_code": ["A1"]})
        assert {w["lemma"] for w in res2.json()["items"]} == set()  # A10 must not match A1

        assert client.get("/api/browse/words", params={"top_code": ["not-a-real-code"]}).status_code == 404
    finally:
        restore()


@pg
def test_category_counts_parent_scoped_to_children_at_any_depth():
    schema = "cc_test_browse_catcounts_parent"
    client, conn, restore = _setup(schema)
    try:
        cat_i2 = _category(conn, schema, "I2", "Business", level=1)
        cat_i22 = _category(conn, schema, "I2.2", "Business: Selling", level=2)

        direct = _insert_word(conn, schema, "i2word")
        _tag_domain(conn, schema, direct, cat_i2)
        nested = _insert_word(conn, schema, "i22word")
        _tag_domain(conn, schema, nested, cat_i22)
        conn.commit()

        # parent=I -> I1..I4, and I2's own count includes the I2.2-tagged word
        # too (whole-subtree count, same policy left(c.code,1) already applied).
        res = client.get("/api/browse/category-counts", params={"parent": "I"})
        assert res.status_code == 200, res.text
        by_code = {c["code"]: c for c in res.json()}
        assert set(by_code) == {"I1", "I2", "I3", "I4"}
        assert by_code["I2"]["word_count"] == 2

        # parent=I2 -> I2.1/I2.2, with I2.2 itself getting the nested word.
        res2 = client.get("/api/browse/category-counts", params={"parent": "I2"}).json()
        by_code2 = {c["code"]: c for c in res2}
        assert set(by_code2) == {"I2.1", "I2.2"}
        assert by_code2["I2.2"]["word_count"] == 1

        # A real leaf has no children -> [].
        assert client.get("/api/browse/category-counts", params={"parent": "I2.2"}).json() == []

        # An unknown code -> 404; bucket+parent together -> 400.
        assert client.get("/api/browse/category-counts", params={"parent": "not-a-real-code"}).status_code == 404
        assert client.get("/api/browse/category-counts",
                           params={"bucket": "time_space_commerce", "parent": "I"}).status_code == 400
    finally:
        restore()


@pg
def test_category_counts_parent_level2_to_level3():
    # The least-exercised branch: one of the only 5 level-2 codes with any
    # real level-3 children at all.
    schema = "cc_test_browse_catcounts_level3"
    client, conn, restore = _setup(schema)
    try:
        cat_s111 = _category(conn, schema, "S1.1.1", "General", level=3)
        word = _insert_word(conn, schema, "s111word")
        _tag_domain(conn, schema, word, cat_s111)
        conn.commit()

        res = client.get("/api/browse/category-counts", params={"parent": "S1.1"}).json()
        by_code = {c["code"]: c for c in res}
        assert set(by_code) == {"S1.1.1", "S1.1.2", "S1.1.3", "S1.1.4"}
        assert by_code["S1.1.1"]["word_count"] == 1
        assert by_code["S1.1.2"]["word_count"] == 0
    finally:
        restore()


@pg
def test_category_leaders_top_code_counts_descendant_too():
    schema = "cc_test_browse_leaders_subtree"
    client, conn, restore = _setup(schema)
    try:
        cat_i2 = _category(conn, schema, "I2", "Business", level=1)
        cat_i22 = _category(conn, schema, "I2.2", "Business: Selling", level=2)

        book = _insert_book(conn, schema, "Business Book", author="Business, Writer")
        words = _insert_bulk_words(conn, schema, "bizword", 60)
        for wid in words:
            _link(conn, schema, wid, book)
        # 10 tagged directly at I2, 10 more at the descendant I2.2 -- both
        # must count toward top_code=I2's category_word_count (20 total).
        for wid in words[:10]:
            _tag_domain(conn, schema, wid, cat_i2)
        for wid in words[10:20]:
            _tag_domain(conn, schema, wid, cat_i22)
        conn.commit()

        res = client.get("/api/browse/category-leaders",
                          params={"entity": "book", "top_code": "I2", "min_words": 50})
        assert res.status_code == 200, res.text
        items = res.json()["items"]
        assert len(items) == 1
        assert items[0]["category_word_count"] == 20
    finally:
        restore()
