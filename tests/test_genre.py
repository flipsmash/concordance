"""Genre classifier code-validation, redundancy filter, and the
classify_and_store_genres chunk/commit/resume/RDF-fetch plumbing.
No-model tests need no DB; classify_and_store_genres tests need a
disposable Postgres (same CONCORDANCE_TEST_DB_URL convention as
test_classify.py)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from concordance import db, genre

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


# --- apply_redundancy_filter --------------------------------------------------

def test_redundancy_filter_drops_fiction_when_specific_fiction_tag_present():
    assert genre.apply_redundancy_filter(["Fiction", "Science Fiction"]) == ["Science Fiction"]


def test_redundancy_filter_drops_nonfiction_when_specific_nonfiction_tag_present():
    assert genre.apply_redundancy_filter(["Nonfiction", "Biography"]) == ["Biography"]


def test_redundancy_filter_keeps_bare_fiction_when_nothing_more_specific():
    # No specific fiction tag alongside it -- nothing to prefer over the bare label.
    assert genre.apply_redundancy_filter(["Fiction", "Young Adult"]) == ["Fiction", "Young Adult"]


def test_redundancy_filter_never_touches_format_audience_or_canon_tags():
    # Graphic Novels/Children's/Classics are orthogonal axes -- a graphic
    # memoir (Persepolis) should keep Memoir, drop the redundant
    # Nonfiction, but never lose Graphic Novels/Classics to the filter.
    result = genre.apply_redundancy_filter(["Nonfiction", "Memoir", "Graphic Novels", "Classics"])
    assert set(result) == {"Memoir", "Graphic Novels", "Classics"}


def test_redundancy_filter_leaves_cross_cutting_theme_tags_alone():
    # Christian/Gay and Lesbian/Music deliberately sit outside both
    # redundancy sets -- each spans fiction and nonfiction in practice.
    result = genre.apply_redundancy_filter(["Fiction", "Christian", "Romance"])
    assert set(result) == {"Christian", "Romance"}


def test_redundancy_filter_is_a_noop_without_a_generic_tag():
    assert genre.apply_redundancy_filter(["Science Fiction", "Fantasy"]) == ["Science Fiction", "Fantasy"]


# --- _validate -----------------------------------------------------------------

def test_validate_normalizes_case_to_canonical_form():
    assert genre._validate(["science fiction", "MEMOIR"]) == ["Science Fiction", "Memoir"]


def test_validate_drops_unrecognized_labels():
    assert genre._validate(["Science Fiction", "Not A Real Genre"]) == ["Science Fiction"]


def test_validate_dedupes_preserving_order():
    assert genre._validate(["Fantasy", "Fantasy", "Horror"]) == ["Fantasy", "Horror"]


def test_validate_ignores_non_list():
    assert genre._validate("Fantasy") == []
    assert genre._validate(None) == []


# --- _parse (identical shape to classify._parse, own copy since these modules
# don't share code -- smoke-tested here rather than assuming) -----------------

def test_parse_bare_array():
    assert genre._parse('[{"b":1,"g":["Fantasy"]}]') == [{"b": 1, "g": ["Fantasy"]}]


def test_parse_strips_fence_and_trailing_prose():
    raw = '```json\n[{"b":1,"g":["Fantasy"]}]\n```\ndone'
    assert genre._parse(raw) == [{"b": 1, "g": ["Fantasy"]}]


def test_parse_garbage_returns_empty():
    assert genre._parse("I cannot comply") == []


# --- classify_and_store_genres (DB-backed) --------------------------------------

class _FakeGenreClassifier:
    """Deterministic stand-in for GenreClassifier -- no LLM, no GPU. Same
    rationale as test_classify.py's _FakeClassifier: a real GenreClassifier()
    would try to load the 14B model."""

    def __init__(self, cfg=None):
        self.batch = 8
        self.seen_chunks: list[list[int]] = []
        self.closed = False
        self.tags_by_book: dict[int, list[str]] = {}

    def classify(self, items):
        self.seen_chunks.append([it["_id"] for it in items])
        return {it["_id"]: self.tags_by_book.get(it["_id"], ["Fantasy"]) for it in items}

    def close(self):
        self.closed = True


def _seed_books(conn, schema, n, *, archive_path=None, publication_year=None):
    db.apply_schema(conn, schema)
    ids = []
    with conn.cursor() as cur:
        for i in range(n):
            cur.execute(
                f"""INSERT INTO {schema}.book (title, author, archive_path, publication_year)
                    VALUES (%s, %s, %s, %s) RETURNING id""",
                (f"Book {i}", f"Author {i}", archive_path, publication_year),
            )
            ids.append(cur.fetchone()[0])
    conn.commit()
    return ids


@pg
def test_classify_and_store_genres_commits_in_chunks_not_once_at_the_end(monkeypatch):
    schema = "cc_test_genre_chunking"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    _seed_books(conn, schema, 25)

    fake = _FakeGenreClassifier()
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    stats = genre.classify_and_store_genres(conn, schema, only_missing=True, commit_every=10, fetch_rdf=False)
    assert stats["books"] == 25
    assert stats["classified"] == 25
    assert [len(c) for c in fake.seen_chunks] == [10, 10, 5]
    assert fake.closed

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.book_genre")
        assert cur.fetchone()[0] == 25
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_genres_only_missing_resumes_after_a_partial_run(monkeypatch):
    schema = "cc_test_genre_resume"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    ids = _seed_books(conn, schema, 10)
    already_done = ids[:4]
    with conn.cursor() as cur:
        for book_id in already_done:
            cur.execute(f"INSERT INTO {schema}.book_genre (book_id, genre, source) VALUES (%s,'Fantasy','llm')",
                        (book_id,))
    conn.commit()

    fake = _FakeGenreClassifier()
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    stats = genre.classify_and_store_genres(conn, schema, only_missing=True, commit_every=10, fetch_rdf=False)
    assert stats["books"] == 6
    assert stats["classified"] == 6
    seen_ids = {b for chunk in fake.seen_chunks for b in chunk}
    assert not (seen_ids & set(already_done))

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.book_genre")
        assert cur.fetchone()[0] == 10
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_genres_skips_a_book_deleted_mid_run(monkeypatch):
    schema = "cc_test_genre_vanished"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    ids = _seed_books(conn, schema, 20)
    vanishing_id = ids[15]

    class _DeletingClassifier(_FakeGenreClassifier):
        def classify(self, items):
            if len(self.seen_chunks) == 0:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {schema}.book WHERE id=%s", (vanishing_id,))
                conn.commit()
            return super().classify(items)

    fake = _DeletingClassifier()
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    stats = genre.classify_and_store_genres(conn, schema, only_missing=True, commit_every=10, fetch_rdf=False)
    assert stats["vanished"] == 1
    assert stats["books"] == 20
    assert stats["classified"] == 19

    with conn.cursor() as cur:
        cur.execute(f"select count(*) from {schema}.book_genre where book_id=%s", (vanishing_id,))
        assert cur.fetchone()[0] == 0
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_genres_applies_redundancy_filter_end_to_end(monkeypatch):
    # The fake classifier returns RAW, unfiltered tags (as if _query's own
    # internal filtering were bypassed or absent) -- classify_and_store_genres
    # itself must still apply apply_redundancy_filter before writing, not
    # merely trust that whatever classify() returned was already clean.
    schema = "cc_test_genre_redundancy_e2e"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()
    ids = _seed_books(conn, schema, 1)

    fake = _FakeGenreClassifier()
    fake.tags_by_book = {ids[0]: ["Fiction", "Mystery"]}  # raw, redundant -- not pre-filtered
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    genre.classify_and_store_genres(conn, schema, only_missing=True, fetch_rdf=False)

    with conn.cursor() as cur:
        cur.execute(f"select genre from {schema}.book_genre where book_id=%s order by genre", (ids[0],))
        assert [r[0] for r in cur.fetchall()] == ["Mystery"]
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_genres_fetches_rdf_hints_and_backfills_publication_year(monkeypatch, tmp_path):
    # Combined-fetch behavior: a book with an archive_path but no
    # publication_year gets both its genre hints AND its publication_year
    # from one mocked RDF fetch -- the whole reason fetch_gutenberg_rdf
    # exists instead of reusing fetch_publication_info.
    schema = "cc_test_genre_rdf_combined"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()

    book_file = tmp_path / "book.txt"
    book_file.write_text("*** START OF THE PROJECT GUTENBERG EBOOK ***\n[eBook #999]\nSome text.\n")
    ids = _seed_books(conn, schema, 1, archive_path=str(book_file), publication_year=None)

    def fake_fetch_gutenberg_rdf(gutenberg_id, timeout=15.0):
        assert gutenberg_id == 999
        return {"publication_year": 1888, "publication_era": "late 19th century",
                "genre_hints": ["Science fiction", "Category: Novels"]}

    monkeypatch.setattr(genre, "fetch_gutenberg_rdf", fake_fetch_gutenberg_rdf)

    fake = _FakeGenreClassifier()
    captured_hints = {}

    def classify_and_capture(items):
        for it in items:
            captured_hints[it["_id"]] = it["hints"]
        fake.seen_chunks.append([it["_id"] for it in items])
        return {it["_id"]: ["Science Fiction"] for it in items}

    fake.classify = classify_and_capture
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    stats = genre.classify_and_store_genres(conn, schema, only_missing=True, delay=0, fetch_rdf=True)
    assert stats["rdf_fetched"] == 1
    assert stats["publication_info_backfilled"] == 1
    assert captured_hints[ids[0]] == ["Science fiction", "Category: Novels"]

    with conn.cursor() as cur:
        cur.execute(f"select publication_year, publication_era from {schema}.book where id=%s", (ids[0],))
        assert cur.fetchone() == (1888, "late 19th century")
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()


@pg
def test_classify_and_store_genres_does_not_overwrite_an_existing_publication_year(monkeypatch, tmp_path):
    schema = "cc_test_genre_rdf_no_clobber"
    conn = db.connect(_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    conn.commit()

    book_file = tmp_path / "book.txt"
    book_file.write_text("*** START OF THE PROJECT GUTENBERG EBOOK ***\n[eBook #999]\nSome text.\n")
    ids = _seed_books(conn, schema, 1, archive_path=str(book_file), publication_year=1900)

    def fake_fetch_gutenberg_rdf(gutenberg_id, timeout=15.0):
        return {"publication_year": 1888, "publication_era": "late 19th century", "genre_hints": []}

    monkeypatch.setattr(genre, "fetch_gutenberg_rdf", fake_fetch_gutenberg_rdf)
    fake = _FakeGenreClassifier()
    monkeypatch.setattr(genre, "GenreClassifier", lambda cfg=None: fake)

    stats = genre.classify_and_store_genres(conn, schema, only_missing=True, delay=0, fetch_rdf=True)
    assert stats["publication_info_backfilled"] == 0

    with conn.cursor() as cur:
        cur.execute(f"select publication_year from {schema}.book where id=%s", (ids[0],))
        assert cur.fetchone() == (1900,)  # untouched
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    conn.commit()
    conn.close()
