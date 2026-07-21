"""End-user word-browsing API (§ word browsing) -- author/book/domain/
difficulty facets, freely combinable in any order, plus an A-Z jump, an
archaic-ness filter, text search, and a random-word picker.

Imports `main` as a module and always accesses `_main.SCHEMA`/
`_main.get_conn()`/`_main.require_viewer` via dotted attribute lookup, not a
bare `from ... import`, for the same reason quiz.py does: tests monkeypatch
`main.SCHEMA` after import, and a bare import would freeze the value before
that monkeypatch runs. Registered into `app` at the bottom of main.py, after
get_conn/SCHEMA/require_viewer are all defined, for the same ordering reason
quiz.py's router is.

Every endpoint here is `require_viewer` -- this is the end-user browsing
surface, distinct from /api/words's `require_admin` curation view (different
audience, different response shape: no `rescued_from_reject` etc.).

--- The dedup rule (see the word-browsing plan) ---

word_book and word_category are both many-to-many against word (62% of words
appear in more than one book). Two situations, two different SQL shapes:

  - FILTERING through the junction table (author/book/domain narrow which
    words qualify, in _build_word_filters below): always EXISTS(...), never
    JOIN ... ON book_id = ANY(%s). A JOIN produces one row per matching
    (word, book) pair, silently duplicating a multi-book word in a paginated
    list and inflating `total`. EXISTS collapses that to true/false per word.
  - AGGREGATING through the junction table (authors/books listings, where
    book/author IS the thing being counted): a JOIN ... GROUP BY is correct
    here -- the fan-out is the point -- with count(DISTINCT w.id) so a word
    isn't double-counted for an author just because it's in two of their
    books.

Mixing these up in either direction is the bug to avoid.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from concordance import usas_domains
from webapp.backend import main as _main

router = APIRouter()

_WORD_SORT_COLUMNS = {
    "lemma": "w.lemma",
    "difficulty": "wd.difficulty",
    "part_of_speech": "w.part_of_speech",
}


# --- shared filter builder ----------------------------------------------------

def _build_word_filters(
    author: str | None,
    book_id: list[int],
    domain: list[str],
    difficulty_min: float | None,
    difficulty_max: float | None,
    archaic: list[str],
    pos: list[str],
    quizzable_only: bool,
) -> tuple[list[str], list]:
    """The combinable-facet WHERE clause every endpoint below shares, each
    facet independently optional -- author and book_id can both be set at
    once (author picked, then narrowed to one of their books), not mutually
    exclusive branches. `w`/`wd` alias `word`/`word_difficulty` (LEFT JOIN)
    in every caller."""
    filters = ["w.active"]
    params: list = []

    if book_id:
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb
                        WHERE wb.word_id = w.id AND wb.book_id = ANY(%s))"""
        )
        params.append(book_id)
    if author:
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb
                        JOIN {_main.SCHEMA}.book b ON b.id = wb.book_id
                        WHERE wb.word_id = w.id AND b.author = %s)"""
        )
        params.append(author)
    if domain:
        codes = [code for bucket in domain
                 for code in usas_domains.DOMAIN_BUCKETS.get(bucket, {}).get("codes", [])]
        if codes:
            filters.append(
                f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                            JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                            WHERE wc.word_id = w.id AND left(c.code, 1) = ANY(%s))"""
            )
            params.append(codes)
    if archaic:
        filters.append("wd.archaic = ANY(%s)")
        params.append(archaic)
    if difficulty_min is not None:
        filters.append("wd.difficulty >= %s")
        params.append(difficulty_min)
    if difficulty_max is not None:
        filters.append("wd.difficulty <= %s")
        params.append(difficulty_max)
    if pos:
        filters.append("w.part_of_speech = ANY(%s)")
        params.append(pos)
    if quizzable_only:
        filters.append("wd.quizzable = true")

    return filters, params


# --- /api/browse/authors -------------------------------------------------------

class AuthorRow(BaseModel):
    author: str
    book_count: int
    word_count: int


class AuthorPage(BaseModel):
    items: list[AuthorRow]
    total: int
    page: int
    page_size: int


@router.get("/api/browse/authors", response_model=AuthorPage)
def browse_authors(
    q: str | None = None,
    book_id: list[int] = Query([]),
    domain: list[str] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["author", "word_count"] = "word_count",
    dir: Literal["asc", "desc"] = "desc",
    _: dict = Depends(_main.require_viewer),
) -> AuthorPage:
    # author/book_id are NOT passed through _build_word_filters here: that
    # helper's EXISTS subquery is only correct when the outer query is
    # anchored on `word` (browse_words) -- it correlates solely to w.id, with
    # no connection to whichever book/word_book row the outer query happens
    # to be iterating. browse_authors/browse_books already have a real,
    # correctly-scoped `b` in their own FROM/JOIN, so filtering by book_id
    # here is a direct condition against it instead. (Confirmed in
    # production: passing book_id through the word-anchored EXISTS made
    # EVERY author who ever shares so much as one common word with a book
    # match that book_id -- e.g. every Shakespeare play pulled in nearly the
    # entire corpus as "co-authors.")
    filters, params = _build_word_filters(
        None, [], domain, difficulty_min, difficulty_max, archaic, pos, quizzable_only
    )
    if book_id:
        filters.append("b.id = ANY(%s)")
        params.append(book_id)
    filters.append("b.author IS NOT NULL")
    if q:
        filters.append("b.author ILIKE %s")
        params.append(f"%{q}%")
    where = " AND ".join(filters)
    order_col = "b.author" if sort == "author" else "word_count"
    order_dir = "ASC" if dir == "asc" else "DESC"
    offset = (page - 1) * page_size

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*) FROM (
                    SELECT b.author
                    FROM {_main.SCHEMA}.book b
                    JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                    JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                    LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                    WHERE {where}
                    GROUP BY b.author
                ) sub""",
            params,
        )
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT b.author, count(DISTINCT b.id) AS book_count, count(DISTINCT w.id) AS word_count
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                GROUP BY b.author
                ORDER BY {order_col} {order_dir}, b.author ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    items = [AuthorRow(author=r[0], book_count=r[1], word_count=r[2]) for r in rows]
    return AuthorPage(items=items, total=total, page=page, page_size=page_size)


# --- /api/browse/books ---------------------------------------------------------

class BookRow(BaseModel):
    id: int
    title: str
    author: str | None
    word_count: int
    scored_word_count: int
    mean_difficulty: float | None
    stddev_difficulty: float | None


class BookPage(BaseModel):
    items: list[BookRow]
    total: int
    page: int
    page_size: int


@router.get("/api/browse/books", response_model=BookPage)
def browse_books(
    author: str | None = None,
    book_id: list[int] = Query([]),
    q: str | None = None,
    domain: list[str] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["title", "word_count"] = "title",
    dir: Literal["asc", "desc"] = "asc",
    _: dict = Depends(_main.require_viewer),
) -> BookPage:
    # author/book_id are NOT passed through _build_word_filters -- same
    # reasoning as browse_authors' book_id fix: this endpoint's outer query
    # already has a correctly-scoped `b`, so filtering by either is a direct
    # condition on it, not the word-anchored EXISTS meant for browse_words.
    # book_id here is a single-work lookup (the work-detail page needs this
    # endpoint's title/author/stats for one specific book), every other
    # browse endpoint already accepts book_id as a filter -- this was the one
    # inconsistent exception.
    filters, params = _build_word_filters(
        None, [], domain, difficulty_min, difficulty_max, archaic, pos, quizzable_only
    )
    if author:
        filters.append("b.author = %s")
        params.append(author)
    if book_id:
        filters.append("b.id = ANY(%s)")
        params.append(book_id)
    if q:
        filters.append("b.title ILIKE %s")
        params.append(f"%{q}%")
    where = " AND ".join(filters)
    order_col = "b.title" if sort == "title" else "word_count"
    order_dir = "ASC" if dir == "asc" else "DESC"
    offset = (page - 1) * page_size

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*) FROM (
                    SELECT b.id
                    FROM {_main.SCHEMA}.book b
                    JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                    JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                    LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                    WHERE {where}
                    GROUP BY b.id
                ) sub""",
            params,
        )
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                GROUP BY b.id, b.title, b.author
                ORDER BY {order_col} {order_dir}, b.title ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    items = [
        BookRow(id=r[0], title=r[1], author=r[2], word_count=r[3],
                scored_word_count=r[4], mean_difficulty=r[5], stddev_difficulty=r[6])
        for r in rows
    ]
    return BookPage(items=items, total=total, page=page, page_size=page_size)


# --- /api/browse/words ----------------------------------------------------------

class BrowseWordRow(BaseModel):
    id: int
    lemma: str
    part_of_speech: str | None
    definition: str | None
    difficulty: float | None
    archaic: str | None
    quizzable: bool | None


class BrowseWordPage(BaseModel):
    items: list[BrowseWordRow]
    total: int
    page: int
    page_size: int


@router.get("/api/browse/words", response_model=BrowseWordPage)
def browse_words(
    author: str | None = None,
    book_id: list[int] = Query([]),
    domain: list[str] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    q: str | None = None,
    letter: str | None = Query(None, min_length=1, max_length=1),
    random: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["lemma", "difficulty", "part_of_speech"] = "lemma",
    dir: Literal["asc", "desc"] = "asc",
    _: dict = Depends(_main.require_viewer),
) -> BrowseWordPage:
    filters, params = _build_word_filters(
        author, book_id, domain, difficulty_min, difficulty_max, archaic, pos, quizzable_only
    )
    if letter:
        filters.append("w.lemma_lc LIKE %s")
        params.append(f"{letter.lower()}%")
    if q:
        filters.append("similarity(w.lemma, %s) > 0.1")
        params.append(q)
    where = " AND ".join(filters)

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}""",
            params,
        )
        total = cur.fetchone()[0]

        if random:
            order_by = "random()"
            limit = 1
        elif q:
            # Best text match wins over the requested sort -- "search within
            # these filters" and "alphabetical/difficulty order" aren't
            # reconcilable in one ORDER BY, and search intent dominates.
            order_by = "similarity(w.lemma, %s) DESC"
            params = params + [q]
            limit = page_size
        else:
            order_col = _WORD_SORT_COLUMNS[sort]
            order_by = f"{order_col} {'ASC' if dir == 'asc' else 'DESC'} NULLS LAST, w.lemma ASC"
            limit = page_size
        offset = 0 if random else (page - 1) * page_size

        cur.execute(
            f"""SELECT w.id, w.lemma, w.part_of_speech, w.definition,
                       wd.difficulty, wd.archaic, wd.quizzable
                FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                ORDER BY {order_by}
                LIMIT %s OFFSET %s""",
            (*params, limit, offset),
        )
        rows = cur.fetchall()

    items = [
        BrowseWordRow(id=r[0], lemma=r[1], part_of_speech=r[2], definition=r[3],
                      difficulty=r[4], archaic=r[5], quizzable=r[6])
        for r in rows
    ]
    return BrowseWordPage(items=items, total=total, page=page, page_size=page_size)


# --- /api/browse/domains --------------------------------------------------------

class DomainBucketCount(BaseModel):
    bucket: str
    name: str
    word_count: int


def _bucket_counts(cur, where: str, params: list) -> list[DomainBucketCount]:
    """Word count per USAS color bucket, conditioned on whatever `where`
    already encodes. Six independent EXISTS-gated counts, not one GROUP BY --
    a word can carry categories in more than one bucket (up to 3 categories/
    word), so a naive GROUP BY would over/under-count words straddling
    buckets. Each count answers "how many words would this bucket add,"
    the right semantics for a filter facet, not "how many words primarily
    belong here." Shared by /api/browse/domains and /api/browse/domain-summary
    -- the two endpoints differ only in whether an uncategorized entry and a
    total are appended around this same per-bucket loop."""
    results = []
    for entry in usas_domains.legend_entries():
        codes = usas_domains.DOMAIN_BUCKETS[entry["bucket"]]["codes"]
        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where} AND EXISTS (
                    SELECT 1 FROM {_main.SCHEMA}.word_category wc
                    JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                    WHERE wc.word_id = w.id AND left(c.code, 1) = ANY(%s)
                )""",
            (*params, codes),
        )
        count = cur.fetchone()[0]
        results.append(DomainBucketCount(bucket=entry["bucket"], name=entry["name"], word_count=count))
    return results


@router.get("/api/browse/domains", response_model=list[DomainBucketCount])
def browse_domains(
    author: str | None = None,
    book_id: list[int] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    _: dict = Depends(_main.require_viewer),
) -> list[DomainBucketCount]:
    """The 6 named buckets only, as a bare list -- the shape the faceted
    Browse page's domain-chip row already depends on. See /api/browse/
    domain-summary for the uncategorized-inclusive, total-aware variant used
    by the work-detail chart; that one is a separate endpoint specifically so
    this one's response shape never has to change under an existing caller."""
    base_filters, base_params = _build_word_filters(
        author, book_id, [], difficulty_min, difficulty_max, archaic, pos, quizzable_only
    )
    where = " AND ".join(base_filters)
    with _main.get_conn() as conn, conn.cursor() as cur:
        return _bucket_counts(cur, where, base_params)


class DomainSummary(BaseModel):
    total_words: int
    buckets: list[DomainBucketCount]  # the 6 named buckets + one "uncategorized" entry, always last


@router.get("/api/browse/domain-summary", response_model=DomainSummary)
def browse_domain_summary(
    author: str | None = None,
    book_id: list[int] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    _: dict = Depends(_main.require_viewer),
) -> DomainSummary:
    """Like /api/browse/domains, but wrapped with `total_words` and an
    explicit "uncategorized" bucket (a word with zero category rows at all --
    disjoint from the 6 named buckets by construction, unlike those 6, which
    can overlap on a multi-category word). A separate endpoint rather than
    extending /api/browse/domains itself: that endpoint's bare-list shape is
    already depended on by the faceted Browse page's domain-chip click
    handler, which would silently no-op on an "uncategorized" chip (it isn't
    a real DOMAIN_BUCKETS key) -- safer to add the richer shape here than to
    risk breaking already-shipped UI for the sake of this one."""
    base_filters, base_params = _build_word_filters(
        author, book_id, [], difficulty_min, difficulty_max, archaic, pos, quizzable_only
    )
    where = " AND ".join(base_filters)

    with _main.get_conn() as conn, conn.cursor() as cur:
        buckets = _bucket_counts(cur, where, base_params)

        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where} AND NOT EXISTS (
                    SELECT 1 FROM {_main.SCHEMA}.word_category wc WHERE wc.word_id = w.id
                )""",
            base_params,
        )
        uncategorized = cur.fetchone()[0]
        buckets.append(DomainBucketCount(bucket="uncategorized", name="Uncategorized", word_count=uncategorized))

        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}""",
            base_params,
        )
        total_words = cur.fetchone()[0]

    return DomainSummary(total_words=total_words, buckets=buckets)


# --- /api/browse/difficulty-bands -----------------------------------------------

class DifficultyBandCount(BaseModel):
    band_min: float | None  # None = the "unscored" pseudo-band
    band_max: float | None
    label: str
    word_count: int


@router.get("/api/browse/difficulty-bands", response_model=list[DifficultyBandCount])
def browse_difficulty_bands(
    author: str | None = None,
    book_id: list[int] = Query([]),
    domain: list[str] = Query([]),
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    band_width: int = Query(20, ge=5, le=50),
    _: dict = Depends(_main.require_viewer),
) -> list[DifficultyBandCount]:
    """Word count per difficulty band, conditioned on every OTHER active
    facet, plus one explicit "not yet scored" band -- the mechanism that
    makes the corpus's current sparse difficulty coverage honest in the UI
    rather than silently vanishing once a difficulty filter narrows it."""
    base_filters, base_params = _build_word_filters(
        author, book_id, domain, None, None, archaic, pos, quizzable_only
    )
    where = " AND ".join(base_filters)

    results = []
    with _main.get_conn() as conn, conn.cursor() as cur:
        band_min = 0.0
        while band_min < 100:
            band_max = min(band_min + band_width, 100)
            is_last = band_max >= 100
            op = "<=" if is_last else "<"
            cur.execute(
                f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                    JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                    WHERE {where} AND wd.difficulty >= %s AND wd.difficulty {op} %s""",
                (*base_params, band_min, band_max),
            )
            count = cur.fetchone()[0]
            results.append(DifficultyBandCount(
                band_min=band_min, band_max=band_max,
                label=f"{int(band_min)}-{int(band_max)}", word_count=count,
            ))
            band_min = band_max

        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where} AND wd.difficulty IS NULL""",
            base_params,
        )
        unscored = cur.fetchone()[0]
        results.append(DifficultyBandCount(band_min=None, band_max=None, label="Not yet scored",
                                            word_count=unscored))
    return results


# --- /api/browse/pos-values ------------------------------------------------------

@router.get("/api/browse/pos-values", response_model=list[str])
def browse_pos_values(_: dict = Depends(_main.require_viewer)) -> list[str]:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT part_of_speech FROM {_main.SCHEMA}.word
                WHERE active AND coalesce(part_of_speech, '') <> ''
                ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]
