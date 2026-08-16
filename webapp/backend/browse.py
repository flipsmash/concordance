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

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from concordance import usas, usas_domains
from concordance.db import PLACEHOLDER_AUTHORS
from webapp.backend import main as _main

router = APIRouter()

# The 21 top-level USAS discourse fields ("discipline categories"), in
# usas.py's own fixed _TAGSET order -- level 0 is exactly the single-letter
# top fields (categories()'s own definition: parent_code is None). Computed
# once at import time, not per-request: it's 21 static (code, name) pairs
# derived from a module-level constant, not data that can change at runtime.
_TOP_CODES = [c["code"] for c in usas.categories() if c["level"] == 0]
_TOP_CODE_NAMES = {c["code"]: c["name"] for c in usas.categories() if c["level"] == 0}

# Every real USAS code at any level -- used to validate a `top_code`/`parent`
# query param before it reaches a query (404 on an unknown code), which
# incidentally closes off any %/_-wildcard injection since only this fixed
# 253-row known set ever reaches a LIKE parameter (see usas.subtree_sql).
_ALL_CODES = {c["code"] for c in usas.categories()}

# archive_path (concordance/archive_metadata.py) is repo-root-relative
# (e.g. "archive/1601 -- Twain, Mark.txt") -- resolved against this, the
# same "root + basename only" defensive pattern main.py's word_audio route
# already uses for _AUDIO_ROOT, so this route only ever serves a real file
# actually inside archive/ regardless of what's stored in the column.
_ARCHIVE_ROOT = Path(__file__).resolve().parents[2] / "archive"

_WORD_SORT_COLUMNS = {
    # lemma_lc, not the raw (case-preserving) lemma column -- this DB's
    # collation (C.UTF-8) sorts every uppercase letter before every
    # lowercase one, so "Hellenomania" (capitalized, per dictionary
    # convention for a demonym-derived word) would otherwise land first in
    # an A-Z listing, ahead of "aardvark" -- confirmed live, this exact word
    # is what surfaced the bug. lemma_lc is the already-indexed generated
    # column every other lemma-uniqueness/lookup path in this codebase uses.
    "lemma": "w.lemma_lc",
    "difficulty": "wd.difficulty",
    "part_of_speech": "w.part_of_speech",
    "book_count": "book_count",  # the SELECT-list alias below, not a real column
}

_BOOK_SORT_COLUMNS = {
    "title": "sort_title",
    "word_count": "word_count",
    "difficulty": "mean_difficulty",
    "density": "density",
    "overall_difficulty": "overall_difficulty",
    "fame": "fame_score",
    "unique_word_count": "unique_word_count",
}
_AUTHOR_SORT_COLUMNS = {
    "author": "author",
    "book_count": "book_count",
    "word_count": "word_count",
    "difficulty": "mean_difficulty",
    "density": "density",
    "overall_difficulty": "overall_difficulty",
    "fame": "fame_score",
    "unique_word_count": "unique_word_count",
}
# difficulty/density/overall_difficulty/fame are all sparse (a book with no
# scored words, or no distinct_nonstop_word_count from archive_metadata.py,
# has no value for them; fame_score is only populated for books/authors
# concordance book-fame/author-fame has actually scored) -- NULLS LAST is
# required explicitly for both directions, since Postgres's implicit default
# flips between ASC (NULLS LAST already) and DESC (NULLS FIRST by default,
# which would float unscored books to the top of a "hardest first" sort).
_NULLABLE_SORTS = {"difficulty", "density", "overall_difficulty", "fame"}

# A leading "The"/"A"/"An" is stripped before alphabetizing or bucketing by
# first letter -- a live corpus scan found 4176 of 11357 titles (37%) start
# with "The", which would otherwise pile more than a third of the corpus
# into one letter and make both plain alphabetical sort and an A-Z strip
# nearly useless. Standard library-catalog convention, not a display change:
# the title itself is shown unchanged, only its sort/bucket key is affected.
_SORT_TITLE_EXPR = "regexp_replace(lower(b.title), '^(the|a|an)\\s+', '')"


# --- shared filter builder ----------------------------------------------------

def _subtree_or_sql(codes: list[str]) -> tuple[str, list]:
    """OR of usas.subtree_sql(code) fragments -- "this word's category is
    exactly one of `codes`, or a real descendant of one of them." Used both
    for a bucket's member (level-0) codes and for a single field/sub-field/
    sub-sub-field code wrapped in a 1-element list; see usas.subtree_match
    for why this isn't a naive prefix match."""
    clauses = []
    params: list = []
    for code in codes:
        exact, like = usas.subtree_sql(code)
        clauses.append("(c.code = %s OR c.code LIKE %s)")
        params.extend([exact, like])
    return " OR ".join(clauses), params


def _build_word_filters(
    author: str | None,
    book_id: list[int],
    domain: list[str],
    difficulty_min: float | None,
    difficulty_max: float | None,
    archaic: list[str],
    pos: list[str],
    quizzable_only: bool,
    top_code: list[str] = [],
    all_top_code: list[str] = [],
    uncategorized: bool = False,
    unscored_only: bool = False,
    exclusive: bool = False,
) -> tuple[list[str], list]:
    """The combinable-facet WHERE clause every endpoint below shares, each
    facet independently optional -- author and book_id can both be set at
    once (author picked, then narrowed to one of their books), not mutually
    exclusive branches. `w`/`wd` alias `word`/`word_difficulty` (LEFT JOIN)
    in every caller.

    `top_code` is a second, independent category filter alongside `domain`:
    `domain` resolves a 6-hue bucket key to its member USAS top-level codes
    (usas_domains.DOMAIN_BUCKETS), while `top_code` matches one or more raw
    USAS codes AT ANY DEPTH (e.g. "S", "I2", "I2.2") directly, with no bucket
    indirection -- a word tagged at that code OR a real descendant of it
    matches (see usas.subtree_match) -- the field/sub-field/sub-sub-field
    Categories drill-down needs this, since a field's own code generally
    isn't a bucket key.

    `all_top_code` is a THIRD, independent category filter, deliberately
    separate from `top_code`: `top_code` with multiple values is an OR (any
    one of them), the right semantics for "browse this field" links, but
    the category-overlap heatmap's own click-through needs the opposite --
    "words carrying a category under BOTH A and BOTH B" (an intersection,
    matching exactly what browse_category_overlap's own shared-word count
    computes). One EXISTS per code, so it composes with plain AND via the
    filters list below rather than needing new join/logic here.

    `uncategorized`/`unscored_only` are the two SQL fragments
    /api/browse/domain-summary and /api/browse/difficulty-bands already
    computed standalone for their own "Uncategorized"/"Not yet scored"
    counts -- pulled in here so every caller (including /api/browse/words,
    which needed them to make those two chart segments clickable) shares one
    definition. Rejecting `uncategorized` alongside `domain`/`top_code`, or
    `unscored_only` alongside `difficulty_min`/`difficulty_max`, is each
    caller's job, not this helper's -- it just composes whatever it's given,
    same as `top_code`'s own validation living in its endpoint. `unscored_only`
    only means anything against a LEFT JOIN word_difficulty; a caller that
    INNER JOINs word_difficulty (as browse_difficulty_bands's own band loop
    does) must never pass it.

    `exclusive` answers "does every word_book row for this word point back
    to book_id (or, lacking that, to a book by `author`)" -- i.e. this word
    appears NOWHERE else in the corpus. Meaningless without book_id or
    author already narrowing the scope, so it's a silent no-op if neither is
    set rather than a validation error (mirrors this helper's existing
    "compose whatever it's given" philosophy). book_id takes precedence if
    both are somehow set at once -- never happens from the current UI, which
    always passes exactly one of the two. `!= ALL(%s)` (not `NOT (... = ANY
    (%s))`) so this reads directly as "a word_book row pointing outside this
    set exists" -- generalizes correctly to a multi-book book_id even though
    every current caller only ever passes one."""
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
    if exclusive and book_id:
        filters.append(
            f"""NOT EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb2
                            WHERE wb2.word_id = w.id AND wb2.book_id != ALL(%s))"""
        )
        params.append(book_id)
    elif exclusive and author:
        filters.append(
            f"""NOT EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb2
                            JOIN {_main.SCHEMA}.book b2 ON b2.id = wb2.book_id
                            WHERE wb2.word_id = w.id AND b2.author IS DISTINCT FROM %s)"""
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
    if top_code:
        subtree_where, subtree_params = _subtree_or_sql(top_code)
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                        JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                        WHERE wc.word_id = w.id AND ({subtree_where}))"""
        )
        params.extend(subtree_params)
    for code in all_top_code:
        exact, like = usas.subtree_sql(code)
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                        JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                        WHERE wc.word_id = w.id AND (c.code = %s OR c.code LIKE %s))"""
        )
        params.extend([exact, like])
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
    if uncategorized:
        filters.append(
            f"""NOT EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                            WHERE wc.word_id = w.id)"""
        )
    if unscored_only:
        filters.append("wd.difficulty IS NULL")

    return filters, params


# Fixed, named (not equal-width) buckets for "how many of this book's/
# author's active words appear NOWHERE else in the corpus" -- confirmed live
# against the real corpus: this distribution is heavily right-skewed (book
# scope: median 0, p90 2, p99 11, max 251; author scope: median 1, p90 7,
# p99 41, max 1752), so equal-width bins (as difficulty-bands uses, sensible
# there since difficulty is bounded 0-100) would dump nearly everything into
# one bucket and spread a handful of outliers across a mostly-empty axis.
# These boundaries keep real detail in the low end where most of the corpus
# actually sits, with a final open-ended overflow bucket for the long tail
# (huge "Complete Works"-style compilations legitimately have dozens to
# hundreds of words nothing else in the corpus uses). Shared by
# /api/browse/unique-word-histogram (below) and the `unique_word_bucket`
# click-through filter on /api/browse/books and /api/browse/authors.
_UNIQUE_WORD_BUCKETS: list[tuple[int, int | None, str]] = [
    (0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"),
    (11, 25, "11-25"), (26, 50, "26-50"), (51, 100, "51-100"), (101, None, "101+"),
]


def _unique_word_bucket_label(n: int) -> str:
    for lo, hi, label in _UNIQUE_WORD_BUCKETS:
        if hi is None or n <= hi:
            return label
    return _UNIQUE_WORD_BUCKETS[-1][2]  # unreachable (last bucket's hi is None)


def _unique_word_bucket_filter(scope: Literal["book", "author"], bucket_label: str) -> tuple[str, list]:
    """A WHERE-clause fragment (append to `filters`) + its params, restricting
    to books (scope='book', tests b.id) or to books BY an author (scope=
    'author', tests b.author) whose exclusive-word count falls in the named
    bucket -- same word_book-exclusivity semantics /api/browse/words?
    exclusive=true and /api/browse/unique-word-histogram both use, just
    aggregated per book/author here instead of per word. Both browse_books
    and browse_authors already alias the book table `b`, so this composes
    directly into either endpoint's existing `filters` list.

    The "0" bucket is handled separately (NOT EXISTS rather than a HAVING
    count(*) BETWEEN): a book/author with zero exclusive words has no rows
    at all in the single-book/author-words aggregate, so a `HAVING count(*)
    >= 0` over an inner join would never match it -- there's no row to
    match in the first place."""
    for lo, hi, label in _UNIQUE_WORD_BUCKETS:
        if label == bucket_label:
            break
    else:
        raise HTTPException(404, f"unknown unique-word bucket {bucket_label!r}")

    s = _main.SCHEMA
    if scope == "book":
        single_cte = f"""SELECT wb2.word_id, min(wb2.book_id) AS key
                          FROM {s}.word_book wb2
                          JOIN {s}.word w2 ON w2.id = wb2.word_id AND w2.active
                          GROUP BY wb2.word_id HAVING count(*) = 1"""
        cte_params: list = []
        key_col = "b.id"
    else:
        # PLACEHOLDER_AUTHORS ("Various", "Unknown Author", ...) excluded here
        # too -- browse_authors (the click-through target) already excludes
        # them from its own listing, so counting them here would make this
        # bucket's count not match what that filtered list actually returns.
        single_cte = f"""SELECT wb2.word_id, min(b2.author) AS key
                          FROM {s}.word_book wb2
                          JOIN {s}.book b2 ON b2.id = wb2.book_id
                              AND b2.author IS NOT NULL AND b2.author != ''
                              AND b2.author != ALL(%s)
                          JOIN {s}.word w2 ON w2.id = wb2.word_id AND w2.active
                          GROUP BY wb2.word_id HAVING count(DISTINCT b2.author) = 1"""
        cte_params = [list(PLACEHOLDER_AUTHORS)]
        key_col = "b.author"

    if lo == 0 and hi == 0:
        return f"NOT EXISTS (SELECT 1 FROM ({single_cte}) sw WHERE sw.key = {key_col})", cte_params
    hi_clause = "" if hi is None else "AND count(*) <= %s"
    params = cte_params + ([lo] if hi is None else [lo, hi])
    return (
        f"""{key_col} IN (
            SELECT sw.key FROM ({single_cte}) sw
            GROUP BY sw.key HAVING count(*) >= %s {hi_clause}
        )""",
        params,
    )


# Equal-width (20-point) bands over overall_difficulty's 0-100 percentile
# scale, plus one pseudo-band for books/authors with no overall_difficulty
# at all (missing mean_difficulty or density -- see BookRow.overall_difficulty).
# Same half-open-except-the-last-band convention as /api/browse/difficulty-
# bands (band_min <= x < band_max, but band_min <= x <= 100 for the final
# band) -- NOT fame's inclusive-both-ends convention, which only works there
# because fame's bars are single integers (min == max); a boundary value
# here (e.g. exactly 80.0) must land in exactly one bar.
_OVERALL_DIFFICULTY_BAND_WIDTH = 20
_OVERALL_DIFFICULTY_UNSCORED_LABEL = "Not enough data"


def _overall_difficulty_band_filter(label: str) -> tuple[str, list]:
    """A WHERE-clause fragment (append to a query that already has
    `overall_difficulty` as a materialized column/alias in scope -- never
    valid inside the same SELECT's own WHERE, since that's evaluated before
    a CASE-expression alias exists) + its params, isolating exactly the
    population /api/browse/overall-difficulty-histogram's bar for this label
    counted. Round-trips any label that endpoint actually emits; 404s on
    anything else, same "no filter can silently mean nothing" contract as
    _unique_word_bucket_filter above."""
    if label == _OVERALL_DIFFICULTY_UNSCORED_LABEL:
        return "overall_difficulty IS NULL", []
    try:
        lo_s, hi_s = label.split("-")
        lo, hi = float(lo_s), float(hi_s)
    except ValueError:
        raise HTTPException(404, f"unknown overall-difficulty band {label!r}")
    op = "<=" if hi >= 100 else "<"
    return f"overall_difficulty >= %s AND overall_difficulty {op} %s", [lo, hi]


# --- /api/browse/authors -------------------------------------------------------

class AuthorRow(BaseModel):
    author: str
    book_count: int
    word_count: int
    scored_word_count: int
    mean_difficulty: float | None
    stddev_difficulty: float | None
    density: float | None  # mean of this author's own per-book densities (see
                            # BookRow.density) -- NOT unique words / summed
                            # distinct_nonstop_word_count, which would grow
                            # the denominator with book_count while the
                            # (deduped) numerator saturates, systematically
                            # crushing prolific authors' density.
    overall_difficulty: float | None  # see BookRow.overall_difficulty
    fame_score: float | None      # 1-10, LLM-judged -- see concordance/fame.py
    fame_reasoning: str | None
    unique_word_count: int  # see BookRow.unique_word_count -- same semantics,
                             # aggregated across this author's whole body of work


class AuthorPage(BaseModel):
    items: list[AuthorRow]
    total: int
    page: int
    page_size: int


@router.get("/api/browse/authors", response_model=AuthorPage)
def browse_authors(
    q: str | None = None,
    author: str | None = None,   # exact match -- the author-detail page's own lookup (see book's own `author` filter)
    book_id: list[int] = Query([]),
    domain: list[str] = Query([]),
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    fame_min: float | None = None,
    fame_max: float | None = None,
    fame_unscored_only: bool = False,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    unique_word_bucket: str | None = None,
    overall_difficulty_band: str | None = None,  # see the matching param on browse_books
    letter: str | None = Query(None, min_length=1, max_length=1),
    random: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["author", "book_count", "word_count", "difficulty", "density", "overall_difficulty",
                  "fame", "unique_word_count"] = "word_count",
    dir: Literal["asc", "desc"] = "desc",
    _: dict = Depends(_main.require_viewer),
) -> AuthorPage:
    if fame_unscored_only and (fame_min is not None or fame_max is not None):
        raise HTTPException(400, "fame_unscored_only is mutually exclusive with fame_min/fame_max")
    overall_diff_where, overall_diff_params = "", []
    if overall_difficulty_band is not None:
        _clause, overall_diff_params = _overall_difficulty_band_filter(overall_difficulty_band)
        overall_diff_where = f" AND {_clause}"
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
    # PLACEHOLDER_AUTHORS ("Various", "Unknown Author", ...) are already
    # excluded from author_similarity/clustering (db.py) but this listing
    # endpoint never got the same filter -- live data showed "Various" alone
    # accounts for 1193 books, enough to make it this corpus's single
    # most-common "author" and dominate the A-Z strip's "V" bucket.
    filters.append("b.author != ALL(%s)")
    params.append(list(PLACEHOLDER_AUTHORS))
    if author:
        filters.append("b.author = %s")
        params.append(author)
    if letter:
        filters.append("lower(left(b.author, 1)) = %s")
        params.append(letter.lower())
    if q:
        filters.append("b.author ILIKE %s")
        params.append(f"%{q}%")
    if unique_word_bucket:
        clause, bucket_params = _unique_word_bucket_filter("author", unique_word_bucket)
        filters.append(clause)
        params.extend(bucket_params)
    where = " AND ".join(filters)

    # fame_score lives on author_fame, joined separately below (at the point
    # each query has a clean, un-fanned-out `author` to join against) rather
    # than folded into `filters`/`where` -- the main query below builds
    # author_base through several CTEs that each independently re-apply
    # `WHERE {where}` against the same word/book-fanned-out join, and every
    # one of them would need its own author_fame join for a shared filters
    # list to work; simpler and just as correct to apply this one filter
    # once, after the joins that actually need it.
    fame_filters = []
    fame_params: list = []
    if fame_unscored_only:
        fame_filters.append("af.fame_score IS NULL")
    else:
        if fame_min is not None:
            fame_filters.append("af.fame_score >= %s")
            fame_params.append(fame_min)
        if fame_max is not None:
            fame_filters.append("af.fame_score <= %s")
            fame_params.append(fame_max)
    fame_where = (" AND " + " AND ".join(fame_filters)) if fame_filters else ""

    if random:
        order_by = "random()"
        limit = 1
    else:
        order_col = _AUTHOR_SORT_COLUMNS[sort]
        order_dir = "ASC" if dir == "asc" else "DESC"
        nulls = " NULLS LAST" if sort in _NULLABLE_SORTS else ""
        order_by = f"{order_col} {order_dir}{nulls}, author ASC"
        limit = page_size
    offset = 0 if random else (page - 1) * page_size

    with _main.get_conn() as conn, conn.cursor() as cur:
        if overall_difficulty_band is not None:
            # Same reasoning as browse_books' matching branch: overall_
            # difficulty is a percentile, not a stored column, so counting a
            # band's matches means computing it for the whole `where`-
            # filtered set first -- mean_difficulty/density only, via the
            # same minimal author_base shape the histogram endpoint uses,
            # but scoped to this request's actual `where`/fame_where rather
            # than the histogram's fixed unfiltered baseline.
            cur.execute(
                f"""SELECT count(*) FROM (
                        WITH book_word_counts AS (
                            SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                                   count(DISTINCT w.id) AS book_word_count
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                            LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                            WHERE {where}
                            GROUP BY b.author, b.id, b.distinct_nonstop_word_count
                        ),
                        author_density AS (
                            SELECT author, avg(density) AS density
                            FROM (
                                SELECT author,
                                       CASE WHEN distinct_nonstop_word_count > 0
                                            THEN book_word_count::float / distinct_nonstop_word_count END AS density
                                FROM book_word_counts
                            ) bd
                            WHERE density IS NOT NULL
                            GROUP BY author
                        ),
                        author_word_ids AS (
                            SELECT DISTINCT b.author, w.id AS word_id
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                            LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                            WHERE {where}
                        ),
                        author_word_stats AS (
                            SELECT awi.author, avg(wd.difficulty) AS mean_difficulty
                            FROM author_word_ids awi
                            LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = awi.word_id
                            GROUP BY awi.author
                        ),
                        author_base AS (
                            SELECT aws.author, aws.mean_difficulty, ad.density
                            FROM author_word_stats aws
                            LEFT JOIN author_density ad ON ad.author = aws.author
                        ),
                        diff_rank AS (
                            SELECT author, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                            FROM author_base WHERE mean_difficulty IS NOT NULL
                        ),
                        dens_rank AS (
                            SELECT author, percent_rank() OVER (ORDER BY density) AS dens_pct
                            FROM author_base WHERE density IS NOT NULL
                        )
                        SELECT CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                                    THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                               END AS overall_difficulty
                        FROM author_base ab
                        LEFT JOIN diff_rank dr ON dr.author = ab.author
                        LEFT JOIN dens_rank de ON de.author = ab.author
                        LEFT JOIN {_main.SCHEMA}.author_fame af ON af.author = ab.author
                        WHERE true{fame_where}
                    ) sub WHERE true{overall_diff_where}""",
                (*params, *params, *fame_params, *overall_diff_params),
            )
        else:
            cur.execute(
                f"""SELECT count(*) FROM (
                        SELECT b.author
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                        LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                        LEFT JOIN {_main.SCHEMA}.author_fame af ON af.author = b.author
                        WHERE {where}{fame_where}
                        GROUP BY b.author, af.fame_score
                    ) sub""",
                (*params, *fame_params),
            )
        total = cur.fetchone()[0]

        # Two independent DISTINCT sets, not one: a word appearing in two of
        # an author's books must count once for word_count/mean_difficulty
        # (else a word's difficulty is averaged in twice), while book_count
        # obviously needs one row per book. Folding both into a single
        # (author, book_id, word_id) DISTINCT and joining word_difficulty
        # onto it would silently reintroduce that double-count for any word
        # shared across an author's own books.
        #
        # density is the MEAN OF PER-BOOK densities, not
        # (deduped word_count) / (summed distinct_nonstop_word_count) --
        # the latter's denominator grows with book_count while the numerator
        # saturates (an author's vocabulary overlaps across their own
        # books), which would crush prolific authors' density purely as a
        # function of how many books they wrote.
        #
        # overall_difficulty: see BookRow's docstring comment on the same
        # field in browse_books below -- identical percent_rank-blend
        # reasoning, computed here over authors instead of books.
        cur.execute(
            f"""
            WITH single_author_words AS (
                -- Same exclusivity semantics as _unique_word_bucket_filter's
                -- author branch/browse_unique_word_histogram (every word_book
                -- row for this word points to a book by the SAME author,
                -- globally -- not scoped to `where` below, since exclusivity
                -- is a corpus-wide fact). PLACEHOLDER_AUTHORS excluded so an
                -- author's unique_word_count here can't be inflated by
                -- co-occurrence with an "Various"/"Unknown Author" anthology.
                SELECT wb2.word_id, min(b2.author) AS author
                FROM {_main.SCHEMA}.word_book wb2
                JOIN {_main.SCHEMA}.book b2 ON b2.id = wb2.book_id
                    AND b2.author IS NOT NULL AND b2.author != '' AND b2.author != ALL(%s)
                JOIN {_main.SCHEMA}.word w2 ON w2.id = wb2.word_id AND w2.active
                GROUP BY wb2.word_id HAVING count(DISTINCT b2.author) = 1
            ),
            author_unique_counts AS (
                SELECT author, count(*) AS unique_word_count
                FROM single_author_words
                GROUP BY author
            ),
            book_word_counts AS (
                SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                       count(DISTINCT w.id) AS book_word_count
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                GROUP BY b.author, b.id, b.distinct_nonstop_word_count
            ),
            author_density AS (
                SELECT author, avg(density) AS density
                FROM (
                    SELECT author,
                           CASE WHEN distinct_nonstop_word_count > 0
                                THEN book_word_count::float / distinct_nonstop_word_count END AS density
                    FROM book_word_counts
                ) bd
                WHERE density IS NOT NULL
                GROUP BY author
            ),
            author_books AS (
                SELECT author, count(DISTINCT book_id) AS book_count
                FROM book_word_counts
                GROUP BY author
            ),
            author_word_ids AS (
                SELECT DISTINCT b.author, w.id AS word_id
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
            ),
            author_word_stats AS (
                SELECT awi.author, count(DISTINCT awi.word_id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty
                FROM author_word_ids awi
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = awi.word_id
                GROUP BY awi.author
            ),
            author_base AS (
                SELECT aws.author, ab.book_count, aws.word_count, aws.scored_word_count,
                       aws.mean_difficulty, aws.stddev_difficulty, ad.density,
                       coalesce(auc.unique_word_count, 0) AS unique_word_count
                FROM author_word_stats aws
                JOIN author_books ab ON ab.author = aws.author
                LEFT JOIN author_density ad ON ad.author = aws.author
                LEFT JOIN author_unique_counts auc ON auc.author = aws.author
            ),
            diff_rank AS (
                SELECT author, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                FROM author_base WHERE mean_difficulty IS NOT NULL
            ),
            dens_rank AS (
                SELECT author, percent_rank() OVER (ORDER BY density) AS dens_pct
                FROM author_base WHERE density IS NOT NULL
            ),
            scored AS (
                SELECT ab2.author, ab2.book_count, ab2.word_count, ab2.scored_word_count,
                       ab2.mean_difficulty, ab2.stddev_difficulty, ab2.density,
                       CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                            THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                       END AS overall_difficulty,
                       af.fame_score, af.fame_reasoning, ab2.unique_word_count
                FROM author_base ab2
                LEFT JOIN diff_rank dr ON dr.author = ab2.author
                LEFT JOIN dens_rank de ON de.author = ab2.author
                LEFT JOIN {_main.SCHEMA}.author_fame af ON af.author = ab2.author
                WHERE true{fame_where}
            )
            SELECT author, book_count, word_count, scored_word_count, mean_difficulty,
                   stddev_difficulty, density, overall_difficulty, fame_score, fame_reasoning,
                   unique_word_count
            FROM scored
            WHERE true{overall_diff_where}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s""",
            (list(PLACEHOLDER_AUTHORS), *params, *params, *fame_params, *overall_diff_params, limit, offset),
        )
        rows = cur.fetchall()

    items = [
        AuthorRow(author=r[0], book_count=r[1], word_count=r[2], scored_word_count=r[3],
                  mean_difficulty=r[4], stddev_difficulty=r[5], density=r[6], overall_difficulty=r[7],
                  fame_score=r[8], fame_reasoning=r[9], unique_word_count=r[10])
        for r in rows
    ]
    return AuthorPage(items=items, total=total, page=page, page_size=page_size)


# --- /api/browse/books ---------------------------------------------------------

class BookRow(BaseModel):
    id: int
    title: str
    author: str | None
    word_count: int  # distinct EXTRACTED VOCABULARY words for this book (word_book
                      # count) -- NOT concordance/archive_metadata.py's book.word_count
                      # (the archived text's own raw word count); unrelated metrics that
                      # happen to share a name, this one predates the other.
    scored_word_count: int
    mean_difficulty: float | None
    stddev_difficulty: float | None
    density: float | None  # word_count / archive_metadata.py's distinct_nonstop_word_count
                            # -- this book's extracted vocabulary as a fraction of its own
                            # distinct non-stopword count. A live scan found only 2/11356
                            # books above 0.5 (one just above 1.0, a very short poetry
                            # collection), so no corpus-wide outlier handling beyond what
                            # overall_difficulty's percent_rank already gives for free.
    archive_path: str | None  # set -> /api/browse/books/{id}/text can serve it
    overall_difficulty: float | None
    fame_score: float | None      # 1-10, LLM-judged -- see concordance/fame.py
    fame_reasoning: str | None
    # percent_rank(mean_difficulty) averaged with percent_rank(density), *100,
    # rounded to 1dp -- e.g. 74.3 reads as "harder than 74.3% of the corpus."
    # NOT a z-score blend: a live scan found density's distribution stddev
    # 0.0175 against a max of 1.18 (z up to ~66) while difficulty's z never
    # exceeds ~1.3, so summing raw z-scores would just be a density ranking
    # with a difficulty-shaped rounding error attached. percent_rank is
    # scale-free and immune to that skew. Each percentile is computed only
    # over books that HAVE the underlying metric, so a book missing one
    # doesn't get pushed to an arbitrary end by NULL-ordering -- it's simply
    # excluded from that one ranking, and overall_difficulty itself is only
    # populated when both are available.
    unique_word_count: int  # words appearing NOWHERE else in the corpus -- same
                             # exclusivity semantics as ?exclusive=true and the
                             # unique-word-histogram, always a real count (never
                             # null -- a book contributing zero is still 0, not absent)
    genres: list[str]  # concordance.genre.GENRE_LIST tags, from `concordance book-genres`
                        # -- [] for a book not yet classified, not null


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
    fame_min: float | None = None,
    fame_max: float | None = None,
    fame_unscored_only: bool = False,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    unique_word_bucket: str | None = None,
    overall_difficulty_band: str | None = None,  # a label /api/browse/overall-difficulty-histogram
                                                   # emitted, e.g. "40-60" or "Not enough data" --
                                                   # see _overall_difficulty_band_filter
    letter: str | None = Query(None, min_length=1, max_length=1),
    random: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["title", "word_count", "difficulty", "density", "overall_difficulty",
                  "fame", "unique_word_count"] = "title",
    dir: Literal["asc", "desc"] = "asc",
    _: dict = Depends(_main.require_viewer),
) -> BookPage:
    if fame_unscored_only and (fame_min is not None or fame_max is not None):
        raise HTTPException(400, "fame_unscored_only is mutually exclusive with fame_min/fame_max")
    # overall_difficulty is a CASE-expression alias, not a real column, so
    # this filter can't join `filters`/`where` (evaluated before that alias
    # exists) -- applied later, against the `scored` CTE both queries below
    # build once overall_difficulty is actually materialized.
    overall_diff_where, overall_diff_params = "", []
    if overall_difficulty_band is not None:
        _clause, overall_diff_params = _overall_difficulty_band_filter(overall_difficulty_band)
        overall_diff_where = f" AND {_clause}"
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
    if letter:
        filters.append(f"left({_SORT_TITLE_EXPR}, 1) = %s")
        params.append(letter.lower())
    # fame_score lives on a LEFT JOINed table (book_fame), not the word-level
    # aggregation the rest of `filters` targets -- appended to the same
    # WHERE clause regardless, since a plain equality/range condition on a
    # LEFT JOINed column works identically there (it just also has to be
    # true of NULL-having rows being excluded, which is exactly what a
    # min/max filter on an unscored book should do).
    if fame_unscored_only:
        filters.append("bf.fame_score IS NULL")
    else:
        if fame_min is not None:
            filters.append("bf.fame_score >= %s")
            params.append(fame_min)
        if fame_max is not None:
            filters.append("bf.fame_score <= %s")
            params.append(fame_max)
    if unique_word_bucket:
        clause, bucket_params = _unique_word_bucket_filter("book", unique_word_bucket)
        filters.append(clause)
        params.extend(bucket_params)
    where = " AND ".join(filters)
    if random:
        order_by = "random()"
        limit = 1
    else:
        order_col = _BOOK_SORT_COLUMNS[sort]
        order_dir = "ASC" if dir == "asc" else "DESC"
        nulls = " NULLS LAST" if sort in _NULLABLE_SORTS else ""
        order_by = f"{order_col} {order_dir}{nulls}, sort_title ASC"
        limit = page_size
    offset = 0 if random else (page - 1) * page_size

    with _main.get_conn() as conn, conn.cursor() as cur:
        if overall_difficulty_band is not None:
            # overall_difficulty is a corpus-percentile, not a stored column
            # -- unlike every other filter here, counting matches requires
            # computing it for the whole `where`-filtered set first, so this
            # can't reuse the cheap per-book GROUP BY below. Only book_id/
            # mean_difficulty/density are needed to get there (no title/
            # fame/unique-word-count), but book_fame still has to be joined:
            # `where` may itself reference `bf.fame_score` (fame_min/max
            # above), and dropping the join would 42P01 on that reference.
            cur.execute(
                f"""SELECT count(*) FROM (
                        WITH book_base AS (
                            SELECT b.id, b.distinct_nonstop_word_count,
                                   avg(wd.difficulty) AS mean_difficulty,
                                   CASE WHEN b.distinct_nonstop_word_count > 0
                                        THEN count(DISTINCT w.id)::float / b.distinct_nonstop_word_count END AS density
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                            LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                            LEFT JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = b.id
                            WHERE {where}
                            GROUP BY b.id, b.distinct_nonstop_word_count
                        ),
                        diff_rank AS (
                            SELECT id, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                            FROM book_base WHERE mean_difficulty IS NOT NULL
                        ),
                        dens_rank AS (
                            SELECT id, percent_rank() OVER (ORDER BY density) AS dens_pct
                            FROM book_base WHERE density IS NOT NULL
                        )
                        SELECT CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                                    THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                               END AS overall_difficulty
                        FROM book_base bb
                        LEFT JOIN diff_rank dr ON dr.id = bb.id
                        LEFT JOIN dens_rank de ON de.id = bb.id
                    ) sub WHERE true{overall_diff_where}""",
                (*params, *overall_diff_params),
            )
        else:
            cur.execute(
                f"""SELECT count(*) FROM (
                        SELECT b.id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                        LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                        LEFT JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = b.id
                        WHERE {where}
                        GROUP BY b.id, bf.fame_score
                    ) sub""",
                params,
            )
        total = cur.fetchone()[0]

        # See BookRow's field comments above for why density/overall_difficulty
        # are computed the way they are (percent_rank blend, not z-scores).
        cur.execute(
            f"""
            WITH single_book_words AS (
                -- Same exclusivity semantics as _unique_word_bucket_filter/
                -- browse_unique_word_histogram (word_book row count = 1 for
                -- this word, globally -- not scoped to `where` below, since
                -- a word's exclusivity is a corpus-wide fact independent of
                -- which books this particular page of results is filtered to).
                SELECT wb2.word_id, min(wb2.book_id) AS book_id
                FROM {_main.SCHEMA}.word_book wb2
                JOIN {_main.SCHEMA}.word w2 ON w2.id = wb2.word_id AND w2.active
                GROUP BY wb2.word_id HAVING count(*) = 1
            ),
            book_unique_counts AS (
                -- Pre-aggregated once, then LEFT JOINed by id below -- a
                -- correlated subquery here instead (WHERE sbw.book_id = b.id
                -- per book_base row) measured at 73s against the real corpus:
                -- single_book_words has no index Postgres can use for that
                -- per-row lookup, so it re-scans the whole CTE result once
                -- per book (~20.8k times). This GROUP BY runs once, total.
                SELECT book_id, count(*) AS unique_word_count
                FROM single_book_words
                GROUP BY book_id
            ),
            book_genres AS (
                -- Pre-aggregated to one row per book, same reasoning as
                -- book_unique_counts above: joining book_genre directly into
                -- book_base's per-word GROUP BY would fan out one row per
                -- genre tag, corrupting avg(wd.difficulty)/stddev_samp
                -- (not DISTINCT-safe like count(DISTINCT w.id) is).
                SELECT book_id, array_agg(genre ORDER BY genre) AS genres
                FROM {_main.SCHEMA}.book_genre
                GROUP BY book_id
            ),
            book_base AS (
                SELECT b.id, b.title, b.author, b.archive_path, b.distinct_nonstop_word_count,
                       count(DISTINCT w.id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty,
                       CASE WHEN b.distinct_nonstop_word_count > 0
                            THEN count(DISTINCT w.id)::float / b.distinct_nonstop_word_count END AS density,
                       bf.fame_score, bf.fame_reasoning,
                       {_SORT_TITLE_EXPR} AS sort_title,
                       coalesce(buc.unique_word_count, 0) AS unique_word_count,
                       coalesce(bg.genres, '{{}}') AS genres
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                LEFT JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = b.id
                LEFT JOIN book_unique_counts buc ON buc.book_id = b.id
                LEFT JOIN book_genres bg ON bg.book_id = b.id
                WHERE {where}
                GROUP BY b.id, b.title, b.author, b.archive_path, b.distinct_nonstop_word_count,
                         bf.fame_score, bf.fame_reasoning, buc.unique_word_count, bg.genres
            ),
            diff_rank AS (
                SELECT id, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                FROM book_base WHERE mean_difficulty IS NOT NULL
            ),
            dens_rank AS (
                SELECT id, percent_rank() OVER (ORDER BY density) AS dens_pct
                FROM book_base WHERE density IS NOT NULL
            ),
            scored AS (
                SELECT bb.id, bb.title, bb.author, bb.word_count, bb.scored_word_count,
                       bb.mean_difficulty, bb.stddev_difficulty, bb.density, bb.archive_path,
                       CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                            THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                       END AS overall_difficulty,
                       bb.fame_score, bb.fame_reasoning, bb.unique_word_count, bb.sort_title, bb.genres
                FROM book_base bb
                LEFT JOIN diff_rank dr ON dr.id = bb.id
                LEFT JOIN dens_rank de ON de.id = bb.id
            )
            SELECT id, title, author, word_count, scored_word_count, mean_difficulty,
                   stddev_difficulty, density, archive_path, overall_difficulty,
                   fame_score, fame_reasoning, unique_word_count, genres
            FROM scored
            WHERE true{overall_diff_where}
            ORDER BY {order_by}
            LIMIT %s OFFSET %s""",
            (*params, *overall_diff_params, limit, offset),
        )
        rows = cur.fetchall()

    items = [
        BookRow(id=r[0], title=r[1], author=r[2], word_count=r[3],
                scored_word_count=r[4], mean_difficulty=r[5], stddev_difficulty=r[6],
                density=r[7], archive_path=r[8], overall_difficulty=r[9],
                fame_score=r[10], fame_reasoning=r[11], unique_word_count=r[12], genres=r[13] or [])
        for r in rows
    ]
    return BookPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/api/browse/books/{book_id}/text")
def book_text(book_id: int, _: dict = Depends(_main.require_viewer)):
    """Streams the book's own archived full text (concordance/archive_
    metadata.py's archive_path), the way word_audio streams a word's mp3 --
    looked up from the DB-controlled column rather than exposing archive/
    via a raw StaticFiles mount, so this route only ever serves what a
    book row vouches for."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT archive_path FROM {_main.SCHEMA}.book WHERE id = %s", (book_id,))
        row = cur.fetchone()
    if row is None or not row[0]:
        raise HTTPException(status_code=404, detail="no archived text for this book")
    full_path = (_ARCHIVE_ROOT / Path(row[0]).name).resolve()
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="archive file missing on disk")
    return FileResponse(full_path, media_type="text/plain", filename=full_path.name)


# --- /api/browse/books/{id}/related -----------------------------------------

class BookGraphNode(BaseModel):
    id: int
    title: str
    author: str | None
    ring: int              # 0 = center, 1 = a related book
    word_count: int | None  # populated for the center only -- see book_related's
                             # own comment for why a related node sizes off
                             # shared_word_count instead (no extra JOIN needed)


class BookGraphEdge(BaseModel):
    source: int
    target: int
    score: float            # similarity -- HIGHER means more related. The
                             # opposite of word_graph's GraphEdge.distance
                             # (lower = closer) -- the frontend force-layout
                             # must invert this (link length ~ 1 - score),
                             # not reuse distance-is-already-right assumptions.
    shared_word_count: int
    is_center_edge: bool    # True: center<->neighbor ("why you're looking at
                             # this"). False: a cross-link BETWEEN two of the
                             # displayed neighbors -- surfaced only when it
                             # already happens to be stored (each also made
                             # the other's own top-k), not newly computed. A
                             # neighbor pair that's real but too weak for
                             # either side's top-k cutoff won't appear here;
                             # that's an honest gap, not a bug.


class BookRelatedResponse(BaseModel):
    center: BookGraphNode
    nodes: list[BookGraphNode]   # includes the center (ring=0) AND related books (ring=1)
    edges: list[BookGraphEdge]


@router.get("/api/browse/books/{book_id}/related", response_model=BookRelatedResponse)
def book_related(
    book_id: int,
    top_k: int = Query(8, ge=1, le=20),
    k2: int = Query(5, ge=0, le=15, description="Each ring-1 book's own neighbor count, for ring 2."),
    max_nodes: int = Query(60, ge=10, le=90),
    _: dict = Depends(_main.require_viewer),
) -> BookRelatedResponse:
    """A book's most vocabulary-related books, precomputed by
    `concordance book-similarity` (concordance/db.py's compute_book_similarity)
    -- lexical usage overlap (shared active words, IDF-weighted), NOT
    semantic similarity (that's word_graph's job, at the word level). Serves
    both the "Related books" list widget (read `nodes` where ring==1,
    already sorted by score) and the drill-down graph page (render the
    whole response) from a single cheap query -- unlike word_graph's
    multi-hop BFS, this is a direct top-k table read, so there's no
    separate expensive-vs-cheap split worth two endpoints for.

    Ring 2 (each ring-1 book's own top-k2 neighbors) follows word_graph's
    exact pattern one level up: a single LATERAL-joined query, budget-capped
    at max_nodes, closest-by-score kept first when trimming. Cheap here
    specifically because book_similarity is a precomputed table read, not
    word_graph's live vector distance -- no per-seed embedding lookup."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id)
                        FROM {_main.SCHEMA}.book b
                        LEFT JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        LEFT JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                        WHERE b.id = %s GROUP BY b.id, b.title, b.author""", (book_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="book not found")
        center = BookGraphNode(id=row[0], title=row[1], author=row[2], ring=0, word_count=row[3])

        cur.execute(f"""SELECT b.id, b.title, b.author, bs.score, bs.shared_word_count
                        FROM {_main.SCHEMA}.book_similarity bs
                        JOIN {_main.SCHEMA}.book b ON b.id = bs.book_b_id
                        WHERE bs.book_a_id = %s
                        ORDER BY bs.score DESC LIMIT %s""", (book_id, top_k))
        related = cur.fetchall()

        # Cross-links: real edges BETWEEN two displayed ring-1 neighbors, not
        # just center-to-neighbor -- without this, book_related is a literal
        # star graph (zero topological information beyond "these are the
        # neighbors"). Only surfaces links already stored because a neighbor
        # also has another displayed neighbor in ITS OWN top-k -- no new
        # similarity computation, same query shape authors_relatedness
        # already uses for its own cross-author edges.
        displayed = [book_id] + [r[0] for r in related]
        cur.execute(f"""SELECT book_a_id, book_b_id, score, shared_word_count
                        FROM {_main.SCHEMA}.book_similarity
                        WHERE book_a_id = ANY(%s) AND book_b_id = ANY(%s)
                          AND book_a_id <> %s AND book_b_id <> %s""",
                    (displayed, displayed, book_id, book_id))
        cross_rows = cur.fetchall()

        # Ring 2: one LATERAL per ring-1 seed, same shape as word_graph's own
        # ring-2 query.
        seed_ids = [r[0] for r in related]
        ring2_rows = []
        if k2 and seed_ids:
            cur.execute(f"""SELECT seed.book_id AS seed_id, nb.id, nb.title, nb.author, nb.score, nb.shared_word_count
                            FROM unnest(%s::int[]) AS seed(book_id)
                            CROSS JOIN LATERAL (
                                SELECT b2.id, b2.title, b2.author, bs2.score, bs2.shared_word_count
                                FROM {_main.SCHEMA}.book_similarity bs2
                                JOIN {_main.SCHEMA}.book b2 ON b2.id = bs2.book_b_id
                                WHERE bs2.book_a_id = seed.book_id AND bs2.book_b_id <> %s
                                ORDER BY bs2.score DESC
                                LIMIT %s
                            ) nb""", (seed_ids, book_id, k2))
            ring2_rows = cur.fetchall()

    nodes: dict[int, BookGraphNode] = {book_id: center}
    for r in related:
        nodes.setdefault(r[0], BookGraphNode(id=r[0], title=r[1], author=r[2], ring=1, word_count=None))
    edges = [
        BookGraphEdge(source=book_id, target=r[0], score=r[3], shared_word_count=r[4], is_center_edge=True)
        for r in related
    ]
    seen_pairs: set[frozenset] = set()

    def add_cross_edge(a, b, score, shared) -> None:
        pair = frozenset((a, b))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        edges.append(BookGraphEdge(source=a, target=b, score=score, shared_word_count=shared, is_center_edge=False))

    for a, b, score, shared in cross_rows:
        add_cross_edge(a, b, score, shared)

    # ring-2-only additions get trimmed first if we're over budget -- sort
    # globally by score (higher = closer) so the strongest second-hop books survive.
    ring2_new = [r for r in ring2_rows if r[1] not in nodes]
    ring2_new.sort(key=lambda r: r[4], reverse=True)
    budget_left = max_nodes - len(nodes)
    keep_ids = {r[1] for r in ring2_new[: max(budget_left, 0)]}

    for seed_id, wid, title, author, score, shared in ring2_rows:
        if wid in nodes:
            add_cross_edge(seed_id, wid, score, shared)  # cross-link to an existing node, no new node
        elif wid in keep_ids:
            nodes.setdefault(wid, BookGraphNode(id=wid, title=title, author=author, ring=2, word_count=None))
            add_cross_edge(seed_id, wid, score, shared)

    return BookRelatedResponse(center=center, nodes=list(nodes.values()), edges=edges)


class SharedWord(BaseModel):
    id: int
    lemma: str
    definition: str | None
    idf: float          # same ln(N/df) weighting compute_book_similarity/
                         # compute_author_similarity score on -- higher idf
                         # means this word contributed more to the score.


class SharedWordsResponse(BaseModel):
    shared_words: list[SharedWord]   # sorted by idf desc -- rarest/most
                                      # distinctive shared word first
    total_shared: int


@router.get("/api/browse/books/{book_id_a}/shared-words/{book_id_b}", response_model=SharedWordsResponse)
def book_shared_words(
    book_id_a: int,
    book_id_b: int,
    _: dict = Depends(_main.require_viewer),
) -> SharedWordsResponse:
    """The actual overlapping active vocabulary between two books -- "the
    what" behind book_related's score/shared_word_count ("the why"). Only
    words that pass compute_book_similarity's own max_df_fraction=0.5
    cutoff are returned, so this is consistent with the score: every word
    shown here is one that actually counted toward it, not a superset.
    Computed entirely on demand, bounded to exactly two books' shared
    vocabulary (tens to low hundreds of words) -- nothing like the earlier
    all-authors on-demand mistake, which broke because it recomputed the
    entire corpus per request."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT wb.book_id) FROM {_main.SCHEMA}.word_book wb
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id WHERE w.active""")
        n_books = cur.fetchone()[0]
        max_df = 0.5 * n_books if n_books else 0

        cur.execute(f"""WITH shared AS (
                            SELECT w.id, w.lemma, w.definition
                            FROM {_main.SCHEMA}.word w
                            WHERE w.active
                              AND EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb1
                                          WHERE wb1.word_id = w.id AND wb1.book_id = %s)
                              AND EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb2
                                          WHERE wb2.word_id = w.id AND wb2.book_id = %s)
                        )
                        SELECT s.id, s.lemma, s.definition, count(DISTINCT wb.book_id) AS df
                        FROM shared s
                        JOIN {_main.SCHEMA}.word_book wb ON wb.word_id = s.id
                        JOIN {_main.SCHEMA}.word w2 ON w2.id = wb.word_id AND w2.active
                        GROUP BY s.id, s.lemma, s.definition""", (book_id_a, book_id_b))
        rows = cur.fetchall()

    shared = sorted(
        (SharedWord(id=r[0], lemma=r[1], definition=r[2], idf=math.log(n_books / r[3]))
         for r in rows if r[3] <= max_df),
        key=lambda s: s.idf, reverse=True,
    )
    return SharedWordsResponse(shared_words=shared, total_shared=len(shared))


# --- /api/browse/books/relatedness, /map, /matrix, /dendrogram ------------------
#
# The book-level counterpart to the four global author views below --
# book_related above is the single-book ego graph (one book + its
# neighbors); these four are "every book at once", all reading
# concordance/db.py's precomputed book_similarity/book_cluster/
# book_cluster_run tables (populated by `concordance book-similarity` /
# `book-clustering` / `maintain`), same division of labor as the author
# versions: relatedness reads book_similarity directly (bounded by `limit`,
# the busiest-by-word-count books), while map/matrix/dendrogram all read
# ONE shared book_cluster_run computation pass so they never disagree with
# each other.

class BookRelatednessGraph(BaseModel):
    nodes: list[BookGraphNode]
    edges: list[BookGraphEdge]


@router.get("/api/browse/books/relatedness", response_model=BookRelatednessGraph)
def books_relatedness(
    top_k: int = Query(5, ge=1, le=20, description="Neighbors per book, from book_similarity's stored top-k."),
    limit: int = Query(60, ge=5, le=300, description="Cap on how many books appear at all, by word count. Ignored if scope=fame."),
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> BookRelatednessGraph:
    """The global all-books relatedness graph -- same reasoning as
    authors_relatedness one level up: real corpus scale (~11,000+ books)
    makes an unbounded graph both an expensive query and an unreadable
    hairball, so `limit` restricts it to the busiest books by (extracted-
    vocabulary) word count, and edges are only kept between two books that
    both made the cut. Deliberately a SEPARATE response type from
    BookRelatedResponse (the single-book ego graph), not that type reused
    with a fake center -- there's no real center on a global graph, only
    peers, and RelatednessGraph.jsx already handles a center-less
    {nodes, edges} response correctly (every node falls back to uniform
    "related" styling when `data.center` is undefined), exactly how
    AuthorRelatednessGraph already works for the equivalent author view.

    scope="fame" swaps the node-selection query for the same book_fame >=
    8 set book_cluster_fame holds (see compute_book_clustering's min_fame
    docstring) instead of top-by-word-count -- `limit` is ignored in this
    mode, same "no artificial cap" reasoning as the clustering tables
    themselves. Edges still come straight from book_similarity (this graph
    was never derived from the clustering run's own precomputed grid), just
    restricted to pairs where both ends are in the scoped node set."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "fame":
            cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.book_cluster_fame bcf ON bcf.book_id = b.id
                            JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                            GROUP BY b.id, b.title, b.author""")
        else:
            cur.execute(f"""SELECT b.id, b.title, b.author, count(DISTINCT w.id) AS word_count
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                            GROUP BY b.id, b.title, b.author
                            ORDER BY word_count DESC
                            LIMIT %s""", (limit,))
        books = cur.fetchall()
        book_ids = [r[0] for r in books]

        cur.execute(f"""SELECT book_a_id, book_b_id, score, shared_word_count FROM (
                            SELECT book_a_id, book_b_id, score, shared_word_count,
                                   row_number() OVER (PARTITION BY book_a_id ORDER BY score DESC) AS rn
                            FROM {_main.SCHEMA}.book_similarity
                            WHERE book_a_id = ANY(%s) AND book_b_id = ANY(%s)
                        ) ranked WHERE rn <= %s""", (book_ids, book_ids, top_k))
        rows = cur.fetchall()

    nodes = [BookGraphNode(id=r[0], title=r[1], author=r[2], ring=0, word_count=r[3]) for r in books]
    seen_pairs: set[frozenset] = set()
    edges: list[BookGraphEdge] = []
    for a, b, score, shared in rows:
        pair = frozenset((a, b))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edges.append(BookGraphEdge(source=a, target=b, score=score, shared_word_count=shared, is_center_edge=False))
    return BookRelatednessGraph(nodes=nodes, edges=edges)


class BookMapNode(BaseModel):
    id: int
    title: str
    author: str | None
    cluster_id: int
    x: float
    y: float
    word_count: int


class BookMapResponse(BaseModel):
    nodes: list[BookMapNode]


@router.get("/api/browse/books/map", response_model=BookMapResponse)
def books_map(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> BookMapResponse:
    """The cluster map: every book in the precomputed book_cluster table
    (top-N by word count -- see compute_book_clustering), positioned by
    classical MDS over the same IDF-weighted cosine distance book_similarity's
    scores use, colored by hierarchical cluster membership. Same rationale
    as authors_map one level up: a physics-simulation force-directed layout
    becomes an unstable hairball at this many nodes, where MDS position is
    principled (derived from the real similarity structure) instead.

    scope="fame" reads book_cluster_fame instead -- a second, independently
    computed run selected by book_fame.fame_score threshold rather than
    volume (see compute_book_clustering's min_fame docstring). Same table
    shape either way, so every field below means the same thing in both
    modes."""
    table = "book_cluster_fame" if scope == "fame" else "book_cluster"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT book_id, title, author, cluster_id, mds_x, mds_y, word_count
                        FROM {_main.SCHEMA}.{table}
                        ORDER BY cluster_id, title""")
        rows = cur.fetchall()
    nodes = [BookMapNode(id=r[0], title=r[1], author=r[2], cluster_id=r[3], x=r[4], y=r[5], word_count=r[6])
             for r in rows]
    return BookMapResponse(nodes=nodes)


class BookMatrixEntry(BaseModel):
    id: int
    title: str
    author: str | None


class BookMatrixCell(BaseModel):
    score: float
    shared_word_count: int


class BookMatrixResponse(BaseModel):
    books: list[BookMatrixEntry]
    grid: list[list[BookMatrixCell]]  # grid[i][j] compares books[i] to books[j]


@router.get("/api/browse/books/matrix", response_model=BookMatrixResponse)
def books_matrix(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> BookMatrixResponse:
    """The seriated similarity matrix/heatmap -- same top-N books as the
    cluster map, in compute_book_clustering's hierarchical-clustering leaf
    order so related books form visible blocks. Reads straight from
    book_cluster_run (no new computation), same as authors_matrix.
    scope="fame" reads book_cluster_fame_run instead -- see books_map's
    own scope docstring."""
    table = "book_cluster_fame_run" if scope == "fame" else "book_cluster_run"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT leaf_order, grid FROM {_main.SCHEMA}.{table} WHERE id = 1")
        row = cur.fetchone()
    if row is None:
        return BookMatrixResponse(books=[], grid=[])
    leaf_order, grid = row
    books = [BookMatrixEntry(id=b["id"], title=b["title"], author=b["author"]) for b in leaf_order]
    cells = [[BookMatrixCell(score=c[0], shared_word_count=c[1]) for c in grid_row] for grid_row in grid]
    return BookMatrixResponse(books=books, grid=cells)


class BookDendrogramNode(BaseModel):
    id: int | None = None            # set on leaves only
    title: str | None = None
    author: str | None = None
    size: int                        # number of leaves in this subtree
    distance: float | None = None    # merge height -- unset on leaves (they merge at nothing)
    left: "BookDendrogramNode | None" = None
    right: "BookDendrogramNode | None" = None


BookDendrogramNode.model_rebuild()


class BookDendrogramResponse(BaseModel):
    tree: BookDendrogramNode | None  # None if no clustering run has completed yet
    leaf_order: list[BookMatrixEntry]


@router.get("/api/browse/books/dendrogram", response_model=BookDendrogramResponse)
def books_dendrogram(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> BookDendrogramResponse:
    """The dendrogram: the same clustering run's linkage tree, straight
    from book_cluster_run (no new computation), same as authors_dendrogram.
    scope="fame" reads book_cluster_fame_run instead -- see books_map's own
    scope docstring."""
    table = "book_cluster_fame_run" if scope == "fame" else "book_cluster_run"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT tree_json, leaf_order FROM {_main.SCHEMA}.{table} WHERE id = 1")
        row = cur.fetchone()
    if row is None:
        return BookDendrogramResponse(tree=None, leaf_order=[])
    tree_json, leaf_order = row
    books = [BookMatrixEntry(id=b["id"], title=b["title"], author=b["author"]) for b in leaf_order]
    return BookDendrogramResponse(tree=tree_json, leaf_order=books)


# --- /api/browse/authors/{author}/related, /api/browse/authors/relatedness -----
#
# Both read concordance/db.py's precomputed author_similarity table --
# originally an on-demand per-request computation (the relatedness plan's
# own reasoning: "authors are dozens today, full O(n^2) pairwise is
# cheap"), until real data showed ~3,500 authors and a ~39s full-corpus
# computation time. Moved to the same precompute-table pattern book_related
# already used, populated by `concordance author-similarity` / `maintain`
# (see compute_author_similarity's docstring for the metric itself).

class AuthorGraphNode(BaseModel):
    id: str                  # author name -- string, unlike book/word graph
                              # ids. Divergence is deliberate, see the
                              # relatedness-visualization plan's API contract.
    ring: int                 # 0 = center, 1 = a related author. Always 0 on
                               # the global all-authors graph (no center there).
    book_count: int | None
    word_count: int | None    # populated for the center only, like BookGraphNode


class AuthorGraphEdge(BaseModel):
    source: str
    target: str
    score: float               # similarity -- HIGHER means more related, same
                                # convention (and same frontend link-length
                                # inversion requirement) as BookGraphEdge.score
    shared_word_count: int
    is_center_edge: bool       # see BookGraphEdge.is_center_edge -- same
                                # meaning, same honest-gap caveat, one level up.


class AuthorRelatedResponse(BaseModel):
    center: AuthorGraphNode
    nodes: list[AuthorGraphNode]
    edges: list[AuthorGraphEdge]


class AuthorRelatednessGraph(BaseModel):
    nodes: list[AuthorGraphNode]
    edges: list[AuthorGraphEdge]


@router.get("/api/browse/authors/{author}/related", response_model=AuthorRelatedResponse)
def author_related(
    author: str,
    top_k: int = Query(8, ge=1, le=20),
    k2: int = Query(5, ge=0, le=15, description="Each ring-1 author's own neighbor count, for ring 2."),
    max_nodes: int = Query(60, ge=10, le=90),
    _: dict = Depends(_main.require_viewer),
) -> AuthorRelatedResponse:
    """An author's most vocabulary-related authors, precomputed by
    `concordance author-similarity` -- same lexical-overlap metric and same
    top-k-table-read shape as book_related, including its ring-2 expansion
    (see book_related's docstring for the pattern/reasoning). PLACEHOLDER_AUTHORS
    ("Various", "Unknown Author", ...) are aggregation labels, not real authors --
    author_similarity never has rows for them (see compute_author_similarity),
    so they 404 here too rather than returning a center with an always-empty
    related list."""
    if author in PLACEHOLDER_AUTHORS:
        raise HTTPException(status_code=404, detail="author not found")
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.id), count(DISTINCT w.id)
                        FROM {_main.SCHEMA}.book b
                        LEFT JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        LEFT JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                        WHERE b.author = %s""", (author,))
        row = cur.fetchone()
        if row is None or row[0] == 0:
            raise HTTPException(status_code=404, detail="author not found")
        center = AuthorGraphNode(id=author, ring=0, book_count=row[0], word_count=row[1])

        cur.execute(f"""SELECT author_b, score, shared_word_count
                        FROM {_main.SCHEMA}.author_similarity
                        WHERE author_a = %s
                        ORDER BY score DESC LIMIT %s""", (author, top_k))
        related = cur.fetchall()

        # Cross-links -- see book_related's identical reasoning/caveat.
        displayed = [author] + [r[0] for r in related]
        cur.execute(f"""SELECT author_a, author_b, score, shared_word_count
                        FROM {_main.SCHEMA}.author_similarity
                        WHERE author_a = ANY(%s) AND author_b = ANY(%s)
                          AND author_a <> %s AND author_b <> %s""",
                    (displayed, displayed, author, author))
        cross_rows = cur.fetchall()

        # Ring 2 -- see book_related's identical LATERAL pattern, one level up.
        seed_names = [r[0] for r in related]
        ring2_rows = []
        if k2 and seed_names:
            cur.execute(f"""SELECT seed.author_name AS seed_id, nb.author_b, nb.score, nb.shared_word_count
                            FROM unnest(%s::text[]) AS seed(author_name)
                            CROSS JOIN LATERAL (
                                SELECT as2.author_b, as2.score, as2.shared_word_count
                                FROM {_main.SCHEMA}.author_similarity as2
                                WHERE as2.author_a = seed.author_name AND as2.author_b <> %s
                                ORDER BY as2.score DESC
                                LIMIT %s
                            ) nb""", (seed_names, author, k2))
            ring2_rows = cur.fetchall()

    nodes: dict[str, AuthorGraphNode] = {author: center}
    for r in related:
        nodes.setdefault(r[0], AuthorGraphNode(id=r[0], ring=1, book_count=None, word_count=None))
    edges = [
        AuthorGraphEdge(source=author, target=r[0], score=r[1], shared_word_count=r[2], is_center_edge=True)
        for r in related
    ]
    seen_pairs: set[frozenset] = set()

    def add_cross_edge(a, b, score, shared) -> None:
        pair = frozenset((a, b))
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        edges.append(AuthorGraphEdge(source=a, target=b, score=score, shared_word_count=shared, is_center_edge=False))

    for a, b, score, shared in cross_rows:
        add_cross_edge(a, b, score, shared)

    ring2_new = [r for r in ring2_rows if r[1] not in nodes]
    ring2_new.sort(key=lambda r: r[2], reverse=True)
    budget_left = max_nodes - len(nodes)
    keep_ids = {r[1] for r in ring2_new[: max(budget_left, 0)]}

    for seed_id, name, score, shared in ring2_rows:
        if name in nodes:
            add_cross_edge(seed_id, name, score, shared)
        elif name in keep_ids:
            nodes.setdefault(name, AuthorGraphNode(id=name, ring=2, book_count=None, word_count=None))
            add_cross_edge(seed_id, name, score, shared)

    return AuthorRelatedResponse(center=center, nodes=list(nodes.values()), edges=edges)


@router.get("/api/browse/authors/{author_a}/shared-words/{author_b}", response_model=SharedWordsResponse)
def author_shared_words(
    author_a: str,
    author_b: str,
    _: dict = Depends(_main.require_viewer),
) -> SharedWordsResponse:
    """Same as book_shared_words, one level up: the actual overlapping
    active vocabulary between two authors (a word counts if EITHER author
    used it in any of their books), respecting compute_author_similarity's
    own max_df_fraction=0.5 cutoff and PLACEHOLDER_AUTHORS exclusion so
    author-df here means the same thing it means there."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.author) FROM {_main.SCHEMA}.word_book wb
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                        JOIN {_main.SCHEMA}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))""", (list(PLACEHOLDER_AUTHORS),))
        n_authors = cur.fetchone()[0]
        max_df = 0.5 * n_authors if n_authors else 0

        cur.execute(f"""WITH shared AS (
                            SELECT w.id, w.lemma, w.definition
                            FROM {_main.SCHEMA}.word w
                            WHERE w.active
                              AND EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb1
                                          JOIN {_main.SCHEMA}.book b1 ON b1.id = wb1.book_id
                                          WHERE wb1.word_id = w.id AND b1.author = %s)
                              AND EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb2
                                          JOIN {_main.SCHEMA}.book b2 ON b2.id = wb2.book_id
                                          WHERE wb2.word_id = w.id AND b2.author = %s)
                        )
                        SELECT s.id, s.lemma, s.definition, count(DISTINCT b.author) AS df
                        FROM shared s
                        JOIN {_main.SCHEMA}.word_book wb ON wb.word_id = s.id
                        JOIN {_main.SCHEMA}.book b ON b.id = wb.book_id
                        JOIN {_main.SCHEMA}.word w2 ON w2.id = wb.word_id AND w2.active
                        WHERE b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY s.id, s.lemma, s.definition""",
                    (author_a, author_b, list(PLACEHOLDER_AUTHORS)))
        rows = cur.fetchall()

    shared = sorted(
        (SharedWord(id=r[0], lemma=r[1], definition=r[2], idf=math.log(n_authors / r[3]))
         for r in rows if r[3] <= max_df),
        key=lambda s: s.idf, reverse=True,
    )
    return SharedWordsResponse(shared_words=shared, total_shared=len(shared))


@router.get("/api/browse/authors/relatedness", response_model=AuthorRelatednessGraph)
def authors_relatedness(
    top_k: int = Query(5, ge=1, le=20, description="Neighbors per author, from author_similarity's stored top-k."),
    limit: int = Query(60, ge=5, le=300, description="Cap on how many authors appear at all, by book count. Ignored if scope=fame."),
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> AuthorRelatednessGraph:
    """The global all-authors relatedness graph (secondary page, per the
    relatedness-visualization plan). Real corpus scale (~3,500 authors)
    makes an unbounded version both an expensive query and, more
    importantly, an unreadable hairball in a force-directed layout -- so
    `limit` restricts it to the `limit` authors with the most books
    (proxy for "most represented in the corpus", so the busiest, most
    interconnected part of the graph is what's shown by default), and
    edges are only kept between two authors that both made the cut.

    scope="fame" swaps the node-selection query for the same author_fame
    >= 8 set author_cluster_fame holds (see compute_author_clustering's
    min_fame docstring) instead of top-by-book-count -- `limit` is ignored
    in this mode. Edges still come straight from author_similarity (this
    graph was never derived from the clustering run's own precomputed
    grid), just restricted to pairs where both ends are in the scoped node
    set."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "fame":
            cur.execute(f"""SELECT b.author, count(DISTINCT b.id) AS book_count, count(DISTINCT w.id) AS word_count
                            FROM {_main.SCHEMA}.book b
                            JOIN {_main.SCHEMA}.author_cluster_fame acf ON acf.author = b.author
                            LEFT JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            LEFT JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                            GROUP BY b.author""")
        else:
            cur.execute(f"""SELECT b.author, count(DISTINCT b.id) AS book_count, count(DISTINCT w.id) AS word_count
                            FROM {_main.SCHEMA}.book b
                            LEFT JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                            LEFT JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                            WHERE b.author IS NOT NULL AND b.author <> ''
                              AND NOT (b.author = ANY(%s))
                            GROUP BY b.author
                            ORDER BY book_count DESC
                            LIMIT %s""", (list(PLACEHOLDER_AUTHORS), limit))
        authors = cur.fetchall()
        author_names = [r[0] for r in authors]

        cur.execute(f"""SELECT author_a, author_b, score, shared_word_count FROM (
                            SELECT author_a, author_b, score, shared_word_count,
                                   row_number() OVER (PARTITION BY author_a ORDER BY score DESC) AS rn
                            FROM {_main.SCHEMA}.author_similarity
                            WHERE author_a = ANY(%s) AND author_b = ANY(%s)
                        ) ranked WHERE rn <= %s""", (author_names, author_names, top_k))
        rows = cur.fetchall()

    nodes = [AuthorGraphNode(id=r[0], ring=0, book_count=r[1], word_count=r[2]) for r in authors]
    seen_pairs: set[frozenset] = set()
    edges: list[AuthorGraphEdge] = []
    for a, b, score, shared in rows:
        pair = frozenset((a, b))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        # No "center" concept on the global graph -- every edge here is a
        # peer relationship, so is_center_edge is always False.
        edges.append(AuthorGraphEdge(source=a, target=b, score=score, shared_word_count=shared, is_center_edge=False))
    return AuthorRelatednessGraph(nodes=nodes, edges=edges)


class AuthorMapNode(BaseModel):
    author: str
    cluster_id: int
    x: float
    y: float
    book_count: int


class AuthorMapResponse(BaseModel):
    nodes: list[AuthorMapNode]


@router.get("/api/browse/authors/map", response_model=AuthorMapResponse)
def authors_map(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> AuthorMapResponse:
    """The cluster map: every author in the precomputed author_cluster table
    (top-N by book count -- see concordance/db.py's compute_author_clustering),
    positioned by classical MDS over the same IDF-weighted cosine distance
    author_similarity's scores use, colored by hierarchical cluster
    membership. Default tab on the global authors page -- position and
    color here are both principled (derived from the actual similarity
    structure via clustering + MDS), unlike the force-directed graph's
    physics-simulation compromise layout, which carries no such guarantee
    and (per real usage) becomes an unstable hairball at this many nodes.

    scope="fame" reads author_cluster_fame instead -- a second,
    independently computed run selected by author_fame.fame_score
    threshold rather than book count (see compute_author_clustering's
    min_fame docstring). Same table shape either way."""
    table = "author_cluster_fame" if scope == "fame" else "author_cluster"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT author, cluster_id, mds_x, mds_y, book_count
                        FROM {_main.SCHEMA}.{table}
                        ORDER BY cluster_id, author""")
        rows = cur.fetchall()
    nodes = [AuthorMapNode(author=r[0], cluster_id=r[1], x=r[2], y=r[3], book_count=r[4]) for r in rows]
    return AuthorMapResponse(nodes=nodes)


class AuthorMatrixCell(BaseModel):
    score: float
    shared_word_count: int


class AuthorMatrixResponse(BaseModel):
    authors: list[str]              # seriated (leaf) order -- row/column labels, in order
    grid: list[list[AuthorMatrixCell]]  # grid[i][j] compares authors[i] to authors[j]


@router.get("/api/browse/authors/matrix", response_model=AuthorMatrixResponse)
def authors_matrix(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> AuthorMatrixResponse:
    """The seriated similarity matrix/heatmap: the same top-N authors as
    the cluster map, ordered by compute_author_clustering's hierarchical-
    clustering leaf order so similar authors sit near each other and form
    visible blocks -- an alphabetical or random order would scatter
    related authors across the grid instead. Unlike author_similarity's
    own top-k-only storage, every cell here is a genuine pairwise score,
    including pairs that missed both sides' top-k cutoff -- this is the
    one place that question has an answer at all. Reads straight from
    author_cluster_run, computed once alongside the map and dendrogram
    (no new computation). scope="fame" reads author_cluster_fame_run
    instead -- see authors_map's own scope docstring."""
    table = "author_cluster_fame_run" if scope == "fame" else "author_cluster_run"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT leaf_order, grid FROM {_main.SCHEMA}.{table} WHERE id = 1")
        row = cur.fetchone()
    if row is None:
        return AuthorMatrixResponse(authors=[], grid=[])
    leaf_order, grid = row
    cells = [[AuthorMatrixCell(score=c[0], shared_word_count=c[1]) for c in grid_row] for grid_row in grid]
    return AuthorMatrixResponse(authors=leaf_order, grid=cells)


class DendrogramNode(BaseModel):
    author: str | None = None       # set on leaves only
    size: int                       # number of leaves in this subtree
    distance: float | None = None   # merge height -- unset on leaves (they merge at nothing)
    left: "DendrogramNode | None" = None
    right: "DendrogramNode | None" = None


DendrogramNode.model_rebuild()


class AuthorDendrogramResponse(BaseModel):
    tree: DendrogramNode | None     # None if no clustering run has completed yet
    leaf_order: list[str]


@router.get("/api/browse/authors/dendrogram", response_model=AuthorDendrogramResponse)
def authors_dendrogram(
    scope: Literal["volume", "fame"] = "volume",
    _: dict = Depends(_main.require_viewer),
) -> AuthorDendrogramResponse:
    """The dendrogram: the same clustering run's linkage tree, straight
    from author_cluster_run (no new computation) -- the clearest
    hierarchical narrative of the three global views ("this author's
    whole branch shares X"), and the one that scales best to more authors
    since a deep tree can be explored by collapsing subtrees rather than
    needing every leaf visible at once like the map or matrix do.
    scope="fame" reads author_cluster_fame_run instead -- see authors_map's
    own scope docstring."""
    table = "author_cluster_fame_run" if scope == "fame" else "author_cluster_run"
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT tree_json, leaf_order FROM {_main.SCHEMA}.{table} WHERE id = 1")
        row = cur.fetchone()
    if row is None:
        return AuthorDendrogramResponse(tree=None, leaf_order=[])
    tree_json, leaf_order = row
    return AuthorDendrogramResponse(tree=tree_json, leaf_order=leaf_order)


# --- /api/browse/words ----------------------------------------------------------

class BrowseWordRow(BaseModel):
    id: int
    lemma: str
    part_of_speech: str | None
    definition: str | None
    difficulty: float | None
    archaic: str | None
    quizzable: bool | None
    book_count: int  # distinct books this word appears in -- see word_detail's "Sources (N)"


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
    top_code: list[str] = Query([]),
    all_top_code: list[str] = Query([]),
    uncategorized: bool = False,
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    unscored_only: bool = False,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    exclusive: bool = False,
    q: str | None = None,
    definition_q: str | None = None,
    definition_contains: str | None = None,
    letter: str | None = Query(None, min_length=1, max_length=1),
    random: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort: Literal["lemma", "difficulty", "part_of_speech", "book_count"] = "lemma",
    dir: Literal["asc", "desc"] = "asc",
    _: dict = Depends(_main.require_viewer),
) -> BrowseWordPage:
    for code in top_code:
        if code not in _ALL_CODES:
            raise HTTPException(404, f"unknown category code {code!r}")
    for code in all_top_code:
        if code not in _ALL_CODES:
            raise HTTPException(404, f"unknown category code {code!r}")
    if uncategorized and (domain or top_code or all_top_code):
        raise HTTPException(400, "uncategorized is mutually exclusive with domain/top_code/all_top_code")
    if unscored_only and (difficulty_min is not None or difficulty_max is not None):
        raise HTTPException(400, "unscored_only is mutually exclusive with difficulty_min/difficulty_max")
    if q and definition_q:
        raise HTTPException(400, "q and definition_q are mutually exclusive")
    if definition_contains and (q or definition_q):
        raise HTTPException(400, "definition_contains is mutually exclusive with q/definition_q")
    filters, params = _build_word_filters(
        author, book_id, domain, difficulty_min, difficulty_max, archaic, pos, quizzable_only,
        top_code=top_code, all_top_code=all_top_code, uncategorized=uncategorized, unscored_only=unscored_only,
        exclusive=exclusive,
    )
    if letter:
        filters.append("w.lemma_lc LIKE %s")
        params.append(f"{letter.lower()}%")
    q_is_pattern = bool(q) and ("%" in q or "_" in q)
    if q_is_pattern:
        # SQL-style wildcards typed directly into the search box -- % for
        # any run of characters, _ for exactly one -- e.g. "b_ll" finds
        # ball/bell/bill, "%tion" finds every word ending in "tion". A
        # literal LIKE against lemma_lc, not the fuzzy trigram match below:
        # once the user's typed an actual pattern, "close enough" relevance
        # scoring is the wrong tool -- they want every literal match, not a
        # ranked top-N guess. params.append (not %-formatted into the SQL
        # text) keeps this a normal bound parameter, so the wildcards are
        # LIKE syntax, never a SQL-injection vector.
        filters.append("w.lemma_lc LIKE %s")
        params.append(q.lower())
    elif q:
        filters.append("similarity(w.lemma, %s) > 0.1")
        params.append(q)
    if definition_q:
        # word_similarity, not similarity() -- similarity() compares the
        # WHOLE definition string against the query, which scores a short
        # search term against a full sentence terribly; word_similarity
        # instead asks "does some word-length span of the definition match
        # this query well," the right question for "does this term appear
        # meaningfully in the definition." Same pg_trgm extension q already
        # uses above, just the other one of its two comparison functions.
        filters.append("w.definition IS NOT NULL AND word_similarity(%s, w.definition) > 0.3")
        params.append(definition_q)
    if definition_contains:
        # Plain substring containment, not fuzzy -- the header definition
        # search's Enter behavior (see HeaderSearch.jsx): "show me every
        # word whose definition contains this exact text," landing here on
        # /app/words as a real, sortable, fully-paginated browsable list
        # (unlike the header's own live dropdown, which stays fuzzy/top-8
        # via definition_q above -- a good typo-tolerant preview, but not
        # the right tool for "show me everything"). No relevance ORDER BY
        # override below, unlike q/definition_q -- containment has no
        # notion of a "better" match, so the requested sort just applies
        # normally, same as any other filter.
        filters.append("w.definition ILIKE %s")
        params.append(f"%{definition_contains}%")
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
        elif q_is_pattern:
            # No relevance ranking for a literal pattern -- same reasoning
            # as definition_contains below: containment/pattern-matching has
            # no notion of a "better" match, so just go alphabetical.
            order_by = "w.lemma_lc ASC"
            limit = page_size
        elif q:
            # Best text match wins over the requested sort -- "search within
            # these filters" and "alphabetical/difficulty order" aren't
            # reconcilable in one ORDER BY, and search intent dominates.
            order_by = "similarity(w.lemma, %s) DESC"
            params = params + [q]
            limit = page_size
        elif definition_q:
            order_by = "word_similarity(%s, w.definition) DESC"
            params = params + [definition_q]
            limit = page_size
        else:
            order_col = _WORD_SORT_COLUMNS[sort]
            order_by = f"{order_col} {'ASC' if dir == 'asc' else 'DESC'} NULLS LAST, w.lemma_lc ASC"
            limit = page_size
        offset = 0 if random else (page - 1) * page_size

        cur.execute(
            f"""SELECT w.id, w.lemma, w.part_of_speech, w.definition,
                       wd.difficulty, wd.archaic, wd.quizzable,
                       (SELECT count(*) FROM {_main.SCHEMA}.word_book wb2
                        WHERE wb2.word_id = w.id) AS book_count
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
                      difficulty=r[4], archaic=r[5], quizzable=r[6], book_count=r[7])
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
    unscored_only: bool = False,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    exclusive: bool = False,
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
    risk breaking already-shipped UI for the sake of this one.

    No `domain`/`uncategorized` param here -- the bucket breakdown (including
    the uncategorized count below) IS this endpoint's output, so it can't
    also be an input filter; `unscored_only`/`exclusive` (like
    `difficulty_min/max`) are fair game since they narrow which words get
    bucketed, not which bucket a word lands in."""
    base_filters, base_params = _build_word_filters(
        author, book_id, [], difficulty_min, difficulty_max, archaic, pos, quizzable_only,
        unscored_only=unscored_only, exclusive=exclusive,
    )
    where = " AND ".join(base_filters)

    with _main.get_conn() as conn, conn.cursor() as cur:
        buckets = _bucket_counts(cur, where, base_params)

        uncat_filters, uncat_params = _build_word_filters(
            author, book_id, [], difficulty_min, difficulty_max, archaic, pos, quizzable_only,
            uncategorized=True, unscored_only=unscored_only, exclusive=exclusive,
        )
        uncat_where = " AND ".join(uncat_filters)
        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {uncat_where}""",
            uncat_params,
        )
        uncategorized_count = cur.fetchone()[0]
        buckets.append(DomainBucketCount(bucket="uncategorized", name="Uncategorized", word_count=uncategorized_count))

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
    uncategorized: bool = False,
    archaic: list[str] = Query([]),
    pos: list[str] = Query([]),
    quizzable_only: bool = False,
    exclusive: bool = False,
    band_width: int = Query(20, ge=5, le=50),
    _: dict = Depends(_main.require_viewer),
) -> list[DifficultyBandCount]:
    """Word count per difficulty band, conditioned on every OTHER active
    facet, plus one explicit "not yet scored" band -- the mechanism that
    makes the corpus's current sparse difficulty coverage honest in the UI
    rather than silently vanishing once a difficulty filter narrows it.

    No `difficulty_min/max`/`unscored_only` here -- the band breakdown IS
    this endpoint's output; `domain`/`uncategorized`/`exclusive` (like the
    other facets) are fair game since they narrow which words get banded,
    not which band a word lands in."""
    base_filters, base_params = _build_word_filters(
        author, book_id, domain, None, None, archaic, pos, quizzable_only,
        uncategorized=uncategorized, exclusive=exclusive,
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

        unscored_filters, unscored_params = _build_word_filters(
            author, book_id, domain, None, None, archaic, pos, quizzable_only,
            uncategorized=uncategorized, unscored_only=True, exclusive=exclusive,
        )
        unscored_where = " AND ".join(unscored_filters)
        cur.execute(
            f"""SELECT count(*) FROM {_main.SCHEMA}.word w
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {unscored_where}""",
            unscored_params,
        )
        unscored = cur.fetchone()[0]
        results.append(DifficultyBandCount(band_min=None, band_max=None, label="Not yet scored",
                                            word_count=unscored))
    return results


# --- /api/browse/unique-word-histogram --------------------------------------------
# _UNIQUE_WORD_BUCKETS / _unique_word_bucket_label live up near
# _build_word_filters (shared with the unique_word_bucket click-through
# filter on /api/browse/books and /api/browse/authors).

class UniqueWordBucket(BaseModel):
    label: str
    count: int  # number of books (or authors) whose exclusive-word count falls in this bucket


@router.get("/api/browse/unique-word-histogram", response_model=list[UniqueWordBucket])
def browse_unique_word_histogram(
    scope: Literal["book", "author"] = "book",
    _: dict = Depends(_main.require_viewer),
) -> list[UniqueWordBucket]:
    """Distribution of "how many of this book's (or author's) active words
    appear NOWHERE else in the corpus" -- the same exclusivity semantics
    /api/browse/words?exclusive=true uses for one book/author at a time,
    computed here for every book/author at once via a single aggregate
    query (a per-row NOT EXISTS, workable for one book, would mean 20k+
    separate correlated subqueries at this scale). A word with exactly one
    word_book row is exclusive to that row's book; "exclusive to an author"
    groups by author instead of book_id first, same MIN-collapse trick.
    Every bucket in _UNIQUE_WORD_BUCKETS is always present (count=0 rather
    than omitted) so the frontend's x-axis never reflows between scopes."""
    from collections import Counter
    s = _main.SCHEMA

    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "book":
            cur.execute(
                f"""WITH single_book_words AS (
                        SELECT wb.word_id, min(wb.book_id) AS book_id
                        FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id AND w.active
                        GROUP BY wb.word_id
                        HAVING count(*) = 1
                    )
                    SELECT b.id, count(sbw.word_id)
                    FROM {s}.book b
                    LEFT JOIN single_book_words sbw ON sbw.book_id = b.id
                    GROUP BY b.id"""
            )
        else:
            # PLACEHOLDER_AUTHORS excluded on both sides -- from the "which
            # authors get a row" list AND from counting as a co-occurrence
            # that would disqualify some other real author's exclusivity --
            # so this matches browse_authors, which already excludes them
            # from its own listing (see _unique_word_bucket_filter's same
            # exclusion for why the click-through count has to agree).
            placeholders = list(PLACEHOLDER_AUTHORS)
            cur.execute(
                f"""WITH single_author_words AS (
                        SELECT wb.word_id, min(b.author) AS author
                        FROM {s}.word_book wb
                        JOIN {s}.book b ON b.id = wb.book_id AND b.author IS NOT NULL
                            AND b.author != '' AND b.author != ALL(%s)
                        JOIN {s}.word w ON w.id = wb.word_id AND w.active
                        GROUP BY wb.word_id
                        HAVING count(DISTINCT b.author) = 1
                    )
                    SELECT a.author, count(saw.word_id)
                    FROM (SELECT DISTINCT author FROM {s}.book
                          WHERE author IS NOT NULL AND author != '' AND author != ALL(%s)) a
                    LEFT JOIN single_author_words saw ON saw.author = a.author
                    GROUP BY a.author""",
                (placeholders, placeholders),
            )
        counts = Counter(_unique_word_bucket_label(row[1]) for row in cur.fetchall())

    return [UniqueWordBucket(label=label, count=counts.get(label, 0)) for _, _, label in _UNIQUE_WORD_BUCKETS]


# --- /api/browse/fame-histogram ----------------------------------------------
# Reuses UniqueWordBucket's {label, count} shape -- same reason the frontend
# reuses DifficultyHistogram to render this: one generic bar-chart shape,
# not a new component per histogram.

_FAME_SCORE_LABELS = [str(i) for i in range(1, 11)]
_FAME_UNSCORED_LABEL = "Not yet scored"


@router.get("/api/browse/fame-histogram", response_model=list[UniqueWordBucket])
def browse_fame_histogram(
    scope: Literal["book", "author"] = "book",
    _: dict = Depends(_main.require_viewer),
) -> list[UniqueWordBucket]:
    """Distribution of book_fame/author_fame scores (an absolute 1-10 scale,
    see concordance/fame.py's own module docstring for what it measures and
    why most scores sit low) across every book, or every real author. Fame
    is computed by a separate, manually-run CLI command (`book-fame`/
    `author-fame`), deliberately excluded from `maintain` -- so "not yet
    scored" being the largest bar is expected, not a bug. Every score 1-10
    is always present (count=0 rather than omitted) so the x-axis never
    reflows; "Not yet scored" is appended last, matching
    /api/browse/difficulty-bands' own convention for its unscored pseudo-band."""
    from collections import Counter
    s = _main.SCHEMA

    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "book":
            cur.execute(
                f"""SELECT bf.fame_score
                    FROM {s}.book b
                    LEFT JOIN {s}.book_fame bf ON bf.book_id = b.id"""
            )
        else:
            placeholders = list(PLACEHOLDER_AUTHORS)
            cur.execute(
                f"""SELECT af.fame_score
                    FROM (SELECT DISTINCT author FROM {s}.book
                          WHERE author IS NOT NULL AND author != '' AND author != ALL(%s)) a
                    LEFT JOIN {s}.author_fame af ON af.author = a.author""",
                (placeholders,),
            )
        scores = [row[0] for row in cur.fetchall()]

    # round(), not floor/truncate -- fame_score is a double but the LLM
    # judge is always prompted for an integer 1-10 (see fame.py's own
    # prompt), so this is purely defensive against float drift, never
    # expected to actually change a value.
    counts = Counter(_FAME_UNSCORED_LABEL if score is None else str(round(score)) for score in scores)
    labels = _FAME_SCORE_LABELS + [_FAME_UNSCORED_LABEL]
    return [UniqueWordBucket(label=label, count=counts.get(label, 0)) for label in labels]


# --- /api/browse/overall-difficulty-histogram -------------------------------

@router.get("/api/browse/overall-difficulty-histogram", response_model=list[DifficultyBandCount])
def browse_overall_difficulty_histogram(
    scope: Literal["book", "author"] = "book",
    _: dict = Depends(_main.require_viewer),
) -> list[DifficultyBandCount]:
    """Distribution of overall_difficulty (the percent_rank(mean_difficulty)/
    percent_rank(density) blend -- see BookRow/AuthorRow's own field comments)
    across every book, or every real author -- unfiltered, same "whole
    corpus, no facets applied" scope /api/browse/fame-histogram uses, NOT
    difficulty-bands' "conditioned on every other active facet" scope. That
    conditioning only makes sense for word-level difficulty (a raw, stored
    column); overall_difficulty is already a corpus-wide percentile, so
    narrowing the corpus first would change what the percentile itself means
    out from under the chart.

    The base population is reproduced via _build_word_filters(None, [], [],
    None, None, [], [], False) rather than hand-rolled -- for scope='book'
    this collapses to plain `w.active`, for scope='author' the two extra
    `b.author IS NOT NULL`/`!= ALL(PLACEHOLDER_AUTHORS)` conditions
    browse_authors always appends are added the same way it does -- so this
    histogram's bars are guaranteed to bucket the exact same population
    browse_books/browse_authors themselves would return with no filters
    active, not a subtly different one."""
    s = _main.SCHEMA
    base_filters, base_params = _build_word_filters(None, [], [], None, None, [], [], False)
    where = " AND ".join(base_filters)

    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "book":
            cur.execute(
                f"""WITH book_base AS (
                        SELECT b.id, b.distinct_nonstop_word_count,
                               avg(wd.difficulty) AS mean_difficulty,
                               CASE WHEN b.distinct_nonstop_word_count > 0
                                    THEN count(DISTINCT w.id)::float / b.distinct_nonstop_word_count END AS density
                        FROM {s}.book b
                        JOIN {s}.word_book wb ON wb.book_id = b.id
                        JOIN {s}.word w ON w.id = wb.word_id
                        LEFT JOIN {s}.word_difficulty wd ON wd.word_id = w.id
                        WHERE {where}
                        GROUP BY b.id, b.distinct_nonstop_word_count
                    ),
                    diff_rank AS (
                        SELECT id, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                        FROM book_base WHERE mean_difficulty IS NOT NULL
                    ),
                    dens_rank AS (
                        SELECT id, percent_rank() OVER (ORDER BY density) AS dens_pct
                        FROM book_base WHERE density IS NOT NULL
                    )
                    SELECT CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                                THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                           END AS overall_difficulty
                    FROM book_base bb
                    LEFT JOIN diff_rank dr ON dr.id = bb.id
                    LEFT JOIN dens_rank de ON de.id = bb.id""",
                base_params,
            )
        else:
            placeholders = list(PLACEHOLDER_AUTHORS)
            cur.execute(
                f"""WITH author_word_ids AS (
                        SELECT DISTINCT b.author, w.id AS word_id
                        FROM {s}.book b
                        JOIN {s}.word_book wb ON wb.book_id = b.id
                        JOIN {s}.word w ON w.id = wb.word_id
                        WHERE {where} AND b.author IS NOT NULL AND b.author != ALL(%s)
                    ),
                    author_word_stats AS (
                        SELECT awi.author, avg(wd.difficulty) AS mean_difficulty
                        FROM author_word_ids awi
                        LEFT JOIN {s}.word_difficulty wd ON wd.word_id = awi.word_id
                        GROUP BY awi.author
                    ),
                    book_word_counts AS (
                        SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                               count(DISTINCT w.id) AS book_word_count
                        FROM {s}.book b
                        JOIN {s}.word_book wb ON wb.book_id = b.id
                        JOIN {s}.word w ON w.id = wb.word_id
                        LEFT JOIN {s}.word_difficulty wd ON wd.word_id = w.id
                        WHERE {where} AND b.author IS NOT NULL AND b.author != ALL(%s)
                        GROUP BY b.author, b.id, b.distinct_nonstop_word_count
                    ),
                    author_density AS (
                        SELECT author, avg(density) AS density
                        FROM (
                            SELECT author,
                                   CASE WHEN distinct_nonstop_word_count > 0
                                        THEN book_word_count::float / distinct_nonstop_word_count END AS density
                            FROM book_word_counts
                        ) bd
                        WHERE density IS NOT NULL
                        GROUP BY author
                    ),
                    author_base AS (
                        SELECT aws.author, aws.mean_difficulty, ad.density
                        FROM author_word_stats aws
                        LEFT JOIN author_density ad ON ad.author = aws.author
                    ),
                    diff_rank AS (
                        SELECT author, percent_rank() OVER (ORDER BY mean_difficulty) AS diff_pct
                        FROM author_base WHERE mean_difficulty IS NOT NULL
                    ),
                    dens_rank AS (
                        SELECT author, percent_rank() OVER (ORDER BY density) AS dens_pct
                        FROM author_base WHERE density IS NOT NULL
                    )
                    SELECT CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                                THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                           END AS overall_difficulty
                    FROM author_base ab
                    LEFT JOIN diff_rank dr ON dr.author = ab.author
                    LEFT JOIN dens_rank de ON de.author = ab.author""",
                (*base_params, placeholders, *base_params, placeholders),
            )
        scores = [row[0] for row in cur.fetchall()]

    scored = [float(sc) for sc in scores if sc is not None]
    results = []
    band_min = 0.0
    while band_min < 100:
        band_max = min(band_min + _OVERALL_DIFFICULTY_BAND_WIDTH, 100)
        is_last = band_max >= 100
        count = sum(1 for x in scored if band_min <= x <= band_max) if is_last \
            else sum(1 for x in scored if band_min <= x < band_max)
        results.append(DifficultyBandCount(
            band_min=band_min, band_max=band_max,
            label=f"{int(band_min)}-{int(band_max)}", word_count=count,
        ))
        band_min = band_max
    results.append(DifficultyBandCount(band_min=None, band_max=None, label=_OVERALL_DIFFICULTY_UNSCORED_LABEL,
                                        word_count=len(scores) - len(scored)))
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


# --- /api/browse/category-counts --------------------------------------------------

class CategoryCount(BaseModel):
    code: str
    name: str
    bucket: str | None
    word_count: int


def _child_subtree_counts(cur, children: list[dict]) -> dict[str, int]:
    """Whole-subtree word count for each of `children` (siblings, one level
    down from a common parent) -- one query per child, not a single GROUP
    BY: each child's subtree boundary (usas.subtree_sql) isn't a uniform
    prefix expression across siblings of different code lengths, unlike the
    level-0 case where `left(c.code, 1)` alone is that uniform expression.
    Never more than 15 queries (the max real child count of any code)."""
    counts = {}
    for child in children:
        exact, like = usas.subtree_sql(child["code"])
        cur.execute(
            f"""SELECT count(DISTINCT w.id)
                FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = w.id
                JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                WHERE w.active AND (c.code = %s OR c.code LIKE %s)""",
            (exact, like),
        )
        counts[child["code"]] = cur.fetchone()[0]
    return counts


@router.get("/api/browse/category-counts", response_model=list[CategoryCount])
def browse_category_counts(
    bucket: str | None = None,
    parent: str | None = None,
    _: dict = Depends(_main.require_viewer),
) -> list[CategoryCount]:
    """Word count per USAS category, one tier below either `bucket` (a 6-hue
    color bucket's member top-level fields, 21 total unscoped) or `parent`
    (any real USAS code's direct children, e.g. "I" -> I1..I4, "I2" ->
    I2.1/I2.2) -- feeds every tile subgrid in the Categories drilldown, at
    any depth. `bucket` and `parent` are mutually exclusive; neither given
    keeps the original unscoped-top-21 behavior.

    The `bucket`/unscoped branch uses a single GROUP BY left(c.code, 1), not
    _bucket_counts's sequential-EXISTS-per-bucket loop: that loop exists
    because the 6 BUCKETS can each match more than one top-level code and a
    word can straddle two different buckets, so summing independent bucket
    counts would double count a word across buckets. Here every row is
    already grouped by its own single top-level code, so one query is both
    correct and 21x cheaper. The `parent` branch can't reuse that same single
    GROUP BY shape (see _child_subtree_counts) since a code's children aren't
    always the same length."""
    if bucket is not None and parent is not None:
        raise HTTPException(400, "exactly one of bucket or parent may be given, not both")

    if parent is not None:
        if parent not in _ALL_CODES:
            raise HTTPException(404, f"unknown category code {parent!r}")
        children = usas.children_of(parent)
        with _main.get_conn() as conn, conn.cursor() as cur:
            counts = _child_subtree_counts(cur, children)
        return [
            CategoryCount(code=c["code"], name=c["name"], bucket=usas_domains.bucket_for(c["code"]),
                          word_count=counts.get(c["code"], 0))
            for c in children
        ]

    if bucket is not None and bucket not in usas_domains.DOMAIN_BUCKETS:
        raise HTTPException(404, f"unknown bucket {bucket!r}")
    codes = usas_domains.DOMAIN_BUCKETS[bucket]["codes"] if bucket else _TOP_CODES

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT left(c.code, 1) AS top, count(DISTINCT w.id)
                FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = w.id
                JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                WHERE w.active AND left(c.code, 1) = ANY(%s)
                GROUP BY left(c.code, 1)""",
            (codes,),
        )
        counts = dict(cur.fetchall())

    return [
        CategoryCount(code=code, name=_TOP_CODE_NAMES[code], bucket=usas_domains.bucket_for(code),
                      word_count=counts.get(code, 0))
        for code in codes
    ]


# --- /api/browse/category-overlap -----------------------------------------------

class CategoryOverlapCell(BaseModel):
    code_a: str
    code_b: str
    shared_words: int
    ratio: float  # shared / (size_a + size_b - shared) -- Jaccard, not a raw count:
                   # categories vary hugely in size, so a raw count would mostly just
                   # track how big the two categories are, not how related they are


class CategoryOverlap(BaseModel):
    sizes: list[CategoryCount]        # one row per sibling -- the heatmap's diagonal
    cells: list[CategoryOverlapCell]  # only pairs that actually co-occur at least once


def _overlap_matrix(pairs: list[tuple[int, str]], siblings: list[dict],
                     bucket_lookup: bool) -> CategoryOverlap:
    """Shared aggregation for every branch of browse_category_overlap below.
    `pairs` is (word_id, sibling_code) for every (word, sibling its
    categories land in) combination -- a word contributes one row per
    sibling it touches, so a word touching 2+ siblings in the SAME call is
    exactly the overlap event (a word carries at most 3 USAS categories, so
    this is never more than 3 rows per word). `bucket_lookup` is True only
    for the true top-of-drilldown case (the 6 UI buckets), where a
    sibling's OWN "bucket" field must be itself, not looked up via a
    top-level code it isn't."""
    word_siblings: dict[int, set[str]] = defaultdict(set)
    for word_id, code in pairs:
        word_siblings[word_id].add(code)

    sizes: dict[str, int] = defaultdict(int)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for codes in word_siblings.values():
        for code in codes:
            sizes[code] += 1
        if len(codes) > 1:
            for a, b in itertools.combinations(sorted(codes), 2):
                shared[(a, b)] += 1

    size_rows = [
        CategoryCount(code=s["code"], name=s["name"],
                      bucket=s["code"] if bucket_lookup else usas_domains.bucket_for(s["code"]),
                      word_count=sizes.get(s["code"], 0))
        for s in siblings
    ]
    cells = [
        CategoryOverlapCell(
            code_a=a, code_b=b, shared_words=n,
            ratio=round(n / (sizes[a] + sizes[b] - n), 4) if (sizes[a] + sizes[b] - n) else 0.0,
        )
        for (a, b), n in shared.items()
    ]
    return CategoryOverlap(sizes=size_rows, cells=cells)


@router.get("/api/browse/category-overlap", response_model=CategoryOverlap)
def browse_category_overlap(
    bucket: str | None = None,
    parent: str | None = None,
    _: dict = Depends(_main.require_viewer),
) -> CategoryOverlap:
    """How much a Categories-drilldown level's sibling categories' word sets
    overlap -- feeds the heatmap next to every tile subgrid. Three levels,
    mirroring the drilldown's own three distinct tiers:
      - neither given -> overlap among the 6 UI buckets themselves --
        CategoriesOverview's own level. Different from category-counts's
        unscoped case (the 21 top-level fields): CategoriesOverview never
        calls category-counts, it calls /api/browse/domains, so this
        endpoint's own top tier has to be the buckets, not the fields.
      - bucket given -> overlap among that bucket's member top-level fields
      - parent given -> overlap among that code's direct children

    Degrades to an empty `cells` list (never an error) when a level has
    fewer than 2 siblings to compare -- the common case at the deeper tiers
    (e.g. only 17 sub-sub-sub-fields exist at all, among 100 sub-sub-field
    parents), not a bug to guard against specially."""
    if bucket is not None and parent is not None:
        raise HTTPException(400, "at most one of bucket or parent may be given")

    with _main.get_conn() as conn, conn.cursor() as cur:
        if parent is not None:
            if parent not in _ALL_CODES:
                raise HTTPException(404, f"unknown category code {parent!r}")
            siblings = usas.children_of(parent)
            if len(siblings) < 2:
                return _overlap_matrix([], siblings, bucket_lookup=False)

            or_clauses: list[str] = []
            params: list[str] = []
            for s in siblings:
                exact, like = usas.subtree_sql(s["code"])
                or_clauses.append("(c.code = %s OR c.code LIKE %s)")
                params.extend([exact, like])
            cur.execute(
                f"""SELECT w.id, c.code FROM {_main.SCHEMA}.word w
                    JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = w.id
                    JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                    WHERE w.active AND ({" OR ".join(or_clauses)})""",
                params,
            )
            # A code's children aren't a uniform prefix expression across
            # siblings of different code lengths (same reason
            # _child_subtree_counts can't use one GROUP BY either), so the
            # owning sibling is resolved per row here rather than in SQL.
            pairs = [
                (word_id, next(s["code"] for s in siblings if usas.subtree_match(s["code"], code)))
                for word_id, code in cur.fetchall()
            ]
            return _overlap_matrix(pairs, siblings, bucket_lookup=False)

        if bucket is not None and bucket not in usas_domains.DOMAIN_BUCKETS:
            raise HTTPException(404, f"unknown bucket {bucket!r}")

        if bucket is not None:
            top_codes = usas_domains.DOMAIN_BUCKETS[bucket]["codes"]
            siblings = [{"code": c, "name": _TOP_CODE_NAMES[c]} for c in top_codes]
        else:
            top_codes = _TOP_CODES
            siblings = [{"code": key, "name": v["name"]} for key, v in usas_domains.DOMAIN_BUCKETS.items()]
        if len(siblings) < 2:
            return _overlap_matrix([], siblings, bucket_lookup=bucket is None)

        cur.execute(
            f"""SELECT w.id, left(c.code, 1) FROM {_main.SCHEMA}.word w
                JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = w.id
                JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                WHERE w.active AND left(c.code, 1) = ANY(%s)""",
            (top_codes,),
        )
        rows = cur.fetchall()
        # bucket=X: the fetched top-level code IS the sibling code directly.
        # Unscoped (the 6 buckets): a top-level code maps to its bucket.
        pairs = rows if bucket is not None else [
            (word_id, usas_domains.bucket_for(top_code)) for word_id, top_code in rows
        ]
        return _overlap_matrix(pairs, siblings, bucket_lookup=bucket is None)


# --- /api/browse/category-leaders ---------------------------------------------------

class CategoryLeaderRow(BaseModel):
    id: str            # book id (as a string) or author name -- same convention as DomainMapNode
    label: str
    subtitle: str | None  # author (book rows only); None for author rows
    total_word_count: int
    category_word_count: int
    share: float        # category_word_count / total_word_count
    lift: float          # share / mean(share) across every qualifying entity in this ranking


class CategoryLeaderPage(BaseModel):
    entity: Literal["book", "author"]
    items: list[CategoryLeaderRow]
    total: int
    page: int
    page_size: int


@router.get("/api/browse/category-leaders", response_model=CategoryLeaderPage)
def browse_category_leaders(
    entity: Literal["book", "author"] = "book",
    bucket: str | None = None,
    top_code: str | None = None,
    min_words: int = Query(0, ge=0, description="Overrides the entity's own default floor (50 for books, 100 for authors) if higher."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _: dict = Depends(_main.require_viewer),
) -> CategoryLeaderPage:
    """Ranked "who/what leans on this category's vocabulary the most" list --
    the Categories section's core drill-down mechanism, at either the
    broadest (bucket) or secondary (single top-level field) granularity.

    Deliberately NOT a deep-link into /api/browse/authors|books with their
    existing `domain` filter: passing `domain` there makes THEIR OWN
    word_count column silently mean "words in this domain" rather than total
    vocabulary, and sorting by it ranks by raw count -- which
    _domain_vectors_to_map's own docstring already found empirically useless
    as an intensity signal (a raw-share argmax landed on one field for 82%
    of books, since some fields are corpus-wide dominant everywhere). This
    endpoint applies the same fix that function does: rank by LIFT (an
    entity's own share of the category, divided by the qualifying
    population's OWN mean share for it), not raw share or raw count.

    Exactly one of `bucket`/`top_code` is required -- `bucket` resolves to
    its member USAS top-level codes (usas_domains.DOMAIN_BUCKETS); `top_code`
    is used directly, for the field/sub-field/sub-sub-field drill-down, at
    any depth -- a word tagged at `top_code` OR a real descendant of it
    counts (see usas.subtree_match). Same qualification floors as
    /api/browse/domain-map (50 words/book, 100/author): below that a
    category share is mostly sampling noise, and without a floor a book
    with 3 words all in one field would rank #1 on lift alone.
    """
    if bool(bucket) == bool(top_code):
        raise HTTPException(400, "exactly one of bucket or top_code is required")
    if bucket is not None:
        if bucket not in usas_domains.DOMAIN_BUCKETS:
            raise HTTPException(404, f"unknown bucket {bucket!r}")
        codes = usas_domains.DOMAIN_BUCKETS[bucket]["codes"]
    else:
        if top_code not in _ALL_CODES:
            raise HTTPException(404, f"unknown category code {top_code!r}")
        codes = [top_code]

    subtree_where, subtree_params = _subtree_or_sql(codes)

    default_floor = 50 if entity == "book" else 100
    floor = max(min_words, default_floor)

    with _main.get_conn() as conn, conn.cursor() as cur:
        if entity == "book":
            cur.execute(
                f"""WITH book_words AS (
                        SELECT b.id, b.title, b.author, w.id AS word_id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                    ),
                    book_totals AS (
                        SELECT id, title, author, count(DISTINCT word_id) AS total
                        FROM book_words
                        GROUP BY id, title, author
                        HAVING count(DISTINCT word_id) >= %s
                    ),
                    book_cat AS (
                        SELECT bt.id, count(DISTINCT bw.word_id) AS cat_count
                        FROM book_totals bt
                        JOIN book_words bw ON bw.id = bt.id
                        JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = bw.word_id
                        JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id AND ({subtree_where})
                        GROUP BY bt.id
                    )
                    SELECT bt.id::text, bt.title, bt.author, bt.total, coalesce(bc.cat_count, 0)
                    FROM book_totals bt
                    LEFT JOIN book_cat bc ON bc.id = bt.id""",
                (floor, *subtree_params),
            )
        else:
            cur.execute(
                f"""WITH author_words AS (
                        SELECT DISTINCT b.author, w.id AS word_id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                        WHERE b.author IS NOT NULL AND b.author != ALL(%s)
                    ),
                    author_totals AS (
                        SELECT author, count(DISTINCT word_id) AS total
                        FROM author_words
                        GROUP BY author
                        HAVING count(DISTINCT word_id) >= %s
                    ),
                    author_cat AS (
                        SELECT at.author, count(DISTINCT aw.word_id) AS cat_count
                        FROM author_totals at
                        JOIN author_words aw ON aw.author = at.author
                        JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = aw.word_id
                        JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id AND ({subtree_where})
                        GROUP BY at.author
                    )
                    SELECT at.author, at.author, NULL::text, at.total, coalesce(ac.cat_count, 0)
                    FROM author_totals at
                    LEFT JOIN author_cat ac ON ac.author = at.author""",
                (list(PLACEHOLDER_AUTHORS), floor, *subtree_params),
            )
        rows = cur.fetchall()

    entries = []
    for id_, label, subtitle, total, cat_count in rows:
        share = cat_count / total if total else 0.0
        entries.append({"id": id_, "label": label, "subtitle": subtitle,
                         "total": total, "cat_count": cat_count, "share": share})

    if not entries:
        return CategoryLeaderPage(entity=entity, items=[], total=0, page=page, page_size=page_size)

    # Same lift definition _domain_vectors_to_map uses: this entity's share
    # divided by the unweighted mean share across every OTHER qualifying
    # entity in this same ranking -- "leans on this MORE than a typical
    # qualifying book/author does," not "has a high raw share" (which a
    # corpus-wide-dominant field would win everywhere) or "has a high raw
    # count" (which a long book/prolific author would win everywhere).
    mean_share = sum(e["share"] for e in entries) / len(entries)
    mean_share = mean_share or 1e-9  # guard only -- unreachable while any entity has >0 words in these codes
    for e in entries:
        e["lift"] = e["share"] / mean_share

    entries.sort(key=lambda e: e["lift"], reverse=True)
    total_count = len(entries)
    start = (page - 1) * page_size
    page_entries = entries[start:start + page_size]

    items = [
        CategoryLeaderRow(
            id=e["id"], label=e["label"], subtitle=e["subtitle"],
            total_word_count=e["total"], category_word_count=e["cat_count"],
            share=round(e["share"], 4), lift=round(e["lift"], 3),
        )
        for e in page_entries
    ]
    return CategoryLeaderPage(entity=entity, items=items, total=total_count, page=page, page_size=page_size)


# --- /api/browse/domain-map ------------------------------------------------------

class DomainMapNode(BaseModel):
    id: str            # book id (as a string) or author name
    label: str          # book title, or author name (same as id for authors)
    subtitle: str | None  # author (for a book node); None for author nodes
    word_count: int
    x: float
    y: float
    dominant_code: str | None      # the USAS top-level field this entity's vocabulary
                                    # leans on most, e.g. "Y" -- None only if an entity
                                    # somehow has zero categorized words (not seen live)
    dominant_name: str | None      # e.g. "SCIENCE & TECHNOLOGY"
    dominant_bucket: str | None    # usas_domains.py's 6-hue compression of dominant_code,
                                    # for colorForBucket() -- same palette the word graph uses
    dominant_fraction: float | None  # e.g. 0.34 -- this category's share of the entity's
                                      # own category-incidence distribution


class DomainMapResponse(BaseModel):
    entity: Literal["book", "author"]
    nodes: list[DomainMapNode]


def _domain_vectors_to_map(rows: list[tuple], id_is_int: bool) -> list[DomainMapNode]:
    """Shared second half of browse_domain_map for both entities: rows is
    (id, label, subtitle, total_words, {code: count}) tuples. Builds each
    entity's distribution over the 21 USAS top-level fields, projects to 2D,
    and picks a dominant category for color.

    Deliberately stays 21-dimensional (level-0 only) even though the
    Categories drilldown elsewhere now goes 4 levels deep: this function's
    own empirical finding below (raw-argmax collapsing to one color 82% of
    the time) already shows 21 dims is near the edge of what a dominant-
    category color can distinguish; adding 115+100 more dimensions would
    make node coloring close to meaningless, not more precise.

    PCA, not classical MDS-from-a-distance-matrix (compute_author_clustering's
    approach in db.py): that machinery exists because ITS feature space is
    thousands of sparse word columns, forcing an n x n Gram matrix. Here the
    feature space is a fixed 21 dense columns, so building the n x 21 matrix
    directly and eigh-ing its 21 x 21 covariance is the identical embedding
    (classical MDS on Euclidean distance IS PCA on the centered data) at a
    fraction of the cost -- no scipy, no O(n^2) distance matrix, and it scales
    to the corpus directly rather than needing a top-N cap for tractability.

    Each row is L1-normalized first (so it's a genuine distribution over
    categories, matching this codebase's existing multi-label-domain
    convention -- a word can count toward more than one field, so rows don't
    represent a strict partition of the entity's words) and THEN L2-normalized
    before the PCA step, so Euclidean distance between rows corresponds to
    cosine similarity between the underlying distributions -- the same
    distance definition compute_author_clustering uses, for the same reason
    (a proper metric, not raw 1-cosine).
    """
    import numpy as np

    ids, labels, subtitles, totals, count_dicts = [], [], [], [], []
    for id_, label, subtitle, total, counts in rows:
        ids.append(id_)
        labels.append(label)
        subtitles.append(subtitle)
        totals.append(total)
        count_dicts.append(counts)

    n = len(ids)
    raw = np.array([[cd.get(code, 0) for code in _TOP_CODES] for cd in count_dicts], dtype=float)
    row_sums = raw.sum(axis=1)
    # Not seen live (a corpus-wide scan found >=98.6% category coverage even
    # at the lowest qualifying word counts) but guarded rather than trusted:
    # an entity with zero categorized words can't be L1-normalized.
    valid = row_sums > 0
    if valid.sum() < 2:
        return []

    l1 = np.zeros_like(raw)
    l1[valid] = raw[valid] / row_sums[valid, None]

    # Dominant category is picked by LIFT (this entity's share / the corpus's
    # own average share for that category), not raw share -- "A GENERAL &
    # ABSTRACT TERMS" is corpus-wide the single largest field by a wide
    # margin (general/abstract vocabulary pervades every kind of writing),
    # so a raw-argmax dominant category came back "A" for 3752/4559 books
    # (82%) in a live check -- a map where 4 in 5 dots share one color tells
    # you almost nothing. Dividing by each category's corpus-wide mean share
    # before taking the argmax surfaces what a book/author leans on MORE
    # THAN THE CORPUS TYPICALLY DOES, which is what "the discipline this
    # work belongs to" actually means -- the same relative-to-baseline
    # instinct as this file's own idf fields elsewhere (book_shared_words/
    # author_shared_words), just lift instead of log-lift. The live check
    # above confirmed it: the same 4559 books spread across all 21 fields
    # instead of clustering on 13, none dominating.
    baseline = l1[valid].mean(axis=0)
    baseline[baseline == 0] = 1e-9  # guard only -- corpus scale makes this unreachable live
    lift = l1 / baseline

    l2_norms = np.linalg.norm(l1, axis=1)
    l2_norms[l2_norms == 0] = 1.0
    unit = l1 / l2_norms[:, None]

    mean = unit[valid].mean(axis=0)
    centered = unit - mean
    cov = centered[valid].T @ centered[valid]
    eigvals, eigvecs = np.linalg.eigh(cov)
    top2 = np.argsort(eigvals)[::-1][:2]
    coords = centered @ eigvecs[:, top2]

    # Same arbitrary-eigenvector-sign guard compute_author_clustering uses --
    # eigh's sign is otherwise undetermined and can flip between runs on
    # near-identical input.
    for axis in range(coords.shape[1]):
        col = coords[:, axis]
        if col[np.argmax(np.abs(col))] < 0:
            coords[:, axis] = -col

    nodes = []
    for i in range(n):
        if not valid[i]:
            continue
        dom_idx = int(np.argmax(lift[i]))
        dom_code = _TOP_CODES[dom_idx]
        nodes.append(DomainMapNode(
            id=str(ids[i]) if id_is_int else ids[i],
            label=labels[i],
            subtitle=subtitles[i],
            word_count=totals[i],
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            dominant_code=dom_code,
            dominant_name=_TOP_CODE_NAMES.get(dom_code),
            dominant_bucket=usas_domains.bucket_for(dom_code),
            dominant_fraction=round(float(l1[i, dom_idx]), 4),
        ))
    return nodes


@router.get("/api/browse/domain-map", response_model=DomainMapResponse)
def browse_domain_map(
    entity: Literal["book", "author"] = "book",
    min_words: int = Query(0, ge=0, description="Overrides the entity's own default floor (50 for books, 100 for authors) if higher."),
    _: dict = Depends(_main.require_viewer),
) -> DomainMapResponse:
    """Relationship map (§ discipline-category visualization): every
    qualifying book/author positioned by how their vocabulary distributes
    across the 21 USAS discourse fields -- distinct from every other
    relatedness view in this file, which is all built on SHARED-WORD overlap
    (author_similarity/book_similarity). Two books here can sit close
    together with zero words in common, as long as their words lean on the
    same mix of fields (e.g. both heavy in Nature & Science).

    Live-computed on every request, not precomputed via `maintain` like
    author_cluster -- see _domain_vectors_to_map's docstring for why that's
    cheap enough here (a 21-dim dense feature space, not thousands of sparse
    word columns), and precomputing would mean new schema/DDL on the same
    apply_schema path that's already been seen blocking an app restart for
    minutes behind `maintain`'s own long-running transactions.

    Default floor is 50 qualifying vocab words for a book, 100 for an author
    -- below that a 21-way category distribution is mostly sampling noise,
    not a real profile. A corpus-wide scan found >=98.6% category coverage
    among words even at these floors, so the floor is applied to TOTAL
    vocabulary words, not just categorized ones -- there was no meaningful
    gap between the two to worry about."""
    default_floor = 50 if entity == "book" else 100
    floor = max(min_words, default_floor)

    with _main.get_conn() as conn, conn.cursor() as cur:
        if entity == "book":
            cur.execute(
                f"""WITH book_words AS (
                        SELECT b.id, b.title, b.author, w.id AS word_id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                    ),
                    book_totals AS (
                        SELECT id, title, author, count(DISTINCT word_id) AS total
                        FROM book_words
                        GROUP BY id, title, author
                        HAVING count(DISTINCT word_id) >= %s
                    )
                    SELECT bt.id, bt.title, bt.author, bt.total,
                           left(c.code, 1) AS top_code, count(DISTINCT bw.word_id) AS n
                    FROM book_totals bt
                    JOIN book_words bw ON bw.id = bt.id
                    LEFT JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = bw.word_id
                    LEFT JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                    GROUP BY bt.id, bt.title, bt.author, bt.total, left(c.code, 1)""",
                (floor,),
            )
            rows = cur.fetchall()
            entities: dict[int, dict] = {}
            for id_, title, author, total, top_code, n in rows:
                e = entities.setdefault(id_, {"label": title, "subtitle": author, "total": total, "counts": {}})
                if top_code:
                    e["counts"][top_code] = n
            tuples = [(id_, e["label"], e["subtitle"], e["total"], e["counts"]) for id_, e in entities.items()]
            nodes = _domain_vectors_to_map(tuples, id_is_int=True)
        else:
            cur.execute(
                f"""WITH author_words AS (
                        SELECT DISTINCT b.author, w.id AS word_id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                        WHERE b.author IS NOT NULL AND b.author != ALL(%s)
                    ),
                    author_totals AS (
                        SELECT author, count(DISTINCT word_id) AS total
                        FROM author_words
                        GROUP BY author
                        HAVING count(DISTINCT word_id) >= %s
                    )
                    SELECT at.author, at.total,
                           left(c.code, 1) AS top_code, count(DISTINCT aw.word_id) AS n
                    FROM author_totals at
                    JOIN author_words aw ON aw.author = at.author
                    LEFT JOIN {_main.SCHEMA}.word_category wc ON wc.word_id = aw.word_id
                    LEFT JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                    GROUP BY at.author, at.total, left(c.code, 1)""",
                (list(PLACEHOLDER_AUTHORS), floor),
            )
            rows = cur.fetchall()
            entities = {}
            for author, total, top_code, n in rows:
                e = entities.setdefault(author, {"total": total, "counts": {}})
                if top_code:
                    e["counts"][top_code] = n
            tuples = [(author, author, None, e["total"], e["counts"]) for author, e in entities.items()]
            nodes = _domain_vectors_to_map(tuples, id_is_int=False)

    return DomainMapResponse(entity=entity, nodes=nodes)
