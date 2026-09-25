"""/api/browse/authors and the per-author relatedness, map, matrix and dendrogram endpoints."""

from __future__ import annotations

import math
from typing import Literal

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel
from concordance.db import PLACEHOLDER_AUTHORS
from webapp.backend import main as _main

from .books import SharedWord, SharedWordsResponse
from .common import _AUTHOR_SORT_COLUMNS, _NULLABLE_SORTS, _build_word_filters, _overall_difficulty_band_filter, _unique_word_bucket_filter, _unique_word_bucket_range, router


def _browse_authors_from_stats(
    *, author: str | None, q: str | None, fame_min: float | None, fame_max: float | None,
    fame_unscored_only: bool, unique_word_bucket: str | None, overall_difficulty_band: str | None,
    letter: str | None, random: bool, page: int, page_size: int, sort: str, dir: str,
) -> "AuthorPage":
    """browse_authors' fast path: author_stats already holds this exact
    request's answer precomputed (see that table's CREATE TABLE comment and
    concordance/db's compute_author_stats) -- reached only when no word-
    level facet is active, so every filter here is a plain column condition,
    no per-request corpus-wide aggregation. `author`/`letter`/`q` narrow
    which authors to return, never which words count towards their stats,
    so they're safe under any facet state; browse_authors itself only calls
    this function once none of the stats-invalidating facets are set."""
    filters = []
    params: list = []
    if author:
        filters.append("as_.author = %s")
        params.append(author)
    if letter:
        filters.append("lower(left(as_.author, 1)) = %s")
        params.append(letter.lower())
    if q:
        filters.append("as_.author ILIKE %s")
        params.append(f"%{q}%")
    if unique_word_bucket:
        lo, hi = _unique_word_bucket_range(unique_word_bucket)
        if hi is None:
            filters.append("as_.unique_word_count >= %s")
            params.append(lo)
        else:
            filters.append("as_.unique_word_count BETWEEN %s AND %s")
            params.extend([lo, hi])
    if overall_difficulty_band is not None:
        clause, band_params = _overall_difficulty_band_filter(overall_difficulty_band)
        filters.append(clause.replace("overall_difficulty", "as_.overall_difficulty"))
        params.extend(band_params)
    if fame_unscored_only:
        filters.append("af.fame_score IS NULL")
    else:
        if fame_min is not None:
            filters.append("af.fame_score >= %s")
            params.append(fame_min)
        if fame_max is not None:
            filters.append("af.fame_score <= %s")
            params.append(fame_max)
    where = (" WHERE " + " AND ".join(filters)) if filters else ""

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

    s = _main.SCHEMA
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*) FROM {s}.author_stats as_
                LEFT JOIN {s}.author_fame af ON af.author = as_.author{where}""",
            params,
        )
        total = cur.fetchone()[0]
        cur.execute(
            f"""SELECT as_.author, as_.book_count, as_.word_count, as_.scored_word_count,
                       as_.mean_difficulty, as_.stddev_difficulty, as_.density,
                       as_.overall_difficulty, af.fame_score, af.fame_reasoning,
                       as_.unique_word_count
                FROM {s}.author_stats as_
                LEFT JOIN {s}.author_fame af ON af.author = as_.author{where}
                ORDER BY {order_by}
                LIMIT %s OFFSET %s""",
            (*params, limit, offset),
        )
        rows = cur.fetchall()

    items = [
        AuthorRow(author=r[0], book_count=r[1], word_count=r[2], scored_word_count=r[3],
                  mean_difficulty=r[4], stddev_difficulty=r[5], density=r[6], overall_difficulty=r[7],
                  fame_score=r[8], fame_reasoning=r[9], unique_word_count=r[10])
        for r in rows
    ]
    return AuthorPage(items=items, total=total, page=page, page_size=page_size)


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
    # Fast path: author_stats (concordance/db's compute_author_stats)
    # already holds mean_difficulty/density/overall_difficulty/
    # unique_word_count precomputed for the UNFILTERED population -- but
    # only for that population, since any word-level facet here
    # (domain/difficulty_min/difficulty_max/archaic/pos/quizzable_only) or
    # book_id changes which words count towards an author's own stats (see
    # author_stats' CREATE TABLE comment). None of those active means this
    # request's answer is already sitting in that table, precomputed,
    # instead of needing the full live recomputation below.
    if not (domain or difficulty_min is not None or difficulty_max is not None
            or archaic or pos or quizzable_only or book_id):
        return _browse_authors_from_stats(
            author=author, q=q, fame_min=fame_min, fame_max=fame_max,
            fame_unscored_only=fame_unscored_only, unique_word_bucket=unique_word_bucket,
            overall_difficulty_band=overall_difficulty_band, letter=letter,
            random=random, page=page, page_size=page_size, sort=sort, dir=dir,
        )
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
    # excluded from author_similarity/clustering (db) but this listing
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
                            -- count(*), not count(DISTINCT w.id): grouped by
                            -- b.id, and word_book's PK is (word_id, book_id),
                            -- so a book's own words never repeat within this
                            -- group -- DISTINCT is a no-op here that was
                            -- forcing an extra per-group dedup over millions
                            -- of rows for nothing.
                            SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                                   count(w.id) AS book_word_count
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
                -- count(*): see the matching book_word_counts CTE above --
                -- grouped by b.id, so word_book's PK makes DISTINCT redundant.
                SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                       count(w.id) AS book_word_count
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


# --- /api/browse/authors/{author}/related, /api/browse/authors/relatedness -----
#
# Both read concordance/db's precomputed author_similarity table --
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
    (top-N by book count -- see concordance/db's compute_author_clustering),
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
