"""/api/browse/books, genres, book text, and the per-book relatedness, map, matrix and
dendrogram endpoints."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from webapp.backend import main as _main

from .common import _ARCHIVE_ROOT, _BOOK_SORT_COLUMNS, _NULLABLE_SORTS, _SORT_TITLE_EXPR, _build_word_filters, _overall_difficulty_band_filter, _unique_word_bucket_filter, _unique_word_bucket_range, router


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


@router.get("/api/browse/genres", response_model=list[str])
def browse_genres(_: dict = Depends(_main.require_viewer)) -> list[str]:
    """Genres actually present on at least one book, not the full static
    concordance.genre.GENRE_LIST -- most of that list has no data until
    `concordance book-genres` finishes its backlog, and a filter option that
    always returns zero books is worse than not offering it yet (same
    DISTINCT-over-fixed-list reasoning as quiz.py's quiz_meta genres)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT genre FROM {_main.SCHEMA}.book_genre ORDER BY 1")
        return [r[0] for r in cur.fetchall()]


class GenreCount(BaseModel):
    genre: str
    word_count: int


class GenreOverlapCell(BaseModel):
    genre_a: str
    genre_b: str
    shared_words: int
    ratio: float  # Jaccard (shared / union), same reasoning as
                   # CategoryOverlapCell.ratio -- genres vary hugely in book
                   # count, so a raw shared-word count would mostly just
                   # track genre size, not how related two genres actually are


class GenreOverlap(BaseModel):
    sizes: list[GenreCount]         # one row per genre that has any active words at all
    cells: list[GenreOverlapCell]   # only pairs that actually share at least one word


@router.get("/api/browse/genre-overlap", response_model=GenreOverlap)
def browse_genre_overlap(_: dict = Depends(_main.require_viewer)) -> GenreOverlap:
    """How much active-word vocabulary each pair of genres shares -- feeds
    GenreOverlapGraph on the Visualizations page. Unlike
    browse_category_overlap (three drilldown tiers, since USAS categories
    nest), genre is flat -- concordance/genre.py's GENRE_LIST has no
    hierarchy -- so this is always the full set of genres actually in use,
    never a bucket/parent-scoped subset.

    Genre lives on `book`, not `word` (concordance book-genres tags a whole
    book, not individual words), so a word's own genre membership is
    derived here rather than looked up directly: a word belongs to a genre
    if ANY of its books carries that genre tag. A word touching several
    genres (its own book has multiple tags, or it appears in several
    differently-tagged books) contributes to every one of them, same
    "one row per sibling it touches" shape _overlap_matrix uses for
    categories -- not factored out into a shared helper (see
    CategoryOverlapGraph.jsx's own docstring for why this project
    copy-adapts rather than forcing one abstraction over genuinely
    different join shapes: category attaches to a word directly via
    word_category, genre only reaches a word by way of word_book)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT wb.word_id, bg.genre
                FROM {_main.SCHEMA}.word_book wb
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id AND w.active
                JOIN {_main.SCHEMA}.book_genre bg ON bg.book_id = wb.book_id"""
        )
        word_genres: dict[int, set[str]] = defaultdict(set)
        for word_id, genre in cur.fetchall():
            word_genres[word_id].add(genre)

    sizes: dict[str, int] = defaultdict(int)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for genres in word_genres.values():
        for genre in genres:
            sizes[genre] += 1
        if len(genres) > 1:
            for a, b in itertools.combinations(sorted(genres), 2):
                shared[(a, b)] += 1

    size_rows = [GenreCount(genre=g, word_count=n) for g, n in sorted(sizes.items())]
    cells = [
        GenreOverlapCell(
            genre_a=a, genre_b=b, shared_words=n,
            ratio=round(n / (sizes[a] + sizes[b] - n), 4) if (sizes[a] + sizes[b] - n) else 0.0,
        )
        for (a, b), n in shared.items()
    ]
    return GenreOverlap(sizes=size_rows, cells=cells)


def _browse_books_from_stats(
    *, author: str | None, book_id: list[int], q: str | None, fame_min: float | None,
    fame_max: float | None, fame_unscored_only: bool, genre: list[str],
    unique_word_bucket: str | None, overall_difficulty_band: str | None, letter: str | None,
    random: bool, page: int, page_size: int, sort: str, dir: str,
) -> "BookPage":
    """browse_books' fast path -- see _browse_authors_from_stats for the
    shared shape and book_stats' CREATE TABLE comment for why this table
    exists. Reached only when no word-level facet is active (browse_books
    itself gates this); `author`/`book_id`/`genre`/`letter`/`q` all narrow
    which books to return, never a book's own precomputed stats, so they're
    safe filters here regardless.

    overall_difficulty here is unconditionally corpus-wide (see
    compute_book_stats' docstring) -- matching what the frontend actually
    labels it ("Harder than N% of the corpus"), not the live path's
    filtered-population percentile."""
    filters = []
    params: list = []
    if author:
        filters.append("b.author = %s")
        params.append(author)
    if book_id:
        filters.append("b.id = ANY(%s)")
        params.append(book_id)
    if genre:
        filters.append(
            f"EXISTS (SELECT 1 FROM {_main.SCHEMA}.book_genre bgf WHERE bgf.book_id = b.id AND bgf.genre = ANY(%s))"
        )
        params.append(genre)
    if q:
        filters.append("b.title ILIKE %s")
        params.append(f"%{q}%")
    if letter:
        filters.append(f"left({_SORT_TITLE_EXPR}, 1) = %s")
        params.append(letter.lower())
    if unique_word_bucket:
        lo, hi = _unique_word_bucket_range(unique_word_bucket)
        if hi is None:
            filters.append("bs.unique_word_count >= %s")
            params.append(lo)
        else:
            filters.append("bs.unique_word_count BETWEEN %s AND %s")
            params.extend([lo, hi])
    if overall_difficulty_band is not None:
        clause, band_params = _overall_difficulty_band_filter(overall_difficulty_band)
        filters.append(clause.replace("overall_difficulty", "bs.overall_difficulty"))
        params.extend(band_params)
    if fame_unscored_only:
        filters.append("bf.fame_score IS NULL")
    else:
        if fame_min is not None:
            filters.append("bf.fame_score >= %s")
            params.append(fame_min)
        if fame_max is not None:
            filters.append("bf.fame_score <= %s")
            params.append(fame_max)
    where = (" WHERE " + " AND ".join(filters)) if filters else ""

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

    s = _main.SCHEMA
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*)
                FROM {s}.book b
                JOIN {s}.book_stats bs ON bs.book_id = b.id
                LEFT JOIN {s}.book_fame bf ON bf.book_id = b.id{where}""",
            params,
        )
        total = cur.fetchone()[0]
        cur.execute(
            f"""WITH book_genres AS (
                    SELECT book_id, array_agg(genre ORDER BY genre) AS genres
                    FROM {s}.book_genre
                    GROUP BY book_id
                )
                SELECT b.id, b.title, b.author, bs.word_count, bs.scored_word_count,
                       bs.mean_difficulty, bs.stddev_difficulty, bs.density, b.archive_path,
                       bs.overall_difficulty, bf.fame_score, bf.fame_reasoning,
                       bs.unique_word_count, coalesce(bg.genres, '{{}}'),
                       {_SORT_TITLE_EXPR} AS sort_title
                FROM {s}.book b
                JOIN {s}.book_stats bs ON bs.book_id = b.id
                LEFT JOIN {s}.book_fame bf ON bf.book_id = b.id
                LEFT JOIN book_genres bg ON bg.book_id = b.id{where}
                ORDER BY {order_by}
                LIMIT %s OFFSET %s""",
            (*params, limit, offset),
        )
        rows = cur.fetchall()

    items = [
        BookRow(id=r[0], title=r[1], author=r[2], word_count=r[3], scored_word_count=r[4],
                mean_difficulty=r[5], stddev_difficulty=r[6], density=r[7], archive_path=r[8],
                overall_difficulty=r[9], fame_score=r[10], fame_reasoning=r[11],
                unique_word_count=r[12], genres=r[13])
        for r in rows
    ]
    return BookPage(items=items, total=total, page=page, page_size=page_size)


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
    genre: list[str] = Query([]),  # concordance.genre.GENRE_LIST tags -- ANY match, deep-linked
                                    # from WorkDetail.jsx's clickable genre pills
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
    # Fast path: book_stats (concordance/db's compute_book_stats) already
    # holds mean_difficulty/density/overall_difficulty/unique_word_count
    # precomputed for the UNFILTERED population -- see that table's CREATE
    # TABLE comment and browse_authors' identical fast path above. Any
    # word-level facet here changes which words count towards a book's own
    # stats, so those requests still fall back to the live computation below;
    # author/book_id/genre/letter/q only narrow which books to return, so
    # they're safe under this fast path regardless.
    if not (domain or difficulty_min is not None or difficulty_max is not None
            or archaic or pos or quizzable_only):
        return _browse_books_from_stats(
            author=author, book_id=book_id, q=q, fame_min=fame_min, fame_max=fame_max,
            fame_unscored_only=fame_unscored_only, genre=genre,
            unique_word_bucket=unique_word_bucket, overall_difficulty_band=overall_difficulty_band,
            letter=letter, random=random, page=page, page_size=page_size, sort=sort, dir=dir,
        )
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
    if genre:
        # Own alias (bgf, not book_genres' `bg`) -- this filter is a plain
        # EXISTS against `b`, evaluated in `where` before either query's
        # book_genres CTE exists, so it can't reference that CTE anyway.
        filters.append(
            f"EXISTS (SELECT 1 FROM {_main.SCHEMA}.book_genre bgf WHERE bgf.book_id = b.id AND bgf.genre = ANY(%s))"
        )
        params.append(genre)
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
                            -- count(*): grouped by b.id, and word_book's PK
                            -- is (word_id, book_id), so DISTINCT is a no-op
                            -- here -- see browse_authors' matching comment.
                            SELECT b.id, b.distinct_nonstop_word_count,
                                   avg(wd.difficulty) AS mean_difficulty,
                                   CASE WHEN b.distinct_nonstop_word_count > 0
                                        THEN count(w.id)::float / b.distinct_nonstop_word_count END AS density
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
                # GROUP BY b.id alone: bf.fame_score only needs to be
                # JOINed for `where` to reference it (fame_min/max/
                # fame_unscored_only above) -- it doesn't need to be part of
                # the grouping key for a plain distinct-book-id count, and
                # adding it forced Postgres to hash/compare bf.fame_score
                # across every one of the 4.5M pre-aggregation rows instead
                # of just b.id (measured ~1.5s vs ~0.4s against the real
                # corpus).
                f"""SELECT count(*) FROM (
                        SELECT b.id
                        FROM {_main.SCHEMA}.book b
                        JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                        JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                        LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                        LEFT JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = b.id
                        WHERE {where}
                        GROUP BY b.id
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
                -- count(*), not count(DISTINCT w.id): grouped by b.id, and
                -- word_book's PK is (word_id, book_id), so a book's own
                -- words can't repeat in this group -- DISTINCT was forcing a
                -- redundant per-group dedup over the whole corpus every
                -- request. (Not safe in browse_authors, where a word
                -- legitimately repeats across an author's several books --
                -- see this file's module docstring.)
                --
                -- book_fame/book_unique_counts/book_genres are deliberately
                -- NOT joined here, unlike an earlier version of this query --
                -- each is a 1-row-per-book LEFT JOIN, but Postgres has no way
                -- to know that from here (the functional-dependency exemption
                -- that lets b.title/b.author/etc. skip GROUP BY only applies
                -- to columns of `b` itself, once b.id is grouped). Including
                -- bf.fame_reasoning (free text) and bg.genres (an array) in
                -- GROUP BY forced a hash/compare of those values across every
                -- one of the ~4.5M pre-aggregation rows, not just b.id --
                -- measured ~2.7s in that shape vs ~0.5s joined in below,
                -- after book_base has already collapsed to one row per book.
                SELECT b.id, b.title, b.author, b.archive_path, b.distinct_nonstop_word_count,
                       count(w.id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty,
                       CASE WHEN b.distinct_nonstop_word_count > 0
                            THEN count(w.id)::float / b.distinct_nonstop_word_count END AS density,
                       {_SORT_TITLE_EXPR} AS sort_title
                FROM {_main.SCHEMA}.book b
                JOIN {_main.SCHEMA}.word_book wb ON wb.book_id = b.id
                JOIN {_main.SCHEMA}.word w ON w.id = wb.word_id
                LEFT JOIN {_main.SCHEMA}.word_difficulty wd ON wd.word_id = w.id
                WHERE {where}
                GROUP BY b.id, b.title, b.author, b.archive_path, b.distinct_nonstop_word_count
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
                       bf.fame_score, bf.fame_reasoning,
                       coalesce(buc.unique_word_count, 0) AS unique_word_count,
                       bb.sort_title, coalesce(bg.genres, '{{}}') AS genres
                FROM book_base bb
                LEFT JOIN diff_rank dr ON dr.id = bb.id
                LEFT JOIN dens_rank de ON de.id = bb.id
                LEFT JOIN {_main.SCHEMA}.book_fame bf ON bf.book_id = bb.id
                LEFT JOIN book_unique_counts buc ON buc.book_id = bb.id
                LEFT JOIN book_genres bg ON bg.book_id = bb.id
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
    `concordance book-similarity` (concordance/db's compute_book_similarity)
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
# concordance/db's precomputed book_similarity/book_cluster/
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
