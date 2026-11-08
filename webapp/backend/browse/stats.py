"""Facet counts and histograms: domains, difficulty bands, unique words, fame, growth."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, Query
from pydantic import BaseModel
from concordance import usas_domains
from concordance.db import PLACEHOLDER_AUTHORS
from webapp.backend import main as _main

from .common import _OVERALL_DIFFICULTY_BAND_WIDTH, _OVERALL_DIFFICULTY_UNSCORED_LABEL, _UNIQUE_WORD_BUCKETS, _build_word_filters, _unique_word_bucket_label, router


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

    # Read from book_stats / author_stats' materialized unique_word_count
    # (refreshed by `book-stats` / `author-stats`), the same numbers the
    # unique_word_bucket click-through filters on. Computing it live meant a
    # GROUP BY over all of word_book per request: 10s for authors, long enough
    # that the hub showed an empty section. A book or author with no stats row
    # (no active words at all) has 0 unique words, so it counts into "0".
    placeholders = list(PLACEHOLDER_AUTHORS)
    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "book":
            cur.execute(f"SELECT unique_word_count FROM {s}.book_stats")
            rows = cur.fetchall()
            cur.execute(f"SELECT count(*) FROM {s}.book")
        else:
            cur.execute(f"SELECT unique_word_count FROM {s}.author_stats WHERE author != ALL(%s)", (placeholders,))
            rows = cur.fetchall()
            cur.execute(f"""SELECT count(DISTINCT author) FROM {s}.book
                            WHERE author IS NOT NULL AND author != '' AND author != ALL(%s)""", (placeholders,))
        everyone = cur.fetchone()[0]
        counts = Counter(_unique_word_bucket_label(n or 0) for (n,) in rows)
        counts["0"] += max(0, everyone - len(rows))

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


# --- /api/browse/growth -------------------------------------------------------

class DailyCount(BaseModel):
    date: str   # ISO yyyy-mm-dd
    count: int


@router.get("/api/browse/growth", response_model=list[DailyCount])
def browse_growth(
    metric: Literal["books", "authors", "words"] = "books",
    _: dict = Depends(_main.require_viewer),
) -> list[DailyCount]:
    """How many books/authors/words were added per calendar day, over the
    whole corpus's history -- a real time series (unlike every other
    histogram endpoint in this file, whose x-axis is a value range, not a
    date), so the day-to-day shape here is expected to be bursty rather than
    smooth: ingestion happens in occasional multi-hundred-book batch runs,
    not a steady daily trickle, and a chart that hid that would be
    misleading rather than "cleaner." `date_trunc('day', ...)` in the
    entity's own timezone (book.created_at's, since word.first_added is
    already a bare date with no timezone to normalize).

    'authors' counts a real author's FIRST book (PLACEHOLDER_AUTHORS
    excluded, same as every other author-scoped endpoint here) -- there's no
    separate author table, so "an author was added" only ever means "their
    earliest book landed." 'words' uses word.first_added, which -- per its
    own upsert (db's sync_book_results, LEAST() on conflict) -- only ever
    moves earlier, never later, so grouping by it directly gives each word's
    true first-seen day even across re-ingests.

    Every day in [min, max] gets a row, count=0 for a quiet day rather than
    the day being omitted -- so a bar chart built from this never silently
    compresses a long gap between batch runs into what looks like back-to-
    back days."""
    s = _main.SCHEMA

    if metric == "books":
        date_expr = "b.created_at::date"
        from_clause = f"{s}.book b"
    elif metric == "words":
        date_expr = "w.first_added"
        from_clause = f"{s}.word w WHERE w.first_added IS NOT NULL"
    else:
        placeholders = list(PLACEHOLDER_AUTHORS)
        date_expr = "fb.d"
        from_clause = f"""(SELECT min(b.created_at)::date AS d
                            FROM {s}.book b
                            WHERE b.author IS NOT NULL AND b.author != ''
                              AND b.author != ALL(%s)
                            GROUP BY b.author) fb"""

    with _main.get_conn() as conn, conn.cursor() as cur:
        query = f"""
            WITH counted AS (
                SELECT {date_expr} AS d, count(*) AS n FROM {from_clause} GROUP BY 1
            ), bounds AS (
                SELECT min(d) AS lo, max(d) AS hi FROM counted
            )
            SELECT gs.d::date, coalesce(counted.n, 0)
            FROM bounds, generate_series(bounds.lo, bounds.hi, interval '1 day') AS gs(d)
            LEFT JOIN counted ON counted.d = gs.d::date
            ORDER BY gs.d
        """
        cur.execute(query, (placeholders,) if metric == "authors" else None)
        rows = cur.fetchall()

    return [DailyCount(date=d.isoformat(), count=n) for d, n in rows]


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
    # book_stats / author_stats already hold each row's overall_difficulty
    # (same percent_rank blend, refreshed by `book-stats` / `author-stats`,
    # and what the listings and their band filter read). Recomputing it here
    # was a full word_book aggregate per request -- 8.5s for authors.
    with _main.get_conn() as conn, conn.cursor() as cur:
        if scope == "book":
            cur.execute(f"SELECT overall_difficulty FROM {s}.book_stats")
        else:
            cur.execute(f"SELECT overall_difficulty FROM {s}.author_stats WHERE author != ALL(%s)",
                        (list(PLACEHOLDER_AUTHORS),))
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
