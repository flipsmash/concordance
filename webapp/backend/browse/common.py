"""Shared pieces of the browse API: the router, sort-column maps, and the word-filter builder
(see the package docstring for the EXISTS-vs-JOIN dedup rule it follows)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
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
_ARCHIVE_ROOT = Path(__file__).resolve().parents[3] / "archive"

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
    all_domain: list[str] = [],
    all_genre: list[str] = [],
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

    `all_domain` is the identical AND-intersection fix one level up: `domain`
    with multiple bucket values is an OR (matches _build_word_filters' own
    long-standing behavior, correct for the facet-row bucket chips a user
    toggles on to broaden a search), but the category-overlap graph's
    TOP-LEVEL edge click (the 6 buckets themselves, not their member
    fields) needs the same intersection every deeper level already gets
    from `all_top_code` -- "a word tagged under a category in BOTH bucket A
    and bucket B", matching browse_category_overlap's own shared-word count
    at that tier. Found live (2026-08-17): the click-through was using
    `domain` (OR) here, so two buckets sharing, say, 500 words by the
    overlap graph's own count instead showed the ~40,000-word union of
    both buckets' entire membership -- technically not wrong data, just not
    what the edge the user clicked claimed to represent.

    `all_genre` is genre's own AND-intersection filter, same shape as
    `all_top_code` and for the same reason: browse_genres_overlap's
    click-through needs "words appearing in a book tagged BOTH genre A and
    genre B" (an intersection), not browse_books's own `genre` param (an
    OR -- "in a book tagged ANY of these"). Genre is book-level, not
    word-level like a USAS category, so each clause joins through
    word_book/book_genre rather than word_category.

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
    for bucket in all_domain:
        codes = usas_domains.DOMAIN_BUCKETS.get(bucket, {}).get("codes", [])
        if not codes:
            continue
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_category wc
                        JOIN {_main.SCHEMA}.category c ON c.id = wc.category_id
                        WHERE wc.word_id = w.id AND left(c.code, 1) = ANY(%s))"""
        )
        params.append(codes)
    for genre in all_genre:
        filters.append(
            f"""EXISTS (SELECT 1 FROM {_main.SCHEMA}.word_book wb
                        JOIN {_main.SCHEMA}.book_genre bg ON bg.book_id = wb.book_id
                        WHERE wb.word_id = w.id AND bg.genre = %s)"""
        )
        params.append(genre)
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


def _unique_word_bucket_range(bucket_label: str) -> tuple[int, int | None]:
    """(lo, hi) for a bucket label out of _UNIQUE_WORD_BUCKETS -- the plain
    range check _browse_authors_from_stats needs against author_stats'
    already-materialized unique_word_count column, as opposed to
    _unique_word_bucket_filter's NOT EXISTS/HAVING machinery above (built
    for a live per-book/per-author computation that doesn't exist in this
    fast path -- the count is just a column here)."""
    for lo, hi, label in _UNIQUE_WORD_BUCKETS:
        if label == bucket_label:
            return lo, hi
    raise HTTPException(404, f"unknown unique-word bucket {bucket_label!r}")
