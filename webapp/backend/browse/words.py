"""/api/browse/words and the part-of-speech facet values."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel
from webapp.backend import main as _main

from .common import _ALL_CODES, _WORD_SORT_COLUMNS, _build_word_filters, router


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
    all_domain: list[str] = Query([]),  # AND-intersection, one level up from all_top_code --
                                         # see _build_word_filters' own docstring; deep-linked from
                                         # CategoryOverlapGraph's top-level (6-bucket) link-click
    all_genre: list[str] = Query([]),  # AND-intersection -- see _build_word_filters' own docstring;
                                        # deep-linked from GenreOverlapGraph's link-click
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
    if uncategorized and (domain or top_code or all_top_code or all_domain):
        raise HTTPException(400, "uncategorized is mutually exclusive with domain/top_code/all_top_code/all_domain")
    if unscored_only and (difficulty_min is not None or difficulty_max is not None):
        raise HTTPException(400, "unscored_only is mutually exclusive with difficulty_min/difficulty_max")
    if q and definition_q:
        raise HTTPException(400, "q and definition_q are mutually exclusive")
    if definition_contains and (q or definition_q):
        raise HTTPException(400, "definition_contains is mutually exclusive with q/definition_q")
    filters, params = _build_word_filters(
        author, book_id, domain, difficulty_min, difficulty_max, archaic, pos, quizzable_only,
        top_code=top_code, all_top_code=all_top_code, all_domain=all_domain, all_genre=all_genre,
        uncategorized=uncategorized, unscored_only=unscored_only,
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
