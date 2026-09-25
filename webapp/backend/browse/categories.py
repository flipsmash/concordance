"""Category (USAS domain) counts, overlap, leaders, the domain map and the category dendrogram."""

from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from typing import Literal

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel
from concordance import usas, usas_domains
from concordance.db import PLACEHOLDER_AUTHORS
from webapp.backend import main as _main

from .common import _ALL_CODES, _TOP_CODES, _TOP_CODE_NAMES, _subtree_or_sql, router


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
    approach in db): that machinery exists because ITS feature space is
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


# --- /api/browse/categories/dendrogram -----------------------------------
#
# Every LEAF USAS category (no child categories) that has active words,
# clustered by meaning: each leaf's centroid is the mean definition
# embedding of its active words, joined by average linkage on cosine
# distance (the usual pairing for embeddings), so semantically neighbouring
# categories sit together even across USAS branches (the time fields
# cluster; food/drink/drugs cluster; measurement sits with numbers). Leaves
# arrive in dendrogram order, each carrying its most-used words (by how many
# books use them) for the page's inline word strip.
#
# ~200 leaves -> computed on request (one ~2s aggregate query + a trivial
# linkage), cached in-process for an hour per schema rather than a new
# precompute table: category membership changes only when words are
# classified/pruned, and an hour-stale strip is harmless.

_CATEGORY_TREE_WORDS = 40                 # words per leaf strip (the page shows as many as fit)
_CATEGORY_TREE_TTL = 3600
_category_tree_cache: dict[str, tuple[float, "CategoryDendrogramResponse"]] = {}


class CategoryTreeWord(BaseModel):
    id: int
    lemma: str
    book_count: int


class CategoryTreeLeaf(BaseModel):
    code: str
    name: str
    bucket: str | None           # usas_domains color bucket of its top-level field
    word_count: int
    words: list[CategoryTreeWord]  # most-used first (book count desc, then lemma)


class CategoryDendrogramNode(BaseModel):
    code: str | None = None      # set on leaves only
    size: int
    distance: float | None = None
    left: "CategoryDendrogramNode | None" = None
    right: "CategoryDendrogramNode | None" = None


CategoryDendrogramNode.model_rebuild()


class CategoryDendrogramResponse(BaseModel):
    tree: CategoryDendrogramNode | None
    leaves: list[CategoryTreeLeaf]  # dendrogram (top-to-bottom) order


def _category_dendrogram_compute(schema: str) -> CategoryDendrogramResponse:
    import numpy as np
    from scipy.cluster.hierarchy import linkage, to_tree

    leaf_cte = f"""leaf AS (SELECT c.id, c.code, c.name FROM {schema}.category c
                            WHERE NOT EXISTS (SELECT 1 FROM {schema}.category k WHERE k.parent_id = c.id))"""
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""WITH {leaf_cte}
            SELECT l.code, l.name, count(DISTINCT w.id), avg(e.definition_vector)::text
            FROM leaf l
            JOIN {schema}.word_category wc ON wc.category_id = l.id
            JOIN {schema}.word w ON w.id = wc.word_id AND w.active
            LEFT JOIN {schema}.word_embedding e ON e.word_id = w.id
            GROUP BY l.code, l.name""")
        stats = [r for r in cur.fetchall() if r[3]]      # a leaf with no embedded word can't be placed
        cur.execute(f"""WITH {leaf_cte},
            bc AS (SELECT word_id, count(*) AS n FROM {schema}.word_book GROUP BY word_id),
            ranked AS (
                SELECT l.code, w.id, w.lemma, coalesce(bc.n, 0) AS n,
                       row_number() OVER (PARTITION BY l.code
                                          ORDER BY coalesce(bc.n, 0) DESC, w.lemma_lc) AS rk
                FROM leaf l
                JOIN {schema}.word_category wc ON wc.category_id = l.id
                JOIN {schema}.word w ON w.id = wc.word_id AND w.active
                LEFT JOIN bc ON bc.word_id = w.id)
            SELECT code, id, lemma, n FROM ranked WHERE rk <= %s ORDER BY code, rk""",
                    (_CATEGORY_TREE_WORDS,))
        words: dict[str, list[CategoryTreeWord]] = {}
        for code, wid, lemma, n in cur.fetchall():
            words.setdefault(code, []).append(CategoryTreeWord(id=wid, lemma=lemma, book_count=n))
    if not stats:
        return CategoryDendrogramResponse(tree=None, leaves=[])
    leaves = [CategoryTreeLeaf(code=code, name=name, word_count=n, words=words.get(code, []),
                               bucket=usas_domains.bucket_for(code[0]))
              for code, name, n, _vec in stats]
    if len(leaves) == 1:
        return CategoryDendrogramResponse(tree=CategoryDendrogramNode(code=leaves[0].code, size=1),
                                          leaves=leaves)
    # pgvector's text form is "[x,y,...]" -- valid JSON.
    X = np.array([json.loads(vec) for *_, vec in stats])
    root = to_tree(linkage(X, method="average", metric="cosine"))

    def build(node) -> CategoryDendrogramNode:
        if node.is_leaf():
            return CategoryDendrogramNode(code=leaves[node.id].code, size=1)
        return CategoryDendrogramNode(size=node.count, distance=float(node.dist),
                                      left=build(node.left), right=build(node.right))

    ordered = [leaves[i] for i in root.pre_order()]
    return CategoryDendrogramResponse(tree=build(root), leaves=ordered)


@router.get("/api/browse/categories/dendrogram", response_model=CategoryDendrogramResponse)
def categories_dendrogram(_: dict = Depends(_main.require_viewer)) -> CategoryDendrogramResponse:
    now = time.monotonic()
    hit = _category_tree_cache.get(_main.SCHEMA)
    if hit and now - hit[0] < _CATEGORY_TREE_TTL:
        return hit[1]
    result = _category_dendrogram_compute(_main.SCHEMA)
    _category_tree_cache[_main.SCHEMA] = (now, result)
    return result
