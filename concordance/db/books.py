"""Books and authors: metadata, similarity, stats, fame scoring, and book merges."""

from __future__ import annotations

import json
from pathlib import Path

from .core import DEFAULT_SCHEMA, _safe_schema


def get_book_by_title(conn, title: str, schema: str = DEFAULT_SCHEMA) -> tuple[int, str | None] | None:
    """(book_id, existing archive_path) for a title, or None if no such book
    -- `concordance archive-metadata` uses this to match an archive/
    filename's parsed title (see cli.py's _parse_incoming_name) to its row."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, archive_path FROM {s}.book WHERE title = %s", (title,))
        return cur.fetchone()


def backfill_publication_era(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """Derives publication_era from publication_year (archive_metadata.year_to_era)
    for every book that has an exact year but no free-text era hedge -- a
    pure computation from data already in `book`, no network involved, so
    this is cheap enough to run unconditionally at the start of
    `archive-metadata` rather than needing its own command. Covers both
    books processed before year_to_era existed and any RDF summary that
    stated a year without phrasing a matching century hedge."""
    from ..archive_metadata import year_to_era

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, publication_year FROM {s}.book
                        WHERE publication_year IS NOT NULL AND coalesce(publication_era,'') = ''""")
        rows = cur.fetchall()
        for book_id, year in rows:
            cur.execute(f"UPDATE {s}.book SET publication_era=%s WHERE id=%s",
                        (year_to_era(year), book_id))
    conn.commit()
    return {"backfilled": len(rows)}


def update_book_archive_metadata(conn, book_id: int, *, archive_path: str, word_count: int,
                                  distinct_nonstop_word_count: int, publication_year: int | None,
                                  publication_era: str | None, schema: str = DEFAULT_SCHEMA) -> None:
    """Writes one book's concordance/archive_metadata.py-computed stats --
    see that module's own docstring for what each field means and why
    publication date is two columns, not one."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""UPDATE {s}.book SET archive_path=%s, word_count=%s,
                    distinct_nonstop_word_count=%s, publication_year=%s, publication_era=%s
                WHERE id=%s""",
            (archive_path, word_count, distinct_nonstop_word_count, publication_year, publication_era, book_id),
        )
    conn.commit()


def set_book_publication_info(cur, book_id: int, publication_year: int | None,
                               publication_era: str | None, schema: str = DEFAULT_SCHEMA) -> None:
    """Fills in publication_year/era for a book that doesn't have it yet --
    used by concordance/genre.py's backfill pass, which re-fetches the same
    Gutenberg RDF `archive-metadata` would (for books whose first pass
    predates this field, or never found a match) while it's already there
    for the genre hints. Only overwrites a currently-NULL column so this
    never clobbers a value archive-metadata already found.

    Takes an open cursor and does NOT commit -- the caller's chunk-level
    commit covers it (see classify_and_store_genres), keeping "a chunk's
    writes land together or not at all" true for this update too, not just
    the book_genre inserts."""
    s = _safe_schema(schema)
    cur.execute(
        f"""UPDATE {s}.book SET
                publication_year=coalesce(publication_year, %s),
                publication_era=coalesce(publication_era, %s)
            WHERE id=%s""",
        (publication_year, publication_era, book_id),
    )


def fill_publication_years(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0) -> dict:
    """`archive-metadata --fill-years`: publication_year for archived .txt
    books that lack one, from their own title page
    (archive_metadata.title_page_year, era-checked). Fills only NULL columns
    (set_book_publication_info), deriving an era from the year when that's
    missing too. Commits every 500 books so a long run keeps its progress."""
    from .. import archive_metadata as am
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT id, archive_path, publication_era FROM {s}.book
                        WHERE publication_year IS NULL AND archive_path ILIKE '%%.txt'
                        ORDER BY id""" + (f" LIMIT {int(limit)}" if limit else ""))
        rows = cur.fetchall()
    stats = {"checked": len(rows), "filled": 0, "missing_file": 0}
    with conn.cursor() as cur:
        for i, (book_id, path, era) in enumerate(rows, 1):
            try:
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                stats["missing_file"] += 1
                continue
            year = am.title_page_year(raw, era)
            if year:
                set_book_publication_info(cur, book_id, year, era or am.year_to_era(year), schema)
                stats["filled"] += 1
            if i % 500 == 0:
                conn.commit()
    conn.commit()
    return stats


def compute_book_similarity(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                            top_k: int = 20, min_shared_words: int = 3,
                            max_df_fraction: float = 0.5) -> dict:
    """`concordance book-similarity` / `maintain`'s book-similarity step:
    each book's top-k most vocabulary-related books, by IDF-weighted cosine
    similarity over shared ACTIVE words -- lexical usage overlap, not
    semantic similarity (that's the existing word_embedding graph's job; a
    different axis, deliberately not duplicated here).

    Always recomputes everything in scope (no only-missing gate, same
    reasoning as archaic/difficulty/quizzable): IDF weights are corpus-wide,
    so they shift whenever ANY book's word_book membership changes, not
    just the book being looked at.

    Why cosine, not raw Jaccard: an earlier bug in this same file
    (browse_books/browse_authors, see their docstrings) is exactly what
    unweighted overlap reproduces -- a word shared by nearly every book
    (the/said/table) counts the same as a shared "cangue", so common words
    would dominate every score. IDF weighting fixes that; cosine (rather
    than a weighted Jaccard) also avoids penalizing a short book for having
    a small vocabulary relative to a long one it otherwise overlaps with
    almost entirely, since cosine normalizes each book's own vector
    magnitude away.

    `max_df_fraction` (default 0.5): words appearing in more than half of
    all books are excluded from the similarity computation entirely, not
    just down-weighted. Not merely a performance shortcut (though it is
    one -- without it, a self-join for computing shared-word contributions
    is combinatorial in how many books each word appears in, and a handful
    of ubiquitous words would dominate the join's cost) -- ln(N/df) for
    such a word is already close to zero, so this is a near-lossless
    approximation of the same math, expressed as a scale-independent
    fraction rather than a fixed count so it stays correct as the corpus
    grows. `shared_word_count` (an explainability field, not used for
    ranking) only counts words that passed this same filter -- "N shared
    RARE words" is a more honest, more on-brand number to show a user here
    than a raw count dominated by function words."""
    import math
    from collections import defaultdict

    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT wb.book_id) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id WHERE w.active""")
        n_books = cur.fetchone()[0]
        if n_books < 2:
            # Closing the cursor does NOT end the connection's transaction --
            # a bare SELECT still opens one in the default isolation level,
            # and an early `return` here without an explicit commit leaves
            # `conn` sitting "idle in transaction" indefinitely, holding
            # locks that block anything needing DDL (a schema drop, another
            # connection's ALTER TABLE) until the caller happens to touch
            # this same connection again. Found live: a 2-book test schema
            # hit this path, and a completely separate connection's DROP
            # SCHEMA hung for 10+ minutes waiting on it.
            conn.commit()
            return {"books": n_books, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT wb.book_id) AS df
                        FROM {s}.word_book wb JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active GROUP BY wb.word_id""")
        max_df = max_df_fraction * n_books
        idf = {wid: math.log(n_books / df) for wid, df in cur.fetchall() if df <= max_df}

        if not idf:
            conn.commit()  # same reasoning as the n_books < 2 early return above
            return {"books": n_books, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, wb.book_id FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        WHERE w.active AND wb.word_id = ANY(%s)""", (list(idf.keys()),))
        # Both directions of the same (word, book) rows, kept as two separate
        # indexes rather than one -- books_by_word answers "who else shares
        # this word" (needed once per word while scoring a book's own
        # candidates below), words_by_book answers "what are this book's own
        # words" (needed once per book, as the outer loop's starting point).
        # Combined size is O(sum of df over qualifying words), same order as
        # word_book itself -- nowhere near the O(sum of df^2) blowup below.
        books_by_word: dict[int, list[int]] = defaultdict(list)
        words_by_book: dict[int, list[int]] = defaultdict(list)
        for wid, bid in cur.fetchall():
            books_by_word[wid].append(bid)
            words_by_book[bid].append(wid)

    norm_sq: dict[int, float] = defaultdict(float)
    for wid, books in books_by_word.items():
        w = idf[wid] ** 2
        for bid in books:
            norm_sq[bid] += w
    norm = {bid: math.sqrt(v) for bid, v in norm_sq.items()}

    book_ids = list(norm.keys())
    if limit:
        book_ids = book_ids[: int(limit)]

    stored = 0
    with conn.cursor() as cur:
        # Unqualified (limit=0, the normal/maintain case): truncate the whole
        # table, not just WHERE book_a_id = ANY(book_ids) -- a book that HAD
        # rows last run but is no longer in scope this run (word_book fully
        # emptied, an exclusion filter added, etc.) would otherwise keep its
        # stale rows forever, since it'd never again appear on either side of
        # a targeted delete. Only skip the full wipe when --limit narrows
        # this run to a deliberate subset, where nuking everyone else's rows
        # would be destructive instead of correct.
        if limit:
            cur.execute(f"DELETE FROM {s}.book_similarity WHERE book_a_id = ANY(%s)", (book_ids,))
        else:
            cur.execute(f"DELETE FROM {s}.book_similarity")
        # One book at a time, not one corpus-wide dot[a][b]/shared[a][b] pair
        # of dicts covering every book simultaneously (what used to live
        # here): at 26.5k+ books, a handful of words sitting just under
        # max_df_fraction's cutoff (still tens of thousands of books each)
        # made that structure's key count approach the full O(books^2) pair
        # space -- measured live at ~29GB RSS and climbing, dragging the
        # host into swap and taking Postgres's own I/O down with it
        # (2026-08-16). This local dot_a/shared_a pair covers only the
        # current book's candidates and is discarded every iteration, so
        # peak memory is bounded by one book's neighborhood instead of the
        # whole corpus's. Total arithmetic is NOT reduced -- a pair (a, b)
        # sharing a word gets its idf^2 contribution added once while
        # visiting a and again while visiting b, versus the old code's
        # single combined pass -- this trades runtime for a bounded,
        # predictable memory ceiling, which is the actual scarce resource
        # here.
        for i, a in enumerate(book_ids, 1):
            dot_a: dict[int, float] = defaultdict(float)
            shared_a: dict[int, int] = defaultdict(int)
            for wid in words_by_book[a]:
                w2 = idf[wid] ** 2
                for b in books_by_word[wid]:
                    if b == a:
                        continue
                    dot_a[b] += w2
                    shared_a[b] += 1
            candidates = [
                (b, dot_a[b] / (norm[a] * norm[b]), shared_a[b])
                for b in dot_a if shared_a[b] >= min_shared_words and norm.get(b, 0) > 0
            ]
            candidates.sort(key=lambda t: t[1], reverse=True)
            for b, score, shared_count in candidates[:top_k]:
                cur.execute(
                    f"""INSERT INTO {s}.book_similarity (book_a_id, book_b_id, score, shared_word_count, updated_at)
                        VALUES (%s,%s,%s,%s, now())
                        ON CONFLICT (book_a_id, book_b_id) DO UPDATE SET
                            score=EXCLUDED.score, shared_word_count=EXCLUDED.shared_word_count, updated_at=now()""",
                    (a, b, score, shared_count))
                stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"books": len(book_ids), "pairs_stored": stored}


# book.author values that are an aggregation label, not an actual author --
# an anthology's shared vocabulary owes nothing to any individual writer, so
# treating "Various"/"Unknown Author"/etc. as a real author in the
# relatedness graph produces spurious, meaningless similarity scores (a
# high-book-count phantom author that ends up "related" to nearly everyone).
# Checked against the real corpus (2026-07-23): these four cover every
# non-name book.author value.
PLACEHOLDER_AUTHORS = frozenset({"Various", "Unknown Author", "Unknown", "Anonymous"})


def compute_author_similarity(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                              top_k: int = 20, min_shared_words: int = 3,
                              max_df_fraction: float = 0.5) -> dict:
    """`concordance author-similarity` / `maintain`'s author-similarity step:
    each author's top-k most vocabulary-related authors, by IDF-weighted
    cosine similarity over shared ACTIVE words -- same metric shape as
    compute_book_similarity, one level up (an author's vector is the union
    of their books' word sets).

    Originally shipped as an on-demand, compute-per-request query in
    browse, on the plan's own reasoning that "authors are dozens today,
    full O(n^2) pairwise at request time is cheap." That premise didn't
    survive contact with the real corpus: ~3,500 authors, and a full-corpus
    timing came back at ~39s for a SINGLE request -- unusable behind an
    HTTP endpoint, let alone one a "See full relatedness graph" link would
    hit on every click. Precomputed here instead, exactly like books.

    IDF is *author*-document-frequency (ln(N_authors / df_authors)), NOT
    book-level df: a word spread across 30 books all by one author has high
    book-df (looks common) but low author-df (df=1) -- it's a distinctive
    marker of that one author, and book-df would wash out exactly the
    signal that matters at this granularity. See compute_book_similarity's
    own docstring for the cosine-over-Jaccard and max_df_fraction reasoning,
    which applies identically here.

    An author with several books containing the same word must count once
    per word, not once per book -- the DISTINCT below is load-bearing, not
    decorative: without it, an author with many books sharing a word would
    have that word's weight (and every pair involving them) inflated by
    however many of their own books happen to contain it."""
    import math
    from collections import defaultdict

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    with conn.cursor() as cur:
        cur.execute(f"""SELECT count(DISTINCT b.author) FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))""", (placeholders,))
        n_authors = cur.fetchone()[0]
        if n_authors < 2:
            conn.commit()  # see compute_book_similarity's own early-return commit note
            return {"authors": n_authors, "pairs_stored": 0}

        cur.execute(f"""SELECT wb.word_id, count(DISTINCT b.author) AS df
                        FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                        GROUP BY wb.word_id""", (placeholders,))
        max_df = max_df_fraction * n_authors
        idf = {wid: math.log(n_authors / df) for wid, df in cur.fetchall() if df <= max_df}

        if not idf:
            conn.commit()
            return {"authors": n_authors, "pairs_stored": 0}

        cur.execute(f"""SELECT DISTINCT wb.word_id, b.author FROM {s}.word_book wb
                        JOIN {s}.word w ON w.id = wb.word_id
                        JOIN {s}.book b ON b.id = wb.book_id
                        WHERE w.active AND b.author IS NOT NULL AND b.author <> ''
                          AND NOT (b.author = ANY(%s))
                          AND wb.word_id = ANY(%s)""", (placeholders, list(idf.keys())))
        # See compute_book_similarity's identical books_by_word/words_by_book
        # split for why this is two indexes, not one -- authors_by_word for
        # "who else shares this word" while scoring an author's candidates
        # below, words_by_author for "what are this author's own words" as
        # the outer loop's starting point. Same O(sum of df) size either way,
        # nowhere near the O(sum of df^2) pairwise structure removed below.
        authors_by_word: dict[int, list[str]] = defaultdict(list)
        words_by_author: dict[str, list[int]] = defaultdict(list)
        for wid, author in cur.fetchall():
            authors_by_word[wid].append(author)
            words_by_author[author].append(wid)

    norm_sq: dict[str, float] = defaultdict(float)
    for wid, authors in authors_by_word.items():
        w = idf[wid] ** 2
        for a in authors:
            norm_sq[a] += w
    norm = {a: math.sqrt(v) for a, v in norm_sq.items()}

    author_names = list(norm.keys())
    if limit:
        author_names = author_names[: int(limit)]

    stored = 0
    with conn.cursor() as cur:
        # See compute_book_similarity's identical comment: a targeted delete
        # only reaches authors still in scope THIS run, so an author who
        # drops out of scope (PLACEHOLDER_AUTHORS gaining an entry, their
        # last book losing its words, etc.) would keep stale rows forever.
        # Found live: adding "Anonymous" et al. to PLACEHOLDER_AUTHORS left
        # their old author_similarity rows behind under the targeted delete.
        if limit:
            cur.execute(f"DELETE FROM {s}.author_similarity WHERE author_a = ANY(%s)", (author_names,))
        else:
            cur.execute(f"DELETE FROM {s}.author_similarity")
        # One author at a time, not one corpus-wide dot[a][b]/shared[a][b]
        # pair of dicts covering every author simultaneously -- see
        # compute_book_similarity's identical fix (2026-08-16/17) for why:
        # same O(books^2)-shaped blowup, just against ~3,500 authors today
        # instead of 26.5k+ books, so it hadn't bitten yet. Same tradeoff
        # applies: peak memory bounded by one author's neighborhood instead
        # of the whole corpus, at the cost of recomputing each shared pair's
        # contribution once per side instead of once combined.
        for i, a in enumerate(author_names, 1):
            dot_a: dict[str, float] = defaultdict(float)
            shared_a: dict[str, int] = defaultdict(int)
            for wid in words_by_author[a]:
                w2 = idf[wid] ** 2
                for b in authors_by_word[wid]:
                    if b == a:
                        continue
                    dot_a[b] += w2
                    shared_a[b] += 1
            candidates = [
                (b, dot_a[b] / (norm[a] * norm[b]), shared_a[b])
                for b in dot_a if shared_a[b] >= min_shared_words and norm.get(b, 0) > 0
            ]
            candidates.sort(key=lambda t: t[1], reverse=True)
            for b, score, shared_count in candidates[:top_k]:
                cur.execute(
                    f"""INSERT INTO {s}.author_similarity (author_a, author_b, score, shared_word_count, updated_at)
                        VALUES (%s,%s,%s,%s, now())
                        ON CONFLICT (author_a, author_b) DO UPDATE SET
                            score=EXCLUDED.score, shared_word_count=EXCLUDED.shared_word_count, updated_at=now()""",
                    (a, b, score, shared_count))
                stored += 1
            if i % 200 == 0:
                conn.commit()
    conn.commit()
    return {"authors": len(author_names), "pairs_stored": stored}


# --- fame scoring -------------------------------------------------------------

# If at least this many items have been attempted AND the running
# no-usable-evidence rate crosses this fraction, stop the whole run rather
# than grind through the rest effectively blind -- same "stop, don't
# silently degrade" instinct as mw_backfill's quota-stop, gated on evidence
# quality instead of an API cap. The minimum-sample guard keeps a handful of
# unlucky early misses from tripping this by chance.
_FAME_EVIDENCE_FAILURE_MIN_SAMPLE = 20
_FAME_EVIDENCE_FAILURE_THRESHOLD = 0.30


def _no_usable_author_evidence(factors: dict) -> bool:
    ngram = factors.get("ngram") or {}
    wikidata = factors.get("wikidata") or {}
    ngram_ok = not ngram.get("failed") and not ngram.get("skipped")
    wikidata_ok = bool(wikidata.get("sitelinks")) and wikidata.get("corroborated")
    snippets_ok = not factors.get("snippets_failed")
    return not (ngram_ok or wikidata_ok or snippets_ok)


def _no_usable_book_evidence(factors: dict) -> bool:
    ngram = factors.get("ngram") or {}
    ngram_ok = not ngram.get("failed") and not ngram.get("skipped")
    snippets_ok = not factors.get("snippets_failed")
    return not (ngram_ok or snippets_ok)


def _load_fame_llm():
    from pathlib import Path

    from ..config import Config
    cfg = Config()
    if not (cfg.model_path and Path(cfg.model_path).exists()):
        raise RuntimeError(
            f"no local model available (model_path {cfg.model_path!r} missing) -- "
            "fame scoring needs a real LLM; pass dry_run=True to only gather evidence")
    from llama_cpp import Llama
    return Llama(model_path=cfg.model_path, n_gpu_layers=cfg.n_gpu_layers, n_ctx=cfg.n_ctx, verbose=False)


def compute_author_stats(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """`concordance author-stats`: refreshes author_stats, a precomputed
    stand-in for the aggregate columns webapp/backend/browse's
    browse_authors used to compute live on every request (mean_difficulty,
    stddev_difficulty, density, unique_word_count, overall_difficulty) --
    see author_stats' own CREATE TABLE comment for the "didn't survive
    contact with the real corpus" backstory.

    Deliberately the UNFILTERED population only -- exactly
    _build_word_filters(None, [], [], None, None, [], [], False)'s
    ["w.active"], no book_id -- matching browse_authors' own "no word-level
    facet active" fast path (see that function's use of this table). A
    faceted browse request (domain/difficulty/archaic/pos/quizzable_only/
    book_id) changes which words count towards an author's stats, so it
    still falls back to browse_authors' original live computation; this
    table only ever serves the common, unfiltered "just browsing" case.

    Every formula here is a direct copy of browse_authors' own long-
    documented reasoning (see that function's field comments for the WHY):
    density is the MEAN OF PER-BOOK densities, not deduped word_count over
    summed distinct_nonstop_word_count; unique_word_count uses the same
    single-author word_book-exclusivity CTE; overall_difficulty is a
    percent_rank(mean_difficulty)/percent_rank(density) blend, each computed
    only over authors that HAVE the underlying metric. fame_score/
    fame_reasoning are deliberately NOT stored here -- they live on their
    own independently-refreshed author_fame table, joined in by callers.

    Always a full TRUNCATE + repopulate, like compute_author_cluster* below
    -- author_stats is 100% derived from word/word_book/word_difficulty/
    book, so there's no meaningful "since last run" delta to compute
    incrementally, and a stale row for an author who lost their last book
    (or gained PLACEHOLDER_AUTHORS membership) would otherwise linger
    forever under any targeted-delete approach (see
    compute_author_similarity's own comment on exactly that failure mode)."""
    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.author_stats")
        cur.execute(
            f"""
            WITH single_author_words AS (
                SELECT wb2.word_id, min(b2.author) AS author
                FROM {s}.word_book wb2
                JOIN {s}.book b2 ON b2.id = wb2.book_id
                    AND b2.author IS NOT NULL AND b2.author != '' AND b2.author != ALL(%s)
                JOIN {s}.word w2 ON w2.id = wb2.word_id AND w2.active
                GROUP BY wb2.word_id HAVING count(DISTINCT b2.author) = 1
            ),
            author_unique_counts AS (
                SELECT author, count(*) AS unique_word_count
                FROM single_author_words
                GROUP BY author
            ),
            book_word_counts AS (
                -- count(*): grouped by b.id, and word_book's PK is
                -- (word_id, book_id), so a book's own words can't repeat in
                -- this group -- see browse's matching comment.
                SELECT b.author, b.id AS book_id, b.distinct_nonstop_word_count,
                       count(w.id) AS book_word_count
                FROM {s}.book b
                JOIN {s}.word_book wb ON wb.book_id = b.id
                JOIN {s}.word w ON w.id = wb.word_id AND w.active
                WHERE b.author IS NOT NULL AND b.author != '' AND b.author != ALL(%s)
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
                FROM {s}.book b
                JOIN {s}.word_book wb ON wb.book_id = b.id
                JOIN {s}.word w ON w.id = wb.word_id AND w.active
                WHERE b.author IS NOT NULL AND b.author != '' AND b.author != ALL(%s)
            ),
            author_word_stats AS (
                SELECT awi.author, count(DISTINCT awi.word_id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty
                FROM author_word_ids awi
                LEFT JOIN {s}.word_difficulty wd ON wd.word_id = awi.word_id
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
            )
            INSERT INTO {s}.author_stats
                (author, book_count, word_count, scored_word_count, mean_difficulty,
                 stddev_difficulty, density, unique_word_count, overall_difficulty)
            SELECT ab2.author, ab2.book_count, ab2.word_count, ab2.scored_word_count,
                   ab2.mean_difficulty, ab2.stddev_difficulty, ab2.density, ab2.unique_word_count,
                   CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                        THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                   END
            FROM author_base ab2
            LEFT JOIN diff_rank dr ON dr.author = ab2.author
            LEFT JOIN dens_rank de ON de.author = ab2.author
            """,
            (placeholders, placeholders, placeholders),
        )
        n = cur.rowcount
    conn.commit()
    return {"authors": n}


def compute_author_fame(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                        stale_days: int = 0, llm=None, dry_run: bool = False) -> dict:
    """`concordance author-fame`: an ABSOLUTE (not corpus-relative) 1-10
    historical/cultural importance score per author, LLM-judged against a
    fixed external rubric (see concordance/fame.py's module docstring for
    why absolute over corpus-relative percentile, and for the evidence
    sources). Excludes PLACEHOLDER_AUTHORS -- same reasoning as
    compute_author_similarity: an aggregation label has no individual fame
    to score.

    `checked_at` is a STICKY marker (bumped on every attempt, hit or miss)
    so a rerun with stale_days=0 only touches never-scored authors -- this
    is a genuinely expensive job (several network round-trips + one real
    LLM generation per author, realistically 5-15s each), not a quick
    backfill, so resumability is load-bearing. `dry_run=True` only gathers
    and prints evidence (no LLM call, no DB write, checked_at untouched) --
    for sanity-checking evidence quality before committing to a real run.

    Commits after EVERY author, never batched -- a long-held transaction
    here would block a webapp restart's schema-check ALTER TABLE (this
    already happened once in production). Stops the whole run early if the
    running no-usable-evidence rate crosses
    _FAME_EVIDENCE_FAILURE_THRESHOLD, rather than grinding through
    thousands of effectively-blind LLM guesses."""
    from .. import fame
    from ..dictionary import make_session

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    stale_clause = (
        "(af.checked_at IS NULL OR af.checked_at < now() - (%s * interval '1 day'))"
        if stale_days else "af.checked_at IS NULL"
    )
    extra_params = [stale_days] if stale_days else []
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT b.author FROM {s}.book b
                LEFT JOIN {s}.author_fame af ON af.author = b.author
                WHERE b.author IS NOT NULL AND b.author <> ''
                  AND NOT (b.author = ANY(%s))
                  AND {stale_clause}
                ORDER BY b.author""" + (f" LIMIT {int(limit)}" if limit else ""),
            (placeholders, *extra_params))
        authors = [r[0] for r in cur.fetchall()]

    stats = {"attempted": 0, "scored": 0, "failed_evidence": 0, "failed_parse": 0, "errors": 0,
             "stopped_early": False, "remaining": 0}
    if not authors:
        return stats

    if not dry_run and llm is None:
        llm = _load_fame_llm()

    session = make_session()
    no_evidence_count = 0

    with conn.cursor() as cur:
        for i, author in enumerate(authors, 1):
            try:
                factors = fame.gather_author_evidence(author, session)
                stats["attempted"] += 1
                if _no_usable_author_evidence(factors):
                    no_evidence_count += 1
                    stats["failed_evidence"] += 1

                if dry_run:
                    print(f"[dry-run] {author}: {json.dumps(factors, default=str)[:300]}")
                else:
                    score, why = fame.score_author(llm, author, factors)
                    if score is None:
                        stats["failed_parse"] += 1
                    else:
                        stats["scored"] += 1
                    cur.execute(
                        f"""INSERT INTO {s}.author_fame
                                (author, fame_score, fame_reasoning, fame_factors, computed_at, checked_at)
                            VALUES (%s,%s,%s,%s, CASE WHEN %s THEN now() ELSE NULL END, now())
                            ON CONFLICT (author) DO UPDATE SET
                                fame_score=EXCLUDED.fame_score, fame_reasoning=EXCLUDED.fame_reasoning,
                                fame_factors=EXCLUDED.fame_factors, computed_at=EXCLUDED.computed_at,
                                checked_at=now()""",
                        (author, score, why, json.dumps(factors, default=str), score is not None))
                    conn.commit()
            except Exception as exc:  # noqa: BLE001 -- one poisoned item must not kill a multi-hour run
                conn.rollback()
                stats["errors"] += 1
                print(f"  [author-fame] {author!r} raised {exc!r} -- skipped, left unattempted for a future run")
                continue

            if i % 25 == 0:
                print(f"  ...{i}/{len(authors)} authors attempted "
                      f"({stats['scored']} scored, {stats['failed_evidence']} no usable evidence)")

            if (stats["attempted"] >= _FAME_EVIDENCE_FAILURE_MIN_SAMPLE
                    and no_evidence_count / stats["attempted"] > _FAME_EVIDENCE_FAILURE_THRESHOLD):
                stats["stopped_early"] = True
                stats["remaining"] = len(authors) - i
                break
    return stats


def compute_book_stats(conn, schema: str = DEFAULT_SCHEMA) -> dict:
    """`concordance book-stats`: refreshes book_stats, one level down from
    compute_author_stats -- see that function's docstring for the shared
    backstory, and book_stats' own CREATE TABLE comment for why this one
    specifically was needed (browse_books' unique_word_count, not word_count
    itself, which was already cheap).

    Deliberately the UNFILTERED population, exactly like compute_author_stats
    -- a faceted browse_books request (domain/difficulty_min/difficulty_max/
    archaic/pos/quizzable_only) changes which words count towards a book's
    stats, so it still falls back to browse_books' live computation.
    `author`/`book_id`/`genre`/`letter`/`q` are all safe as post-filters on
    this table's rows, though -- none of them change any book's OWN stats,
    only which books to return (mirrors browse_authors' identical reasoning
    for `author`/`letter`/`q` there).

    overall_difficulty here is unconditionally corpus-wide -- matching
    Books.jsx/Authors.jsx's own "Harder than N% of the corpus" label and
    Visualizations.jsx's "a corpus-relative percentile" description, not the
    live browse_books path's actual behavior (which computes the percentile
    over whatever `{where}`-filtered population happens to be in scope --
    e.g. narrowed to just one author's own books when `author=X` is set).
    That live-path behavior quietly contradicts the UI's own documented
    contract whenever such a filter is active; this table's numbers are the
    ones actually described to the user.

    Every formula is a direct copy of browse_books' own live-computation
    reasoning (see BookRow's field comments for the WHY): density is
    word_count over distinct_nonstop_word_count (unlike author_stats, no
    per-book averaging needed -- there's only the one book), unique_word_count
    uses the same single-book word_book-exclusivity CTE, overall_difficulty
    is the same percent_rank(mean_difficulty)/percent_rank(density) blend.

    Always a full TRUNCATE + repopulate -- same reasoning as
    compute_author_stats: 100% derived from word/word_book/word_difficulty/
    book, no incremental delta worth tracking, and a targeted delete would
    leave a stale row behind for any book that lost its last active word."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {s}.book_stats")
        cur.execute(
            f"""
            WITH single_book_words AS (
                SELECT wb2.word_id, min(wb2.book_id) AS book_id
                FROM {s}.word_book wb2
                JOIN {s}.word w2 ON w2.id = wb2.word_id AND w2.active
                GROUP BY wb2.word_id HAVING count(*) = 1
            ),
            book_unique_counts AS (
                SELECT book_id, count(*) AS unique_word_count
                FROM single_book_words
                GROUP BY book_id
            ),
            book_base AS (
                -- count(*), not count(DISTINCT w.id): grouped by b.id, and
                -- word_book's PK is (word_id, book_id), so a book's own
                -- words can't repeat here -- see webapp/backend/browse's
                -- matching comment.
                SELECT b.id, b.distinct_nonstop_word_count,
                       count(w.id) AS word_count,
                       count(wd.difficulty) AS scored_word_count,
                       avg(wd.difficulty) AS mean_difficulty,
                       stddev_samp(wd.difficulty) AS stddev_difficulty,
                       CASE WHEN b.distinct_nonstop_word_count > 0
                            THEN count(w.id)::float / b.distinct_nonstop_word_count END AS density
                FROM {s}.book b
                JOIN {s}.word_book wb ON wb.book_id = b.id
                JOIN {s}.word w ON w.id = wb.word_id
                LEFT JOIN {s}.word_difficulty wd ON wd.word_id = w.id
                WHERE w.active
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
            INSERT INTO {s}.book_stats
                (book_id, word_count, scored_word_count, mean_difficulty,
                 stddev_difficulty, density, unique_word_count, overall_difficulty)
            SELECT bb.id, bb.word_count, bb.scored_word_count, bb.mean_difficulty,
                   bb.stddev_difficulty, bb.density, coalesce(buc.unique_word_count, 0),
                   CASE WHEN dr.diff_pct IS NOT NULL AND de.dens_pct IS NOT NULL
                        THEN round((((dr.diff_pct + de.dens_pct) / 2) * 100)::numeric, 1)
                   END
            FROM book_base bb
            LEFT JOIN diff_rank dr ON dr.id = bb.id
            LEFT JOIN dens_rank de ON de.id = bb.id
            LEFT JOIN book_unique_counts buc ON buc.book_id = bb.id
            """
        )
        n = cur.rowcount
    conn.commit()
    return {"books": n}


def compute_book_fame(conn, schema: str = DEFAULT_SCHEMA, *, limit: int = 0,
                      stale_days: int = 0, llm=None, dry_run: bool = False) -> dict:
    """`concordance book-fame`: same shape one level down from
    compute_author_fame, scoring the SPECIFIC WORK rather than its author.
    LEFT JOINs author_fame on book.author (NULL-tolerant by design -- a
    first run, or a book by a not-yet-scored or placeholder author, simply
    has no prior; concordance/fame.py's BOOK_RUBRIC already tells the model
    to treat that as "no prior available", not as evidence of obscurity).
    The exact author-fame snapshot shown (score + computed_at, or None) is
    recorded in fame_factors.author_fame_seen so a later author-fame rerun
    never makes an existing book's reasoning unverifiable against what it
    actually saw.

    Run author-fame first for the best results, but this does not require
    it. Ordered by word_count DESC so an interrupted multi-day run banks
    the highest-traffic books first (see book.word_count; NULLS LAST for
    any book that hasn't been through classify.py's counting pass)."""
    from .. import fame
    from ..dictionary import make_session

    s = _safe_schema(schema)
    placeholders = list(PLACEHOLDER_AUTHORS)
    stale_clause = (
        "(bf.checked_at IS NULL OR bf.checked_at < now() - (%s * interval '1 day'))"
        if stale_days else "bf.checked_at IS NULL"
    )
    extra_params = [stale_days] if stale_days else []
    with conn.cursor() as cur:
        cur.execute(
            f"""SELECT b.id, b.title, b.author, af.fame_score, af.fame_reasoning, af.computed_at
                FROM {s}.book b
                LEFT JOIN {s}.book_fame bf ON bf.book_id = b.id
                LEFT JOIN {s}.author_fame af
                    ON af.author = b.author AND NOT (b.author = ANY(%s))
                WHERE {stale_clause}
                ORDER BY b.word_count DESC NULLS LAST, b.title""" +
            (f" LIMIT {int(limit)}" if limit else ""),
            (placeholders, *extra_params))
        rows = cur.fetchall()

    stats = {"attempted": 0, "scored": 0, "failed_evidence": 0, "failed_parse": 0, "errors": 0,
             "stopped_early": False, "remaining": 0}
    if not rows:
        return stats

    if not dry_run and llm is None:
        llm = _load_fame_llm()

    session = make_session()
    no_evidence_count = 0

    with conn.cursor() as cur:
        for i, (book_id, title, author, a_score, a_reasoning, a_computed_at) in enumerate(rows, 1):
            try:
                author_fame = (
                    {"fame_score": a_score, "fame_reasoning": a_reasoning, "computed_at": a_computed_at}
                    if a_score is not None else None
                )
                factors = fame.gather_book_evidence(title, author or "", author_fame, session)
                stats["attempted"] += 1
                if _no_usable_book_evidence(factors):
                    no_evidence_count += 1
                    stats["failed_evidence"] += 1

                if dry_run:
                    print(f"[dry-run] {title!r}: {json.dumps(factors, default=str)[:300]}")
                else:
                    score, why = fame.score_book(llm, title, author or "", factors)
                    if score is None:
                        stats["failed_parse"] += 1
                    else:
                        stats["scored"] += 1
                    cur.execute(
                        f"""INSERT INTO {s}.book_fame
                                (book_id, fame_score, fame_reasoning, fame_factors, computed_at, checked_at)
                            VALUES (%s,%s,%s,%s, CASE WHEN %s THEN now() ELSE NULL END, now())
                            ON CONFLICT (book_id) DO UPDATE SET
                                fame_score=EXCLUDED.fame_score, fame_reasoning=EXCLUDED.fame_reasoning,
                                fame_factors=EXCLUDED.fame_factors, computed_at=EXCLUDED.computed_at,
                                checked_at=now()""",
                        (book_id, score, why, json.dumps(factors, default=str), score is not None))
                    conn.commit()
            except Exception as exc:  # noqa: BLE001 -- one poisoned item must not kill a multi-day run
                conn.rollback()
                stats["errors"] += 1
                print(f"  [book-fame] {title!r} raised {exc!r} -- skipped, left unattempted for a future run")
                continue

            if i % 25 == 0:
                print(f"  ...{i}/{len(rows)} books attempted "
                      f"({stats['scored']} scored, {stats['failed_evidence']} no usable evidence)")

            if (stats["attempted"] >= _FAME_EVIDENCE_FAILURE_MIN_SAMPLE
                    and no_evidence_count / stats["attempted"] > _FAME_EVIDENCE_FAILURE_THRESHOLD):
                stats["stopped_early"] = True
                stats["remaining"] = len(rows) - i
                break
    return stats


# --- book-merge: multi-part-book detection manifest + DB fold-together ------

def upsert_book_merge_group(conn, schema: str, *, title_base: str, author: str,
                            part_book_ids: list[int], part_labels: list[dict],
                            survivor_book_id: int | None, skip_reason: str | None,
                            gap_detail: list[int] | None) -> int:
    """Record (or refresh) one detected group -- see book_merge_group's own
    schema comment for why checked_at/skip_reason/part_book_ids are
    unconditionally overwritten on every call (re-detection is cheap and a
    group's eligibility can change as the corpus changes) while
    compiled_at/merged_at are preserved via COALESCE (they're the only
    terminal, "don't redo this" markers)."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.book_merge_group
                    (title_base, author, part_count, part_book_ids, part_labels,
                     survivor_book_id, skip_reason, gap_detail, checked_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (title_base, author) DO UPDATE SET
                    part_count=EXCLUDED.part_count, part_book_ids=EXCLUDED.part_book_ids,
                    part_labels=EXCLUDED.part_labels, survivor_book_id=EXCLUDED.survivor_book_id,
                    skip_reason=EXCLUDED.skip_reason, gap_detail=EXCLUDED.gap_detail,
                    checked_at=now()
                RETURNING id""",
            (title_base, author, len(part_book_ids), part_book_ids, json.dumps(part_labels),
             survivor_book_id, skip_reason,
             json.dumps(gap_detail) if gap_detail else None))
        return cur.fetchone()[0]


def mark_book_merge_compiled(conn, schema: str, group_id: int, compiled_path: str) -> None:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {s}.book_merge_group SET compiled_path=%s, compiled_at=now() WHERE id=%s",
                    (compiled_path, group_id))
    conn.commit()


def mark_book_merge_merged(conn, schema: str, group_id: int) -> None:
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {s}.book_merge_group SET merged_at=now() WHERE id=%s", (group_id,))
    conn.commit()


def merge_book_group(conn, schema: str, survivor_book_id: int, other_book_ids: list[int], *,
                     title: str, author: str, archive_path: str,
                     word_count: int, distinct_nonstop_word_count: int) -> dict:
    """Folds other_book_ids into survivor_book_id -- one transaction (unlike
    mw_backfill/compute_book_fame's per-item commit: a group can't safely
    be left half-merged). Tolerates other_book_ids that no longer exist at
    all (a rerun after a successful merge is a no-op, not an error -- every
    statement below is already a no-op on an empty match).

    word_book/rejected_word are REPOINTED (with dedup against their own
    unique constraints, never violating them), not deleted -- the point of
    a merge is that this vocabulary still belongs to the compiled book.
    book_similarity/book_cluster/book_fame are DELETED for the whole group
    (survivor included): these are precomputed, periodically-wholesale-
    regenerated derived data with no meaning for a differently-sized
    compiled work, so the right move is deleting and letting the next
    book-similarity/book-clustering/book-fame run rescope the compiled
    whole from scratch rather than hand-merging stale per-volume numbers.

    Explicitly accepted scope boundary: word_book keeps reflecting whatever
    each part's own extraction pipeline already found -- this does NOT
    re-run extraction over the newly compiled text. publication_year/
    publication_era are left untouched on the survivor: they're a
    Gutenberg-catalog lookup keyed on ONE volume's own eBook id, with no
    principled single answer for which volume's info the compiled work
    should inherit."""
    s = _safe_schema(schema)
    all_ids = [survivor_book_id, *other_book_ids]
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {s}.word_book (word_id, book_id)
                SELECT word_id, %s FROM {s}.word_book WHERE book_id = ANY(%s)
                ON CONFLICT DO NOTHING""",
            (survivor_book_id, other_book_ids))

        cur.execute(
            f"""INSERT INTO {s}.rejected_word
                    (book_id, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter)
                SELECT DISTINCT ON (lemma_lc) %s, lemma, reason, detail, count, zipf, pos, as_seen, sentence, chapter
                FROM {s}.rejected_word WHERE book_id = ANY(%s)
                ORDER BY lemma_lc, book_id
                ON CONFLICT (book_id, lemma_lc) DO NOTHING""",
            (survivor_book_id, other_book_ids))

        cur.execute(f"DELETE FROM {s}.book_similarity WHERE book_a_id = ANY(%s) OR book_b_id = ANY(%s)",
                    (all_ids, all_ids))
        cur.execute(f"DELETE FROM {s}.book_cluster WHERE book_id = ANY(%s)", (all_ids,))
        cur.execute(f"DELETE FROM {s}.book_fame WHERE book_id = ANY(%s)", (all_ids,))

        cur.execute(f"DELETE FROM {s}.book WHERE id = ANY(%s)", (other_book_ids,))

        cur.execute(
            f"""UPDATE {s}.book SET title=%s, author=%s, archive_path=%s,
                    word_count=%s, distinct_nonstop_word_count=%s
                WHERE id=%s""",
            (title, author, archive_path, word_count, distinct_nonstop_word_count, survivor_book_id))
    conn.commit()
    return {"survivor_book_id": survivor_book_id, "merged_count": len(other_book_ids)}
