"""API for the vocab review-and-prune UI (first slice of the larger web app).

Serves the active word list for review and lets the user soft-delete ("prune")
terms that turn out to be too common/easy. Pruned words are never hard-deleted —
`word.active` flips to false so audio/ngram/etc. history stays intact and any
later feature (quizzing, stats) just needs to filter on active=true.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from concordance import db as cdb
from concordance import localdict
from concordance.dictionary import enrich as dictionary_enrich
from concordance.model import Candidate, normalize_pos

app = FastAPI(title="Concordance Review API")

SCHEMA = cdb.DEFAULT_SCHEMA
SORT_COLUMNS = {
    "lemma": "w.lemma",
    "part_of_speech": "w.part_of_speech",
    "definition": "w.definition",
    "difficulty": "d.difficulty",
}


@contextmanager
def get_conn():
    conn = cdb.connect()
    try:
        yield conn
    finally:
        conn.close()


@app.on_event("startup")
def on_startup() -> None:
    with get_conn() as conn:
        cdb.apply_schema(conn, SCHEMA)


class WordRow(BaseModel):
    id: int
    lemma: str
    part_of_speech: str | None
    definition: str | None
    difficulty: float | None
    rescued_from_reject: bool


class WordPage(BaseModel):
    items: list[WordRow]
    total: int
    page: int
    page_size: int


@app.get("/api/words", response_model=WordPage)
def list_words(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort: Literal["lemma", "part_of_speech", "definition", "difficulty"] = "difficulty",
    dir: Literal["asc", "desc"] = "asc",
    pos: str | None = None,
) -> WordPage:
    order_col = SORT_COLUMNS[sort]
    order_dir = "ASC" if dir == "asc" else "DESC"
    offset = (page - 1) * page_size
    pos_filter = " AND w.part_of_speech = %s" if pos else ""
    params = (pos,) if pos else ()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {SCHEMA}.word w WHERE w.active{pos_filter}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT w.id, w.lemma, w.part_of_speech, w.definition, d.difficulty, w.rescued_from_reject
                FROM {SCHEMA}.word w
                LEFT JOIN {SCHEMA}.word_difficulty d ON d.word_id = w.id
                WHERE w.active{pos_filter}
                ORDER BY {order_col} {order_dir} NULLS LAST, w.lemma ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    items = [
        WordRow(id=r[0], lemma=r[1], part_of_speech=r[2], definition=r[3], difficulty=r[4],
                 rescued_from_reject=r[5])
        for r in rows
    ]
    return WordPage(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/pos-values", response_model=list[str])
def pos_values() -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT w.part_of_speech FROM {SCHEMA}.word w
                WHERE w.active AND coalesce(w.part_of_speech, '') <> ''
                ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]


@app.delete("/api/words/{word_id}", status_code=204)
def prune_word(word_id: int) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE {SCHEMA}.word SET active = false, updated_at = now() WHERE id = %s",
            (word_id,),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="word not found")
        conn.commit()


REJECTED_SORT_COLUMNS = {
    "lemma": "r.lemma",
    "book": "b.title",
    "reason": "r.reason",
    "count": "r.count",
    "zipf": "r.zipf",
}


class RejectedRow(BaseModel):
    id: int
    lemma: str
    book: str
    reason: str | None
    detail: str | None
    count: int | None
    zipf: float | None


class RejectedPage(BaseModel):
    items: list[RejectedRow]
    total: int
    page: int
    page_size: int


@app.get("/api/rejected", response_model=RejectedPage)
def list_rejected(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort: Literal["lemma", "book", "reason", "count", "zipf"] = "count",
    dir: Literal["asc", "desc"] = "desc",
    book: list[str] = Query([]),
    reason: list[str] = Query([]),
) -> RejectedPage:
    order_col = REJECTED_SORT_COLUMNS[sort]
    order_dir = "ASC" if dir == "asc" else "DESC"
    offset = (page - 1) * page_size

    filters, params = [], []
    if book:
        filters.append("b.title = ANY(%s)")
        params.append(book)
    if reason:
        filters.append("r.reason = ANY(%s)")
        params.append(reason)
    where_extra = "".join(f" AND {f}" for f in filters)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT count(*) FROM {SCHEMA}.rejected_word r
                JOIN {SCHEMA}.book b ON b.id = r.book_id
                WHERE true{where_extra}""",
            params,
        )
        total = cur.fetchone()[0]

        cur.execute(
            f"""SELECT r.id, r.lemma, b.title, r.reason, r.detail, r.count, r.zipf
                FROM {SCHEMA}.rejected_word r
                JOIN {SCHEMA}.book b ON b.id = r.book_id
                WHERE true{where_extra}
                ORDER BY {order_col} {order_dir} NULLS LAST, r.lemma ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    items = [
        RejectedRow(id=r[0], lemma=r[1], book=r[2], reason=r[3], detail=r[4], count=r[5], zipf=r[6])
        for r in rows
    ]
    return RejectedPage(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/rejected/reasons", response_model=list[str])
def rejected_reasons() -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT r.reason FROM {SCHEMA}.rejected_word r
                WHERE r.reason IS NOT NULL ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]


@app.get("/api/rejected/books", response_model=list[str])
def rejected_books() -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT b.title FROM {SCHEMA}.rejected_word r
                JOIN {SCHEMA}.book b ON b.id = r.book_id
                ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]


class AcceptedResult(BaseModel):
    id: int
    lemma: str
    definition: str | None


@app.post("/api/rejected/{rejected_id}/accept", response_model=AcceptedResult)
def accept_rejected(rejected_id: int) -> AcceptedResult:
    """Move a rejected candidate into the accepted word list, as a fully-formed
    incoming term rather than a bare stub: reuses whatever tagger POS/surface
    form/sentence/chapter context the pipeline captured at reject time (so
    dictionary sense-picking gets the same POS hint a normal `ingest` gives
    it), does a live dictionary lookup (rejects are never pre-enriched), then
    upserts into word/word_book against the same book it was rejected from and
    removes the rejected_word row — promoted, not duplicated."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT lemma, book_id, pos, as_seen, sentence, chapter, reason "
            f"FROM {SCHEMA}.rejected_word WHERE id = %s",
            (rejected_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="rejected word not found")
        lemma, book_id, pos, as_seen, sentence, chapter, reason = row

        cand = Candidate(lemma=lemma, pos=pos or "")
        if not localdict.enrich(cand, localdict.build_lexicon(conn, {lemma.lower()})):
            dictionary_enrich(cand)

        cur.execute(
            f"""INSERT INTO {SCHEMA}.word
                (lemma, as_seen, definition, part_of_speech, ipa, sentence,
                 chapter, synonyms, etymology, definition_source, first_added,
                 rescued_from_reject, rescued_at, rescued_reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, CURRENT_DATE, true, now(), %s)
                ON CONFLICT (lemma_lc) DO UPDATE SET
                    definition=COALESCE(NULLIF(EXCLUDED.definition,''), {SCHEMA}.word.definition),
                    part_of_speech=COALESCE(NULLIF(EXCLUDED.part_of_speech,''), {SCHEMA}.word.part_of_speech),
                    ipa=COALESCE(NULLIF(EXCLUDED.ipa,''), {SCHEMA}.word.ipa),
                    sentence=COALESCE(NULLIF(EXCLUDED.sentence,''), {SCHEMA}.word.sentence),
                    chapter=COALESCE(NULLIF(EXCLUDED.chapter,''), {SCHEMA}.word.chapter),
                    etymology=COALESCE(NULLIF(EXCLUDED.etymology,''), {SCHEMA}.word.etymology),
                    definition_source=COALESCE(NULLIF(EXCLUDED.definition_source,''), {SCHEMA}.word.definition_source),
                    active=true, updated_at=now(),
                    rescued_from_reject=true, rescued_at=now(), rescued_reason=EXCLUDED.rescued_reason
                RETURNING id""",
            (lemma, as_seen or lemma, cand.definition, normalize_pos(cand.part_of_speech or pos),
             cand.ipa, sentence or "", chapter or "",
             list(cand.synonyms), cand.etymology, cand.definition_source, reason),
        )
        word_id = cur.fetchone()[0]

        cur.execute(
            f"""INSERT INTO {SCHEMA}.word_book (word_id, book_id) VALUES (%s,%s)
                ON CONFLICT DO NOTHING""",
            (word_id, book_id),
        )
        cur.execute(f"DELETE FROM {SCHEMA}.rejected_word WHERE id = %s", (rejected_id,))
        conn.commit()

    return AcceptedResult(id=word_id, lemma=lemma, definition=cand.definition or None)


class WordSearchResult(BaseModel):
    id: int
    lemma: str


@app.get("/api/words/search", response_model=list[WordSearchResult])
def search_words(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)) -> list[WordSearchResult]:
    """Trigram-similarity lemma search (pg_trgm, already indexed) — the word
    picker for exploring the semantic-distance neighbors of a chosen word."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT id, lemma FROM {SCHEMA}.word
                WHERE active AND similarity(lemma, %s) > 0.1
                ORDER BY similarity(lemma, %s) DESC LIMIT %s""",
            (q, q, limit),
        )
        return [WordSearchResult(id=r[0], lemma=r[1]) for r in cur.fetchall()]


class Neighbor(BaseModel):
    id: int
    lemma: str
    definition: str | None
    part_of_speech: str | None
    distance: float
    shared_usas_field: str | None


class NeighborsResponse(BaseModel):
    word: WordSearchResult
    signal: str
    neighbors: list[Neighbor]


_SIGNAL_COLUMNS = {"definition": "definition_vector", "fasttext": "fasttext_vector"}


@app.get("/api/words/{word_id}/neighbors", response_model=NeighborsResponse)
def word_neighbors(
    word_id: int,
    signal: Literal["definition", "fasttext"] = "definition",
    k: int = Query(10, ge=1, le=50),
    pos: str | None = None,
    quizzable_only: bool = False,
    difficulty_min: float | None = None,
    difficulty_max: float | None = None,
    same_domain_only: bool = False,
    exclude_synonyms: bool = True,
) -> NeighborsResponse:
    """Nearest neighbors of a word by cosine distance on its embedding vector
    (hnsw ANN index — see db.py's word_embedding table), not an all-pairs
    precompute. `signal` picks which vector: 'definition' (meaning, via a
    sentence embedding of the dictionary gloss) or 'fasttext' (word-form
    subwords, works even with no definition). `same_domain_only`/
    `shared_usas_field` are a cheap join against the existing USAS category
    tree — explainability/filtering only, not part of the distance math.
    This is query-time infrastructure for future visualization and distractor
    generation, not the distractor-selection heuristic itself."""
    vec_col = _SIGNAL_COLUMNS[signal]

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT lemma, synonyms FROM {SCHEMA}.word WHERE id = %s AND active", (word_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="word not found")
        lemma, synonyms = row

        cur.execute(
            f"SELECT 1 FROM {SCHEMA}.word_embedding WHERE word_id = %s AND {vec_col} IS NOT NULL",
            (word_id,),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"no {signal} embedding for this word yet")

        # Always fetched (not just under same_domain_only): also powers the
        # shared_usas_field explainability annotation on every neighbor below.
        cur.execute(
            f"""SELECT DISTINCT left(c.code, 1) FROM {SCHEMA}.word_category wc
                JOIN {SCHEMA}.category c ON c.id = wc.category_id
                WHERE wc.word_id = %s""",
            (word_id,),
        )
        target_fields = [r[0] for r in cur.fetchall()]
        if same_domain_only and not target_fields:
            return NeighborsResponse(word=WordSearchResult(id=word_id, lemma=lemma), signal=signal, neighbors=[])

        filters = [f"e2.{vec_col} IS NOT NULL", "w2.active", "w2.id != %s"]
        params: list = [word_id]
        if pos:
            filters.append("w2.part_of_speech = %s")
            params.append(pos)
        if quizzable_only:
            filters.append("wd.quizzable = true")
        if difficulty_min is not None:
            filters.append("wd.difficulty >= %s")
            params.append(difficulty_min)
        if difficulty_max is not None:
            filters.append("wd.difficulty <= %s")
            params.append(difficulty_max)
        if exclude_synonyms and synonyms:
            filters.append("NOT (lower(w2.lemma) = ANY(%s))")
            params.append([s.lower() for s in synonyms])
        if same_domain_only:
            filters.append(
                f"""EXISTS (SELECT 1 FROM {SCHEMA}.word_category wc2
                            JOIN {SCHEMA}.category c2 ON c2.id = wc2.category_id
                            WHERE wc2.word_id = w2.id AND left(c2.code, 1) = ANY(%s))"""
            )
            params.append(target_fields)
        where = " AND ".join(filters)

        cur.execute(
            f"""SELECT w2.id, w2.lemma, w2.definition, w2.part_of_speech,
                       e2.{vec_col} <=> (SELECT {vec_col} FROM {SCHEMA}.word_embedding WHERE word_id = %s) AS distance,
                       (SELECT array_agg(DISTINCT left(c.code, 1)) FROM {SCHEMA}.word_category wc
                        JOIN {SCHEMA}.category c ON c.id = wc.category_id WHERE wc.word_id = w2.id) AS fields
                FROM {SCHEMA}.word_embedding e2
                JOIN {SCHEMA}.word w2 ON w2.id = e2.word_id
                LEFT JOIN {SCHEMA}.word_difficulty wd ON wd.word_id = w2.id
                WHERE {where}
                ORDER BY distance
                LIMIT %s""",
            (word_id, *params, k),
        )
        rows = cur.fetchall()

    target_field_set = set(target_fields)
    neighbors = []
    for wid, wlemma, definition, pos_, distance, fields in rows:
        shared = next(iter(target_field_set & set(fields or [])), None)
        neighbors.append(Neighbor(id=wid, lemma=wlemma, definition=definition, part_of_speech=pos_,
                                   distance=distance, shared_usas_field=shared))

    return NeighborsResponse(word=WordSearchResult(id=word_id, lemma=lemma), signal=signal, neighbors=neighbors)


# Serves the built frontend (webapp/frontend/dist, from `npm run build`) so a
# single port can be exposed publicly. Registered last so it never shadows an
# /api/* route above; absent in plain local dev, where the Vite dev server is
# used instead and this directory doesn't exist.
_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
