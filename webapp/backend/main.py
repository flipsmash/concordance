"""API for the vocab review-and-prune UI (first slice of the larger web app).

Serves the active word list for review and lets the user soft-delete ("prune")
terms that turn out to be too common/easy. Pruned words are never hard-deleted —
`word.active` flips to false so audio/ngram/etc. history stays intact and any
later feature (quizzing, stats) just needs to filter on active=true.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from concordance import db as cdb

app = FastAPI(title="Concordance Review API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "DELETE"],
    allow_headers=["*"],
)

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
            f"""SELECT w.id, w.lemma, w.part_of_speech, w.definition, d.difficulty
                FROM {SCHEMA}.word w
                LEFT JOIN {SCHEMA}.word_difficulty d ON d.word_id = w.id
                WHERE w.active{pos_filter}
                ORDER BY {order_col} {order_dir} NULLS LAST, w.lemma ASC
                LIMIT %s OFFSET %s""",
            (*params, page_size, offset),
        )
        rows = cur.fetchall()

    items = [
        WordRow(id=r[0], lemma=r[1], part_of_speech=r[2], definition=r[3], difficulty=r[4])
        for r in rows
    ]
    return WordPage(items=items, total=total, page=page, page_size=page_size)


@app.get("/api/pos-values", response_model=list[str])
def pos_values() -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT w.part_of_speech FROM {SCHEMA}.word w
                WHERE w.active AND w.part_of_speech IS NOT NULL
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
