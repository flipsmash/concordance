"""User word sets + flashcards API (§ word sets) -- named, per-user
collections of words with a "mastered" flag per (set, word), and a
flashcard deck endpoint that excludes mastered words.

Deliberately much simpler than quiz.py's session model: there's no
scoring/grading step here, just flip-a-card + optionally mark mastered, so
a flashcard "run" is stateless server-side -- GET .../flashcards just
returns the current not-mastered words for a set; the frontend shuffles
and steps through them client-side. No flashcard_session/flashcard_answer
tables exist because there's nothing worth reconstructing on reload that
GET .../flashcards doesn't already give you fresh.

Imports `main` as a module (not `from webapp.backend.main import ...`) and
is registered at the bottom of main.py, same reason/ordering requirement
as quiz.py and browse.py (see their own docstrings) -- this module's own
`from webapp.backend import main as _main` resolves against main.py's
already-populated namespace at word_sets.py's own module-load time.

Every endpoint depends on require_user, not require_viewer/require_admin:
those two can return a null-id identity via the Cloudflare-Access path
(see main.py's own definitions), and every write here is keyed on a real
user_id -- the same reason quiz.py depends on require_user exclusively for
its own per-user writes.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from webapp.backend import main as _main

router = APIRouter()


# --- request/response models -------------------------------------------------

class SetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class SetRenameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class AddWordsRequest(BaseModel):
    word_ids: list[int] = Field(..., min_length=1)


class MasteredUpdate(BaseModel):
    mastered: bool


class SetSummary(BaseModel):
    id: int
    name: str
    created_at: datetime
    word_count: int
    mastered_count: int


class SetItem(BaseModel):
    word_id: int
    lemma: str
    definition: str | None
    mastered: bool


class SetDetail(BaseModel):
    id: int
    name: str
    items: list[SetItem]


class FlashcardItem(BaseModel):
    word_id: int
    lemma: str
    definition: str | None


class FlashcardDeck(BaseModel):
    items: list[FlashcardItem]


# --- helpers -------------------------------------------------------------

def _get_owned_set(conn, set_id: int, user_id: int) -> tuple[int, str]:
    """(id, name) for a set owned by this user, or 404 -- same ownership-
    check idiom as quiz.py's _get_owned_session."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, name FROM {_main.SCHEMA}.word_set WHERE id = %s AND user_id = %s",
            (set_id, user_id),
        )
        row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="set not found")
    return row[0], row[1]


def _set_summary_row(cur, set_id: int) -> SetSummary:
    cur.execute(
        f"""SELECT s.id, s.name, s.created_at,
                   count(i.word_id) AS word_count,
                   count(i.word_id) FILTER (WHERE i.mastered) AS mastered_count
            FROM {_main.SCHEMA}.word_set s
            LEFT JOIN {_main.SCHEMA}.word_set_item i ON i.set_id = s.id
            WHERE s.id = %s
            GROUP BY s.id""",
        (set_id,),
    )
    r = cur.fetchone()
    return SetSummary(id=r[0], name=r[1], created_at=r[2], word_count=r[3], mastered_count=r[4])


# --- routes ----------------------------------------------------------------

@router.get("/api/sets", response_model=list[SetSummary])
def list_sets(user: dict = Depends(_main.require_user)) -> list[SetSummary]:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT s.id, s.name, s.created_at,
                       count(i.word_id) AS word_count,
                       count(i.word_id) FILTER (WHERE i.mastered) AS mastered_count
                FROM {_main.SCHEMA}.word_set s
                LEFT JOIN {_main.SCHEMA}.word_set_item i ON i.set_id = s.id
                WHERE s.user_id = %s
                GROUP BY s.id
                ORDER BY s.created_at DESC""",
            (user["id"],),
        )
        rows = cur.fetchall()
    return [SetSummary(id=r[0], name=r[1], created_at=r[2], word_count=r[3], mastered_count=r[4]) for r in rows]


@router.post("/api/sets", response_model=SetSummary, status_code=201)
def create_set(body: SetCreateRequest, user: dict = Depends(_main.require_user)) -> SetSummary:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT 1 FROM {_main.SCHEMA}.word_set WHERE user_id = %s AND name = %s",
                    (user["id"], body.name))
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="a set with this name already exists")
        cur.execute(
            f"INSERT INTO {_main.SCHEMA}.word_set (user_id, name) VALUES (%s, %s) RETURNING id",
            (user["id"], body.name),
        )
        set_id = cur.fetchone()[0]
        result = _set_summary_row(cur, set_id)
        conn.commit()
    return result


@router.patch("/api/sets/{set_id}", response_model=SetSummary)
def rename_set(set_id: int, body: SetRenameRequest, user: dict = Depends(_main.require_user)) -> SetSummary:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        cur.execute(
            f"SELECT 1 FROM {_main.SCHEMA}.word_set WHERE user_id = %s AND name = %s AND id != %s",
            (user["id"], body.name, set_id),
        )
        if cur.fetchone():
            raise HTTPException(status_code=409, detail="a set with this name already exists")
        cur.execute(f"UPDATE {_main.SCHEMA}.word_set SET name = %s WHERE id = %s", (body.name, set_id))
        result = _set_summary_row(cur, set_id)
        conn.commit()
    return result


@router.delete("/api/sets/{set_id}", status_code=204)
def delete_set(set_id: int, user: dict = Depends(_main.require_user)) -> None:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        cur.execute(f"DELETE FROM {_main.SCHEMA}.word_set WHERE id = %s", (set_id,))
        conn.commit()


@router.get("/api/sets/{set_id}", response_model=SetDetail)
def get_set(set_id: int, user: dict = Depends(_main.require_user)) -> SetDetail:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _, name = _get_owned_set(conn, set_id, user["id"])
        cur.execute(
            f"""SELECT w.id, w.lemma, w.definition, i.mastered
                FROM {_main.SCHEMA}.word_set_item i
                JOIN {_main.SCHEMA}.word w ON w.id = i.word_id
                WHERE i.set_id = %s
                ORDER BY w.lemma_lc ASC""",
            (set_id,),
        )
        items = [SetItem(word_id=r[0], lemma=r[1], definition=r[2], mastered=r[3]) for r in cur.fetchall()]
    return SetDetail(id=set_id, name=name, items=items)


@router.post("/api/sets/{set_id}/words", status_code=204)
def add_words(set_id: int, body: AddWordsRequest, user: dict = Depends(_main.require_user)) -> None:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        # SELECT-filtered insert, not a blind executemany -- silently drops any
        # word_id that doesn't exist/isn't active rather than a raw FK-violation
        # 500, since this list comes straight from a search-result page (Browse's
        # bulk-select or a word-detail button), not free-form user input.
        cur.execute(
            f"""INSERT INTO {_main.SCHEMA}.word_set_item (set_id, word_id)
                SELECT %s, w.id FROM {_main.SCHEMA}.word w WHERE w.id = ANY(%s) AND w.active
                ON CONFLICT (set_id, word_id) DO NOTHING""",
            (set_id, body.word_ids),
        )
        conn.commit()


@router.delete("/api/sets/{set_id}/words/{word_id}", status_code=204)
def remove_word(set_id: int, word_id: int, user: dict = Depends(_main.require_user)) -> None:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        cur.execute(
            f"DELETE FROM {_main.SCHEMA}.word_set_item WHERE set_id = %s AND word_id = %s",
            (set_id, word_id),
        )
        conn.commit()


@router.patch("/api/sets/{set_id}/words/{word_id}", status_code=204)
def set_mastered(set_id: int, word_id: int, body: MasteredUpdate, user: dict = Depends(_main.require_user)) -> None:
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        cur.execute(
            f"""UPDATE {_main.SCHEMA}.word_set_item
                SET mastered = %s, mastered_at = CASE WHEN %s THEN now() ELSE NULL END
                WHERE set_id = %s AND word_id = %s""",
            (body.mastered, body.mastered, set_id, word_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="word not in this set")
        conn.commit()


@router.get("/api/sets/{set_id}/flashcards", response_model=FlashcardDeck)
def get_flashcards(set_id: int, user: dict = Depends(_main.require_user)) -> FlashcardDeck:
    """Every NOT-mastered word in the set -- see module docstring for why
    this is a stateless "current deck" fetch, not a session. The frontend
    shuffles this itself; order here is arbitrary (by word_id)."""
    with _main.get_conn() as conn, conn.cursor() as cur:
        _get_owned_set(conn, set_id, user["id"])
        cur.execute(
            f"""SELECT w.id, w.lemma, w.definition
                FROM {_main.SCHEMA}.word_set_item i
                JOIN {_main.SCHEMA}.word w ON w.id = i.word_id
                WHERE i.set_id = %s AND i.mastered = false
                ORDER BY w.id""",
            (set_id,),
        )
        items = [FlashcardItem(word_id=r[0], lemma=r[1], definition=r[2]) for r in cur.fetchall()]
    return FlashcardDeck(items=items)
