"""API for the vocab review-and-prune UI (first slice of the larger web app).

Serves the active word list for review and lets the user soft-delete ("prune")
terms that turn out to be too common/easy. Pruned words are never hard-deleted —
`word.active` flips to false so audio/ngram/etc. history stays intact and any
later feature (quizzing, stats) just needs to filter on active=true.
"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.types import Scope

from concordance import db as cdb
from concordance import localdict
from concordance import usas_domains
from concordance.dictionary import enrich as dictionary_enrich
from concordance.model import Candidate, junk_pos_reason, normalize_pos
from webapp.backend import auth

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


# --- auth dependencies -------------------------------------------------------
# require_viewer/require_admin both accept EITHER an app session or a verified
# Cloudflare Access JWT -- an app session is the reliable mechanism for the
# admin's own browser (Brian logs in once; see the plan's "Admin auth model"),
# while CF Access verification lets an Access-authenticated request through
# even with no app session, so the existing admin UI keeps working with zero
# new login step for paths Cloudflare Access still gates.

def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(auth.SESSION_COOKIE_NAME)
    if not token:
        return None
    with get_conn() as conn:
        return auth.get_session_user(conn, SCHEMA, token)


def require_user(user: dict | None = Depends(get_current_user)) -> dict:
    if user is None:
        raise HTTPException(status_code=401, detail="login required")
    return user


def require_viewer(request: Request, user: dict | None = Depends(get_current_user)) -> dict:
    if user is not None:
        return user
    if auth.verify_cf_access(request) is not None:
        return {"id": None, "username": None, "is_admin": False}
    raise HTTPException(status_code=401, detail="login required")


def require_admin(request: Request, user: dict | None = Depends(get_current_user)) -> dict:
    if user is not None and user["is_admin"]:
        return user
    if auth.verify_cf_access(request) is not None:
        return {"id": None, "username": None, "is_admin": True}
    raise HTTPException(status_code=403, detail="admin access required")


@app.on_event("startup")
def on_startup() -> None:
    with get_conn() as conn:
        cdb.apply_schema(conn, SCHEMA)


class UserOut(BaseModel):
    id: int | None
    username: str | None
    is_admin: bool


class MeResponse(BaseModel):
    user: UserOut | None


class RegisterRequest(BaseModel):
    token: str
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE_NAME, token, httponly=True, secure=True, samesite="lax",
        max_age=int(auth.SESSION_LIFETIME.total_seconds()), path="/",
    )


@app.post("/api/auth/register", response_model=MeResponse)
def register(body: RegisterRequest, response: Response) -> MeResponse:
    """Consumes a one-time invite token (see /api/admin/invites) and creates a
    non-admin account. No email verification -- there's no email sending in
    this app -- so the invite token itself is the only gate on who can sign up."""
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT id FROM {SCHEMA}.invite_tokens
                WHERE token = %s AND used_at IS NULL AND expires_at > now()""",
            (body.token,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="invite link is invalid or expired")
        invite_id = row[0]

        cur.execute(f"SELECT 1 FROM {SCHEMA}.users WHERE username_lc = lower(%s)", (body.username,))
        if cur.fetchone() is not None:
            raise HTTPException(status_code=409, detail="username taken")

        cur.execute(
            f"INSERT INTO {SCHEMA}.users (username, password_hash) VALUES (%s,%s) RETURNING id",
            (body.username, auth.hash_password(body.password)),
        )
        user_id = cur.fetchone()[0]
        cur.execute(
            f"UPDATE {SCHEMA}.invite_tokens SET used_at = now(), used_by_user_id = %s WHERE id = %s",
            (user_id, invite_id),
        )
        conn.commit()

        token, _ = auth.create_session(conn, SCHEMA, user_id)

    _set_session_cookie(response, token)
    return MeResponse(user=UserOut(id=user_id, username=body.username, is_admin=False))


@app.post("/api/auth/login", response_model=MeResponse)
def login(body: LoginRequest, response: Response) -> MeResponse:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, username, password_hash, is_admin FROM {SCHEMA}.users WHERE username_lc = lower(%s)",
            (body.username,),
        )
        row = cur.fetchone()
        # Generic failure message either way -- don't reveal whether the
        # username exists.
        if row is None or not auth.verify_password(body.password, row[2]):
            raise HTTPException(status_code=401, detail="invalid username or password")
        user_id, username, _, is_admin = row

        cur.execute(f"UPDATE {SCHEMA}.users SET last_login_at = now() WHERE id = %s", (user_id,))
        conn.commit()
        token, _ = auth.create_session(conn, SCHEMA, user_id)

    _set_session_cookie(response, token)
    return MeResponse(user=UserOut(id=user_id, username=username, is_admin=is_admin))


@app.post("/api/auth/logout", status_code=204)
def logout(request: Request, response: Response, _: dict = Depends(require_viewer)) -> None:
    token = request.cookies.get(auth.SESSION_COOKIE_NAME)
    if token:
        with get_conn() as conn:
            auth.destroy_session(conn, SCHEMA, token)
    response.delete_cookie(auth.SESSION_COOKIE_NAME, path="/")


@app.get("/api/auth/me", response_model=MeResponse)
def me(user: dict | None = Depends(get_current_user)) -> MeResponse:
    return MeResponse(user=UserOut(**user) if user else None)


class InviteRequest(BaseModel):
    label: str | None = None
    expires_in_days: int = 7


class InviteResponse(BaseModel):
    token: str
    expires_at: datetime
    register_path: str


@app.post("/api/admin/invites", response_model=InviteResponse)
def create_invite(body: InviteRequest, _: dict = Depends(require_admin)) -> InviteResponse:
    token = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(days=body.expires_in_days)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {SCHEMA}.invite_tokens (token, label, expires_at) VALUES (%s,%s,%s)",
            (token, body.label, expires_at),
        )
        conn.commit()
    return InviteResponse(token=token, expires_at=expires_at, register_path=f"/register?token={token}")


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
    _: dict = Depends(require_admin),
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
def pos_values(_: dict = Depends(require_admin)) -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT w.part_of_speech FROM {SCHEMA}.word w
                WHERE w.active AND coalesce(w.part_of_speech, '') <> ''
                ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]


@app.delete("/api/words/{word_id}", status_code=204)
def prune_word(word_id: int, _: dict = Depends(require_admin)) -> None:
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
    _: dict = Depends(require_admin),
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
def rejected_reasons(_: dict = Depends(require_admin)) -> list[str]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT DISTINCT r.reason FROM {SCHEMA}.rejected_word r
                WHERE r.reason IS NOT NULL ORDER BY 1"""
        )
        return [r[0] for r in cur.fetchall()]


@app.get("/api/rejected/books", response_model=list[str])
def rejected_books(_: dict = Depends(require_admin)) -> list[str]:
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
def accept_rejected(rejected_id: int, _: dict = Depends(require_admin)) -> AcceptedResult:
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

        # The dictionary's only resolvable sense for this lemma is a symbol/
        # proper noun (see model.junk_pos_reason, the same gate ingest and
        # the refill/deepen backfills apply) — refuse the rescue rather than
        # silently promoting the exact junk this gate exists to keep out.
        reason = junk_pos_reason(cand.part_of_speech)
        if reason:
            raise HTTPException(
                status_code=422,
                detail=f"cannot rescue {lemma!r}: dictionary resolves it as "
                       f"{normalize_pos(cand.part_of_speech)!r} ({reason.value})",
            )

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
def search_words(
    q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50),
    _: dict = Depends(require_viewer),
) -> list[WordSearchResult]:
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


class WordCategory(BaseModel):
    code: str
    name: str
    is_primary: bool
    confidence: float | None
    color_bucket: str | None  # usas_domains.bucket_for(code[:1]) -- None -> gray chip


class DifficultyFactors(BaseModel):
    zipf: float
    rarity: float
    archaic: float
    domain: float
    morph: float
    why: str


class WordDetail(BaseModel):
    id: int
    lemma: str
    part_of_speech: str | None
    definition: str | None
    ipa: str | None
    sentence: str | None
    chapter: str | None
    synonyms: list[str]
    etymology: str | None
    definition_source: str | None
    first_added: date | None

    zipf: float  # live wordfreq, not stored -- same as /graph
    difficulty: float | None
    difficulty_factors: DifficultyFactors | None
    archaic: str | None
    archaic_evidence: str | None
    archaic_confidence: float | None
    quizzable: bool | None
    quizzable_reason: str | None

    ngram_peak: float | None
    ngram_recent: float | None
    ngram_recency_ratio: float | None
    ngram_peak_year: int | None

    audio_source: str | None  # 'commons'|'azure'|'azure_guess'|'none'|None (no row)

    categories: list[WordCategory]
    books: list[str]


@app.get("/api/words/{word_id}", response_model=WordDetail)
def word_detail(word_id: int, _: dict = Depends(require_viewer)) -> WordDetail:
    """Everything known about one accepted word: definition/IPA/etymology from
    `word`, the composite difficulty + its factor breakdown from
    `word_difficulty`, raw Ngram history from `word_ngram`, which audio source
    (if any) `word_audio` has, every USAS category it carries (not just the
    one /graph picks for a node color), and which book(s) it came from. Three
    round trips rather than one fused query: word_category and word_book are
    both 1:many against word, so joining either into the 1:1
    difficulty/ngram/audio row would fan out into duplicate rows."""
    from wordfreq import zipf_frequency

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""SELECT w.lemma, w.part_of_speech, w.definition, w.ipa, w.sentence, w.chapter,
                       w.synonyms, w.etymology, w.definition_source, w.first_added,
                       d.archaic, d.archaic_evidence, d.archaic_confidence,
                       d.difficulty, d.difficulty_factors, d.quizzable, d.quizzable_reason,
                       n.peak, n.recent, n.recency_ratio, n.peak_year,
                       a.source
                FROM {SCHEMA}.word w
                LEFT JOIN {SCHEMA}.word_difficulty d ON d.word_id = w.id
                LEFT JOIN {SCHEMA}.word_ngram n ON n.word_id = w.id
                LEFT JOIN {SCHEMA}.word_audio a ON a.word_id = w.id
                WHERE w.id = %s AND w.active""",
            (word_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="word not found")
        (lemma, pos, definition, ipa, sentence, chapter, synonyms, etymology, definition_source,
         first_added, archaic, archaic_evidence, archaic_confidence, difficulty, difficulty_factors,
         quizzable, quizzable_reason, ngram_peak, ngram_recent, ngram_recency_ratio, ngram_peak_year,
         audio_source) = row

        cur.execute(
            f"""SELECT c.code, c.name, wc.is_primary, wc.confidence
                FROM {SCHEMA}.word_category wc
                JOIN {SCHEMA}.category c ON c.id = wc.category_id
                WHERE wc.word_id = %s
                ORDER BY wc.is_primary DESC, wc.confidence DESC NULLS LAST, c.code ASC""",
            (word_id,),
        )
        categories = [
            WordCategory(code=code, name=name, is_primary=is_primary, confidence=confidence,
                         color_bucket=usas_domains.bucket_for(code[:1] if code else None))
            for code, name, is_primary, confidence in cur.fetchall()
        ]

        cur.execute(
            f"""SELECT b.title FROM {SCHEMA}.word_book wb
                JOIN {SCHEMA}.book b ON b.id = wb.book_id
                WHERE wb.word_id = %s ORDER BY b.title ASC""",
            (word_id,),
        )
        books = [r[0] for r in cur.fetchall()]

    return WordDetail(
        id=word_id, lemma=lemma, part_of_speech=pos, definition=definition, ipa=ipa,
        sentence=sentence, chapter=chapter, synonyms=synonyms, etymology=etymology,
        definition_source=definition_source, first_added=first_added,
        zipf=zipf_frequency(lemma, "en"), difficulty=difficulty,
        difficulty_factors=DifficultyFactors(**difficulty_factors) if difficulty_factors else None,
        archaic=archaic, archaic_evidence=archaic_evidence, archaic_confidence=archaic_confidence,
        quizzable=quizzable, quizzable_reason=quizzable_reason,
        ngram_peak=ngram_peak, ngram_recent=ngram_recent, ngram_recency_ratio=ngram_recency_ratio,
        ngram_peak_year=ngram_peak_year, audio_source=audio_source,
        categories=categories, books=books,
    )


_AUDIO_ROOT = Path(__file__).resolve().parents[2] / "audio"
# main.py -> parents[2] is the repo root (parents[0]=webapp/backend,
# parents[1]=webapp) -- matches concordance/audio.py's AUDIO_DIR=Path("audio"),
# which is CWD-relative there because the ingest pipeline always runs from repo
# root; the API process's CWD isn't guaranteed, so anchor explicitly.


@app.get("/api/words/{word_id}/audio")
def word_audio(word_id: int, _: dict = Depends(require_viewer)):
    """Streams the mp3 for a word's pronunciation. Looks up the DB-controlled
    file_path rather than exposing the audio/ directory via a raw StaticFiles
    mount, so this route only ever serves what word_audio vouches for."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT source, file_path FROM {SCHEMA}.word_audio WHERE word_id = %s",
            (word_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] == "none" or not row[1]:
        raise HTTPException(status_code=404, detail="no audio for this word")
    _, file_path = row
    # .name strips any directory component -- file_path is pipeline-written,
    # not user input, but this keeps the route from ever resolving outside
    # _AUDIO_ROOT even if that assumption changes later.
    full_path = (_AUDIO_ROOT / Path(file_path).name).resolve()
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="audio file missing on disk")
    return FileResponse(full_path, media_type="audio/mpeg")


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
    _: dict = Depends(require_viewer),
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


class GraphNode(BaseModel):
    id: int
    lemma: str
    definition: str | None
    part_of_speech: str | None
    ring: int                   # 0 = center, 1 = first hop, 2 = second hop
    zipf: float                 # wordfreq.zipf_frequency(lemma, "en") — frontend maps to radius
    usas_code: str | None       # specific top-level code, e.g. "S" — for tooltip text
    usas_name: str | None       # specific name, e.g. "SOCIAL ACTIONS..." — for tooltip text
    color_bucket: str | None    # one of usas_domains.DOMAIN_BUCKETS' keys, or None -> render gray


class GraphEdge(BaseModel):
    source: int
    target: int
    distance: float


class GraphResponse(BaseModel):
    center: WordSearchResult
    signal: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class LegendEntry(BaseModel):
    bucket: str
    name: str


@app.get("/api/graph/legend", response_model=list[LegendEntry])
def graph_legend(_: dict = Depends(require_viewer)) -> list[LegendEntry]:
    """The 6 macro-domain buckets a graph node's color can be — independent of
    any search, so the frontend can show a complete legend immediately (color
    is never the sole identity carrier; a gray "Uncategorized" swatch is a
    client-side-only addition, not a real USAS bucket, so it isn't listed here)."""
    return [LegendEntry(**e) for e in usas_domains.legend_entries()]


@app.get("/api/words/{word_id}/graph", response_model=GraphResponse)
def word_graph(
    word_id: int,
    signal: Literal["definition", "fasttext"] = "definition",
    k1: int = Query(8, ge=3, le=15, description="Center word's direct neighbor count."),
    k2: int = Query(6, ge=0, le=15, description="Each first-hop node's own neighbor count."),
    max_nodes: int = Query(70, ge=10, le=90),
    exclude_synonyms: bool = True,
    _: dict = Depends(require_viewer),
) -> GraphResponse:
    """A multi-hop similarity network around one word: its top-k1 neighbors,
    then each of those gets its own smaller neighbor set too (cross-links kept
    as edges, not duplicate nodes), capped at max_nodes so it reads as an
    actual web rather than a star. Bounded round trips regardless of k1/k2 —
    one query for the center, one for ring-1, one batched query (a LATERAL
    join per ring-1 seed) for all of ring-2 at once, one batched USAS lookup
    over the final node set. Node color is a coarse macro-domain bucket (see
    usas_domains.py); node size is driven by live wordfreq zipf rather than
    the sparse word_difficulty.difficulty column, so every node gets a size
    with no backfill dependency."""
    vec_col = _SIGNAL_COLUMNS[signal]

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT lemma, synonyms, definition, part_of_speech FROM {SCHEMA}.word WHERE id = %s AND active",
            (word_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="word not found")
        center_lemma, synonyms, center_definition, center_pos = row

        cur.execute(
            f"SELECT 1 FROM {SCHEMA}.word_embedding WHERE word_id = %s AND {vec_col} IS NOT NULL",
            (word_id,),
        )
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail=f"no {signal} embedding for this word yet")

        # --- ring 1: center's own top-k1 neighbors (same shape as /neighbors) ---
        filters = [f"e2.{vec_col} IS NOT NULL", "w2.active", "w2.id != %s"]
        params: list = [word_id]
        if exclude_synonyms and synonyms:
            filters.append("NOT (lower(w2.lemma) = ANY(%s))")
            params.append([s.lower() for s in synonyms])
        where = " AND ".join(filters)

        cur.execute(
            f"""SELECT w2.id, w2.lemma, w2.definition, w2.part_of_speech,
                       e2.{vec_col} <=> (SELECT {vec_col} FROM {SCHEMA}.word_embedding WHERE word_id = %s) AS distance
                FROM {SCHEMA}.word_embedding e2
                JOIN {SCHEMA}.word w2 ON w2.id = e2.word_id
                WHERE {where}
                ORDER BY distance
                LIMIT %s""",
            (word_id, *params, k1),
        )
        ring1_rows = cur.fetchall()

        # --- ring 2: one batched query, a LATERAL per ring-1 seed, no vector
        # values ever leave Postgres (each seed's own vector is looked up via
        # a join on its word_id, not passed in from Python) ---
        ring2_rows = []
        seed_ids = [r[0] for r in ring1_rows]
        if k2 and seed_ids:
            cur.execute(
                f"""SELECT seed.word_id AS seed_id, nb.id, nb.lemma, nb.definition, nb.part_of_speech, nb.distance
                    FROM unnest(%s::int[]) AS seed(word_id)
                    JOIN {SCHEMA}.word_embedding e1 ON e1.word_id = seed.word_id
                    CROSS JOIN LATERAL (
                        SELECT w2.id, w2.lemma, w2.definition, w2.part_of_speech,
                               e2.{vec_col} <=> e1.{vec_col} AS distance
                        FROM {SCHEMA}.word_embedding e2
                        JOIN {SCHEMA}.word w2 ON w2.id = e2.word_id
                        WHERE e2.{vec_col} IS NOT NULL AND w2.active
                          AND w2.id != seed.word_id AND w2.id != %s
                        ORDER BY e2.{vec_col} <=> e1.{vec_col}
                        LIMIT %s
                    ) nb""",
                (seed_ids, word_id, k2),
            )
            ring2_rows = cur.fetchall()

        # --- dedup nodes/edges in Python; first-seen ring wins ---
        nodes: dict[int, dict] = {word_id: {"id": word_id, "lemma": center_lemma, "definition": center_definition,
                                             "part_of_speech": center_pos, "ring": 0}}
        edges: dict[tuple[int, int], float] = {}

        def add_edge(a: int, b: int, distance: float) -> None:
            key = (a, b) if a < b else (b, a)
            if key not in edges or distance < edges[key]:
                edges[key] = distance

        for wid, wlemma, definition, pos_, distance in ring1_rows:
            nodes.setdefault(wid, {"id": wid, "lemma": wlemma, "definition": definition,
                                    "part_of_speech": pos_, "ring": 1})
            add_edge(word_id, wid, distance)

        # ring-2-only additions get trimmed first if we're over budget — sort
        # globally by distance so the closest second-hop words survive.
        ring2_new = [r for r in ring2_rows if r[1] not in nodes]
        ring2_new.sort(key=lambda r: r[5])
        budget_left = max_nodes - len(nodes)
        keep_ids = {r[1] for r in ring2_new[: max(budget_left, 0)]}

        for seed_id, wid, wlemma, definition, pos_, distance in ring2_rows:
            if wid in nodes:
                add_edge(seed_id, wid, distance)  # cross-link to an existing node, no new node
            elif wid in keep_ids:
                nodes.setdefault(wid, {"id": wid, "lemma": wlemma, "definition": definition,
                                        "part_of_speech": pos_, "ring": 2})
                add_edge(seed_id, wid, distance)

        # --- batched USAS lookup: one top-level field per node (unlike
        # /neighbors, which aggregates every field a word has for shared-domain
        # explainability, a single node color needs exactly one pick) ---
        node_ids = list(nodes.keys())
        usas_by_word: dict[int, tuple[str, str]] = {}
        if node_ids:
            cur.execute(
                f"""SELECT DISTINCT ON (wc.word_id) wc.word_id, left(c.code, 1), c.name
                    FROM {SCHEMA}.word_category wc
                    JOIN {SCHEMA}.category c ON c.id = wc.category_id
                    WHERE wc.word_id = ANY(%s)
                    ORDER BY wc.word_id, wc.is_primary DESC, wc.confidence DESC NULLS LAST, c.code ASC""",
                (node_ids,),
            )
            usas_by_word = {wid: (code, name) for wid, code, name in cur.fetchall()}

    from wordfreq import zipf_frequency

    result_nodes = []
    for wid, n in nodes.items():
        code, name = usas_by_word.get(wid, (None, None))
        result_nodes.append(GraphNode(
            id=wid, lemma=n["lemma"], definition=n["definition"], part_of_speech=n["part_of_speech"],
            ring=n["ring"], zipf=zipf_frequency(n["lemma"], "en"),
            usas_code=code, usas_name=name, color_bucket=usas_domains.bucket_for(code),
        ))
    result_edges = [GraphEdge(source=a, target=b, distance=d) for (a, b), d in edges.items()]

    return GraphResponse(center=WordSearchResult(id=word_id, lemma=center_lemma), signal=signal,
                          nodes=result_nodes, edges=result_edges)


class SPAStaticFiles(StaticFiles):
    """Falls back to index.html on a 404 so client-side routes (e.g.
    /words/142) survive a hard refresh instead of 404ing against the static
    mount -- react-router only renders those paths client-side, so the server
    has no file at that path to serve directly. StaticFiles.get_response
    raises starlette.exceptions.HTTPException(404) rather than returning a
    404 Response -- the base class, not fastapi.HTTPException (a subclass),
    so the except clause below has to name the base class specifically or it
    silently never matches."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


# Quiz routes live in their own router (quiz.py) rather than growing this
# already-974-line file further. Imported here, at the bottom, deliberately:
# quiz.py does `from webapp.backend import main as _main` and immediately uses
# _main.SCHEMA/_main.require_user/_main.require_admin at its own module-load
# time (route decoration) -- those names must already exist in this module's
# namespace when that happens, i.e. this import must stay below every def
# above it. Moving it earlier breaks the import with an AttributeError.
from webapp.backend import quiz as _quiz  # noqa: E402
app.include_router(_quiz.router)

# Same ordering requirement as quiz.py, same reason -- browse.py's own
# `from webapp.backend import main as _main` resolves against this already-
# populated module namespace at its own module-load time.
from webapp.backend import browse as _browse  # noqa: E402
app.include_router(_browse.router)

# Serves the built frontend (webapp/frontend/dist, from `npm run build`) so a
# single port can be exposed publicly. Registered last so it never shadows an
# /api/* route above; absent in plain local dev, where the Vite dev server is
# used instead and this directory doesn't exist.
_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST_DIR.is_dir():
    app.mount("/", SPAStaticFiles(directory=_DIST_DIR, html=True), name="frontend")
