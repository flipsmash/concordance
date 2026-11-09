"""Admin book upload: POST a .txt/.pdf/.epub with its title and author; the
background job (`concordance process-uploads`, concordance/uploads.py)
ingests it and runs the upkeep. GET lists recent uploads and how far each
has got.

The file is stored under uploads/ by a generated name -- never the
uploader's filename -- outside incoming/ so a hand-run ingest can't pick it
up mid-flight. The title and author travel in the database, not a filename
(titles hold characters an NTFS filename can't).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from webapp.backend import jobs
from webapp.backend import main as _main

router = APIRouter()

UPLOAD_DIR = jobs.REPO / "uploads"
# Under Cloudflare's own request-body cap, so an oversized file gets this
# app's message rather than a bare 413 from the proxy.
MAX_BYTES = 90 * 1024 * 1024
_CHUNK = 1024 * 1024
_MAGIC = {".pdf": b"%PDF", ".epub": b"PK"}


class Upload(BaseModel):
    id: int
    filename: str
    title: str
    author: str | None
    status: str
    step: str | None
    book_id: int | None
    words_kept: int | None
    words_rejected: int | None
    error: str | None
    uploaded_by: str | None
    uploaded_at: datetime
    finished_at: datetime | None


def _check_start(ext: str, head: bytes) -> None:
    if ext in _MAGIC and not head.startswith(_MAGIC[ext]):
        raise HTTPException(status_code=400, detail=f"that doesn't look like a real {ext} file")
    if ext == ".txt" and b"\x00" in head:
        raise HTTPException(status_code=400, detail="that .txt file isn't plain text")


@router.post("/api/admin/uploads", response_model=Upload)
def upload_book(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: str = Form(""),
    replace: bool = Form(False),
    user: dict = Depends(_main.require_admin),
) -> Upload:
    """409 when a book (or a queued upload) already has this title: ingest
    updates a book of the same title in place, so that needs saying yes to
    (replace=true)."""
    title = " ".join(title.split())
    author = " ".join(author.split()) or None
    ext = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext not in (".txt", ".pdf", ".epub"):
        raise HTTPException(status_code=400, detail="upload a .txt, .pdf or .epub file")
    if not title or len(title) > 300:
        raise HTTPException(status_code=400, detail="give the book a title (up to 300 characters)")

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT id, title, author FROM {_main.SCHEMA}.book WHERE lower(title) = lower(%s)""", (title,))
        existing = cur.fetchone()
        cur.execute(f"""SELECT 1 FROM {_main.SCHEMA}.book_upload
                        WHERE lower(title) = lower(%s) AND status IN ('queued', 'ingesting')""", (title,))
        pending = cur.fetchone()
    if pending:
        raise HTTPException(status_code=409, detail={"message": "a book with that title is already waiting to be imported"})
    if existing and not replace:
        raise HTTPException(status_code=409, detail={
            "message": f"“{existing[1]}” is already in the collection",
            "existing": {"id": existing[0], "title": existing[1], "author": existing[2]}})

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with dest.open("wb") as out:
            first = True
            while chunk := file.file.read(_CHUNK):
                if first:
                    _check_start(ext, chunk[:4096])
                    first = False
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail=f"files are limited to {MAX_BYTES // 2**20} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="that file is empty")
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""INSERT INTO {_main.SCHEMA}.book_upload (filename, stored_path, title, author, uploaded_by)
                        VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                    (file.filename, str(dest), title, author, user.get("username") or "admin"))
        uid = cur.fetchone()[0]
        conn.commit()
    jobs.start("process-uploads", _main.SCHEMA)
    return _get(uid)


def _rows(where: str = "", params: tuple = (), limit: int = 30) -> list[Upload]:
    with _main.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""SELECT id, filename, title, author, status, step, book_id, words_kept, words_rejected,
                               error, uploaded_by, uploaded_at, finished_at
                        FROM {_main.SCHEMA}.book_upload {where} ORDER BY id DESC LIMIT {int(limit)}""", params)
        cols = [c.name for c in cur.description]
        return [Upload(**dict(zip(cols, r))) for r in cur.fetchall()]


def _get(uid: int) -> Upload:
    return _rows("WHERE id = %s", (uid,), 1)[0]


@router.get("/api/admin/uploads", response_model=list[Upload])
def list_uploads(_: dict = Depends(_main.require_admin)) -> list[Upload]:
    return _rows()
