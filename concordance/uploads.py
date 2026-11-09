"""Books an admin uploads from the web app, and the background job that
brings each one into the corpus.

The web app stores the file under uploads/ (outside incoming/, so a manual
`concordance ingest` never grabs one mid-flight) and records it in
book_upload. `concordance process-uploads`, started in its own systemd unit,
then does what a person would after dropping a book into incoming/:

  1. ingest each queued upload (title and author as entered, archived into
     archive/) -- its words are live from here on ("in_corpus");
  2. the upkeep, once for the whole batch: maintain, book-genres,
     author-fame, book-fame, author-stats, book-stats, refresh-rejected-index;
  3. the corpus-wide relatedness rebuild, once: relate, corpus-map.

Each step is its own `concordance` subprocess, so each takes (and frees) the
GPU lock and the GPU memory itself -- this process never holds the lock, or a
child waiting for it would wait forever. A failed upkeep step is noted and
the chain carries on (as scripts/run_maintenance_chain.sh does); a failed
ingest fails that upload only. Uploads queued while the job runs are picked
up before it finishes, so a batch costs one relatedness rebuild, not one per
book. Step logs go to logs/upload_<step>_<time>.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import db
from .db import _safe_schema

_RUNNER_LOCK = "concordance-upload-runner"

UPKEEP = [
    ("maintain", ["maintain"]),
    ("book-genres", ["book-genres", "--only-missing"]),
    ("author-fame", ["author-fame"]),
    ("book-fame", ["book-fame"]),
    ("author-stats", ["author-stats"]),
    ("book-stats", ["book-stats"]),
    ("refresh-rejected-index", ["refresh-rejected-index"]),
]
RELATEDNESS = [("relate", ["relate"]), ("corpus-map", ["corpus-map"])]


def _run(step: str, args: list[str], schema: str, log_dir: Path, log=print,
         database_url: str | None = None) -> tuple[bool, str]:
    """Run one `concordance` command; (succeeded, the end of its output)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"upload_{step}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log(f"{step}…")
    with path.open("w", encoding="utf-8") as f:
        extra = ["--database-url", database_url] if database_url else []
        rc = subprocess.run([sys.executable, "-m", "concordance", *args, "--schema", schema, *extra],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    tail = path.read_text(encoding="utf-8", errors="replace")[-1500:]
    return rc == 0, tail


def _set(conn, s: str, ids: list[int], **cols) -> None:
    if not ids:
        return
    sets = ", ".join(f"{k} = %({k})s" for k in cols)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {s}.book_upload SET {sets} WHERE id = ANY(%(ids)s)", {**cols, "ids": ids})
    conn.commit()


def _note(conn, s: str, ids: list[int], step: str, tail: str) -> None:
    """A failed upkeep step: noted on the uploads, the chain goes on."""
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {s}.book_upload SET error = concat_ws(E'\\n\\n', error, %s::text)
                        WHERE id = ANY(%s)""", (f"{step} failed:\n{tail[-600:]}", ids))
    conn.commit()


def ingest_queued(conn, schema: str, *, archive_dir: Path, log_dir: Path, log=print,
                  database_url: str | None = None) -> list[int]:
    """Ingest every queued upload; returns the ids that made it into the corpus."""
    s = _safe_schema(schema)
    with conn.cursor() as cur:
        cur.execute(f"""UPDATE {s}.book_upload SET status = 'ingesting', step = 'ingest', started_at = now()
                        WHERE status = 'queued' RETURNING id, stored_path, title, author""")
        claimed = cur.fetchall()
    conn.commit()
    done = []
    for uid, path, title, author in claimed:
        args = ["ingest", path, "--title", title, "--archive-dir", str(archive_dir)]
        if author:
            args += ["--author", author]
        ok, tail = _run(f"ingest_{uid}", args, schema, log_dir, log, database_url)
        with conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {s}.book WHERE lower(title) = lower(%s)", (title,))
            row = cur.fetchone()
            book_id = row[0] if row else None
            kept = rejected = None
            if book_id:
                cur.execute(f"SELECT count(*) FROM {s}.word_book WHERE book_id = %s", (book_id,))
                kept = cur.fetchone()[0]
                cur.execute(f"SELECT count(*) FROM {s}.rejected_word WHERE book_id = %s", (book_id,))
                rejected = cur.fetchone()[0]
        conn.commit()
        if ok and book_id:
            _set(conn, s, [uid], status="in_corpus", step=None, book_id=book_id, words_kept=kept,
                 words_rejected=rejected)
            done.append(uid)
        else:
            with conn.cursor() as cur:
                cur.execute(f"""UPDATE {s}.book_upload SET status = 'failed', step = NULL, finished_at = now(),
                                    error = %s WHERE id = %s""", (f"ingest failed:\n{tail}", uid))
            conn.commit()
    return done


def process_uploads(conn, schema: str, *, database_url: str | None = None, archive_dir: Path = Path("archive"),
                    log_dir: Path = Path("logs"), upkeep=UPKEEP, relatedness=RELATEDNESS, log=print) -> dict:
    """The whole job; see the module docstring. Returns {"uploads": n} (0
    when another runner already has it)."""
    s = _safe_schema(schema)
    lock = db.connect(database_url)     # its own session, held for the whole run
    lock.autocommit = True
    try:
        with lock.cursor() as cur:
            # per schema: a test schema's job mustn't wait on the live one (the
            # GPU lock, which its steps take, stays global -- one GPU)
            key = _RUNNER_LOCK if s == db.DEFAULT_SCHEMA else f"{_RUNNER_LOCK}:{s}"
            cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (key,))
            if not cur.fetchone()[0]:
                log("another upload job is running; it will pick these up")
                return {"uploads": 0}
        # a previous runner died mid-way: what it had claimed goes round again
        with conn.cursor() as cur:
            cur.execute(f"UPDATE {s}.book_upload SET status = 'queued', step = NULL WHERE status = 'ingesting'")
            cur.execute(f"SELECT id FROM {s}.book_upload WHERE status IN ('in_corpus', 'upkeep')")
            carried = [r[0] for r in cur.fetchall()]
        conn.commit()

        batch = list(carried)
        while True:
            new = ingest_queued(conn, schema, archive_dir=archive_dir, log_dir=log_dir, log=log,
                                database_url=database_url)
            if not new and not batch:
                return {"uploads": 0}
            batch += new
            for step, args in upkeep:
                _set(conn, s, batch, status="upkeep", step=step)
                ok, tail = _run(step, args, schema, log_dir, log, database_url)
                if not ok:
                    _note(conn, s, batch, step, tail)
            with conn.cursor() as cur:   # more arrived meanwhile: bring them up to here first
                cur.execute(f"SELECT 1 FROM {s}.book_upload WHERE status = 'queued' LIMIT 1")
                more = cur.fetchone()
            if not more:
                break
        for step, args in relatedness:
            _set(conn, s, batch, status="upkeep", step=step)
            ok, tail = _run(step, args, schema, log_dir, log, database_url)
            if not ok:
                _note(conn, s, batch, step, tail)
        with conn.cursor() as cur:
            cur.execute(f"""UPDATE {s}.book_upload SET status = 'done', step = NULL, finished_at = now()
                            WHERE id = ANY(%s)""", (batch,))
        conn.commit()
        return {"uploads": len(batch)}
    finally:
        lock.close()
