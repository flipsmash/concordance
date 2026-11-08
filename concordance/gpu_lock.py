"""One GPU, one model at a time.

The local 14B model fills most of the card, so two commands loading it at
once (an ingest and a respelling's refresh job, say) run out of memory.
Every long GPU user takes this Postgres advisory lock for as long as it may
load a model; a second one waits for the first to finish. Held on its own
connection, so it lasts exactly as long as the `with` block, whatever the
caller does with its working connection.
"""

from __future__ import annotations

from contextlib import contextmanager

from . import db

_KEY = "concordance-gpu"


@contextmanager
def gpu_lock(database_url: str | None = None, *, on_wait=None):
    """Block until no other command holds the GPU, then hold it. `on_wait`
    is called once if the lock is busy (to tell the user why it's waiting)."""
    conn = db.connect(database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (_KEY,))
            if not cur.fetchone()[0]:
                if on_wait:
                    on_wait()
                cur.execute("SELECT pg_advisory_lock(hashtext(%s))", (_KEY,))
        yield
    finally:
        conn.close()   # ends the session, which releases the lock


_held = []


def hold_for_process(database_url: str | None = None, *, on_wait=None) -> None:
    """Take the GPU lock until this process exits -- for a command (ingest,
    maintain) that may load a model at any point in its run."""
    cm = gpu_lock(database_url, on_wait=on_wait)
    cm.__enter__()
    _held.append(cm)   # keep the connection (and so the lock) alive
