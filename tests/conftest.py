"""Run the database tests by default.

DB tests are gated on CONCORDANCE_TEST_DB_URL. When it isn't set, fall back
to the project's own DATABASE_URL (env or .env): every DB test works in its
own throwaway cc_test_* / oed_test_* schema, never the real ones, and
pytest_sessionfinish below drops any of those a test left behind. Skipping
silently let tests go stale unseen (the browse listing tests broke that way). Set CONCORDANCE_SKIP_DB_TESTS=1
to skip them anyway.
"""

from __future__ import annotations

import os

_status = "CONCORDANCE_TEST_DB_URL set explicitly"


def _connectable(url: str) -> bool:
    try:
        import psycopg
        psycopg.connect(url, connect_timeout=3).close()
        return True
    except Exception:  # noqa: BLE001
        return False


if os.environ.get("CONCORDANCE_SKIP_DB_TESTS"):
    os.environ.pop("CONCORDANCE_TEST_DB_URL", None)
    _status = "skipped (CONCORDANCE_SKIP_DB_TESTS)"
elif not os.environ.get("CONCORDANCE_TEST_DB_URL"):
    from concordance import db

    url = db.database_url()
    if url and _connectable(url):
        os.environ["CONCORDANCE_TEST_DB_URL"] = url
        _status = "using DATABASE_URL (throwaway test schemas)"
    else:
        _status = "SKIPPED -- no reachable DATABASE_URL"


def pytest_report_header(config):
    return f"database tests: {_status}"


def pytest_sessionfinish(session, exitstatus):
    url = os.environ.get("CONCORDANCE_TEST_DB_URL")
    if not url:
        return
    try:
        import psycopg
        with psycopg.connect(url, connect_timeout=3) as conn, conn.cursor() as cur:
            # A failed test can leave its connection open holding a lock on its
            # schema; skip that schema rather than wait on it forever.
            cur.execute("SET lock_timeout = '5s'")
            cur.execute(r"""SELECT nspname FROM pg_namespace
                            WHERE nspname LIKE 'cc\_test\_%' OR nspname LIKE 'oed\_test\_%'""")
            for (name,) in cur.fetchall():
                try:
                    cur.execute(f'DROP SCHEMA "{name}" CASCADE')
                    conn.commit()       # one per transaction: a batch exhausts max_locks_per_transaction
                except psycopg.errors.LockNotAvailable:
                    conn.rollback()
    except Exception:  # noqa: BLE001 -- cleanup is best-effort
        pass
